"""Loading and validating the target ontology.

An ontology is a two-column table, ``code`` and ``term``, stored as TSV or
Parquet. One code may have several terms (synonyms) and one term may belong
to several codes (homonyms); both are legitimate and both are kept.
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import OntologyError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Ontology"]

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("code", "term")
_TSV_SUFFIXES = (".tsv", ".tsv.gz", ".tab", ".txt")
_PARQUET_SUFFIXES = (".parquet", ".pq")


def _suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in (*_TSV_SUFFIXES, *_PARQUET_SUFFIXES):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


@dataclass(frozen=True)
class Ontology:
    """An immutable, validated ``code | term`` table.

    Build it with :meth:`from_path` or :meth:`from_frame` rather than the
    constructor, which assumes its input is already clean.

    Attributes
    ----------
    codes
        Concept codes, one per row, always strings (leading zeros preserved).
    terms
        Surface forms, aligned with ``codes``.
    name
        A human-readable identifier, by default the file name.
    """

    codes: tuple[str, ...]
    terms: tuple[str, ...]
    name: str = "ontology"
    source: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if len(self.codes) != len(self.terms):
            raise OntologyError(
                f"codes and terms differ in length ({len(self.codes)} != {len(self.terms)})"
            )
        if not self.codes:
            raise OntologyError(f"ontology {self.name!r} has no usable rows")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | os.PathLike[str], *, name: str | None = None) -> Ontology:
        """Read an ontology from a TSV or Parquet file.

        TSV files are read with every value as a string and with quoting
        disabled, so codes such as ``"0012"`` and terms that contain quote
        characters or the literal text ``"NA"`` survive unchanged.

        Parameters
        ----------
        path
            ``.tsv``, ``.tsv.gz``, ``.tab``, ``.txt``, ``.parquet`` or ``.pq``.
        name
            Identifier recorded in the results; defaults to the file name.

        Raises
        ------
        OntologyError
            If the file does not exist, has an unknown extension, or lacks
            the ``code`` and ``term`` columns.
        """
        import pandas as pd

        path = Path(path)
        if not path.is_file():
            raise OntologyError(f"ontology file not found: {path}")

        suffix = _suffix(path)
        if suffix in _TSV_SUFFIXES:
            frame = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
            )
        elif suffix in _PARQUET_SUFFIXES:
            frame = pd.read_parquet(path)
        else:
            raise OntologyError(
                f"unsupported ontology format {suffix!r} for {path}; "
                f"expected one of {_TSV_SUFFIXES + _PARQUET_SUFFIXES}"
            )
        return cls.from_frame(frame, name=name or path.name, source=str(path))

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        name: str = "ontology",
        source: str | None = None,
    ) -> Ontology:
        """Build an ontology from a DataFrame with ``code`` and ``term`` columns.

        Values are converted to stripped strings. Rows with an empty code or
        term are dropped, as are exact duplicate ``(code, term)`` rows; both
        are reported through :mod:`logging`. Extra columns are ignored with a
        warning.
        """
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise OntologyError(
                f"ontology {name!r} is missing column(s) {missing}; "
                f"found {list(map(str, frame.columns))}. Expected exactly 'code' and 'term'."
            )
        extra = [str(column) for column in frame.columns if column not in REQUIRED_COLUMNS]
        if extra:
            logger.warning("ontology %r: ignoring extra column(s) %s", name, extra)

        table = frame.loc[:, list(REQUIRED_COLUMNS)]
        table = table.dropna()
        table = table.astype(str).apply(lambda column: column.str.strip())
        table = table[(table["code"] != "") & (table["term"] != "")]
        n_empty = len(frame) - len(table)
        if n_empty:
            logger.warning(
                "ontology %r: dropped %d row(s) with an empty code or term", name, n_empty
            )

        n_before = len(table)
        table = table.drop_duplicates()
        if n_before - len(table):
            logger.info(
                "ontology %r: dropped %d duplicate (code, term) row(s)", name, n_before - len(table)
            )

        return cls(
            codes=tuple(table["code"].tolist()),
            terms=tuple(table["term"].tolist()),
            name=name,
            source=source,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.codes)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return zip(self.codes, self.terms, strict=True)

    @property
    def n_codes(self) -> int:
        """Number of distinct codes."""
        return len(set(self.codes))

    def to_frame(self) -> pd.DataFrame:
        """Return the ontology as a ``code | term`` DataFrame."""
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame({"code": list(self.codes), "term": list(self.terms)})
        return frame

    def __repr__(self) -> str:
        return f"Ontology(name={self.name!r}, rows={len(self)}, codes={self.n_codes})"
