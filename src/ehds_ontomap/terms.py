"""Accepting the terms to map in as many shapes as is reasonable.

Every accepted shape is converted to a :class:`TermList`: a sequence of
``(id, term)`` pairs. IDs come from the input when it has them (dict keys, a
Series index, an ID column) and are positions ``0..n-1`` otherwise, so every
result can be traced back to the input row it came from.
"""

from __future__ import annotations

import json
import os
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, Union, cast

from .errors import InputError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TermList", "TermsLike", "load_terms"]

TermId: TypeAlias = Hashable
TermsLike: TypeAlias = Union[
    "TermList",
    "os.PathLike[str]",
    str,
    Mapping[Any, str],
    Iterable[str],
    "pd.Series",
    "pd.DataFrame",
]

_TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t", ".tab": "\t"}


@dataclass(frozen=True)
class TermList:
    """Validated query terms with stable, unique identifiers."""

    ids: tuple[TermId, ...]
    terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.terms):
            raise InputError(
                f"ids and terms differ in length ({len(self.ids)} != {len(self.terms)})"
            )
        seen: set[TermId] = set()
        duplicates: list[TermId] = []
        for term_id in self.ids:
            if term_id in seen:
                duplicates.append(term_id)
            seen.add(term_id)
        if duplicates:
            raise InputError(f"term ids must be unique; duplicated: {duplicates[:5]}")
        bad = [
            (term_id, term)
            for term_id, term in zip(self.ids, self.terms, strict=True)
            if not isinstance(term, str)
        ]
        if bad:
            shown = ", ".join(f"{term_id!r}: {term!r}" for term_id, term in bad[:5])
            raise InputError(
                f"{len(bad)} term(s) are not strings (missing values?); first: {shown}"
            )

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[tuple[TermId, str]]:
        return zip(self.ids, self.terms, strict=True)

    def unique_terms(self) -> list[str]:
        """Distinct terms in first-seen order, so each is matched only once."""
        return list(dict.fromkeys(self.terms))


def _from_pairs(pairs: Iterable[tuple[TermId, Any]]) -> TermList:
    ids: list[TermId] = []
    terms: list[Any] = []
    for term_id, term in pairs:
        ids.append(term_id)
        terms.append(term)
    return TermList(ids=tuple(ids), terms=tuple(terms))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _from_frame(frame: pd.DataFrame, column: str | None, id_column: str | None) -> TermList:
    if column is None:
        if frame.shape[1] != 1:
            raise InputError(
                f"the table has {frame.shape[1]} columns {list(frame.columns)}; "
                "pass column=... to choose the one holding the terms"
            )
        column = frame.columns[0]
    if column not in frame.columns:
        raise InputError(f"column {column!r} not found; available: {list(frame.columns)}")
    if id_column is not None and id_column not in frame.columns:
        raise InputError(f"id_column {id_column!r} not found; available: {list(frame.columns)}")

    ids = frame[id_column].tolist() if id_column is not None else frame.index.tolist()
    terms = [None if _is_missing(value) else value for value in frame[column].tolist()]
    return _from_pairs(zip(ids, terms, strict=True))


def _read_json_records(records: Any, column: str | None, id_column: str | None) -> TermList:
    if isinstance(records, Mapping):
        return _from_pairs(records.items())
    if not isinstance(records, list):
        raise InputError(
            "JSON input must be a list of terms, a list of objects or an id→term object"
        )
    if all(isinstance(record, str) for record in records):
        return _from_pairs(enumerate(records))
    if all(isinstance(record, Mapping) for record in records):
        import pandas as pd

        return _from_frame(pd.DataFrame.from_records(records), column, id_column)
    raise InputError("JSON list must contain only strings or only objects")


def _from_path(path: Path, column: str | None, id_column: str | None) -> TermList:
    import pandas as pd

    if not path.is_file():
        raise InputError(f"terms file not found: {path}")
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open(encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        return _from_pairs(enumerate(line for line in lines if line))
    if suffix in _TABLE_SUFFIXES:
        frame = pd.read_csv(
            path,
            sep=_TABLE_SUFFIXES[suffix],
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8",
        )
        return _from_frame(frame, column, id_column)
    if suffix in {".parquet", ".pq"}:
        return _from_frame(pd.read_parquet(path), column, id_column)
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return _read_json_records(json.load(handle), column, id_column)
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return _read_json_records(records, column, id_column)
    raise InputError(
        f"unsupported terms file format {suffix!r}; "
        "expected .txt, .csv, .tsv, .tab, .parquet, .pq, .json or .jsonl"
    )


def load_terms(
    source: TermsLike,
    *,
    column: str | None = None,
    id_column: str | None = None,
) -> TermList:
    """Convert any supported input into a :class:`TermList`.

    Supported inputs:

    * a list, tuple or other iterable of strings: IDs are ``0..n-1``;
    * a mapping ``{id: term}``;
    * a :class:`pandas.Series` (IDs from the index) or :class:`pandas.DataFrame`
      (``column`` holds the terms, IDs from ``id_column`` or the index);
    * any table object with a ``to_pandas()`` method (Polars, PyArrow);
    * a path (:class:`pathlib.Path`, or a ``str`` naming an existing file) to a
      ``.txt`` file with one term per line, a ``.csv``/``.tsv``/``.parquet``
      table, a ``.json`` file (list of strings, list of objects or id→term
      object) or a ``.jsonl`` file (one string or object per line).

    A bare ``str`` that is not an existing file is rejected rather than
    silently treated as one term: wrap it in a list.

    Parameters
    ----------
    source
        The terms, in any of the shapes above.
    column
        Column holding the terms, for tabular inputs with more than one column.
    id_column
        Column holding the IDs, for tabular inputs.

    Raises
    ------
    InputError
        If the input shape is not recognised, a column is missing, an entry
        is not a string, or IDs are duplicated.
    """
    if isinstance(source, TermList):
        return source
    if isinstance(source, str):
        if Path(source).is_file():
            return _from_path(Path(source), column, id_column)
        raise InputError(
            f"{source!r} is not an existing file. To map a single term, pass a list: [{source!r}]"
        )
    if isinstance(source, os.PathLike):
        return _from_path(Path(source), column, id_column)
    if isinstance(source, Mapping):
        return _from_pairs(source.items())

    import pandas as pd

    if isinstance(source, pd.DataFrame):
        return _from_frame(source, column, id_column)
    if isinstance(source, pd.Series):
        return _from_frame(source.to_frame(name="term"), "term", None)
    to_pandas = getattr(source, "to_pandas", None)
    if callable(to_pandas):
        converted = to_pandas()
        if isinstance(converted, pd.Series):
            converted = converted.to_frame(name="term")
            column = "term"
        return _from_frame(converted, column, id_column)
    if isinstance(cast(object, source), (bytes, bytearray)):
        raise InputError("bytes are not supported; decode them to str first")
    if isinstance(source, Iterable):
        return _from_pairs(enumerate(source))
    raise InputError(f"cannot read terms from an object of type {type(source).__name__}")
