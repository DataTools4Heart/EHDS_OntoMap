"""Mapping results and the rules that turn raw scores into ranked candidates.

Every method produces raw ``(code, term, score)`` triples per query. They all
pass through :func:`select_candidates`, so threshold, top-k, deduplication and
tie handling behave the same whichever method produced the scores.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .normalize import Normalizer
from .terms import TermId

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "Candidate",
    "MappingParams",
    "MappingResult",
    "QueryResult",
    "Ties",
    "select_candidates",
    "validate_selection",
]

Ties: TypeAlias = Literal["keep", "cut"]
"""What happens to candidates tied with the last place inside ``top_k``.

``"keep"`` returns all of them, so the list can be longer than ``top_k``;
``"cut"`` returns exactly ``top_k``, breaking ties by code order.
"""

LONG_COLUMNS = ("query_id", "query", "rank", "code", "term", "score", "method")
BEST_COLUMNS = ("query_id", "query", "code", "term", "score", "n_candidates", "ambiguous", "method")


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ontology concept proposed for one query term.

    Attributes
    ----------
    code
        Ontology code. Each code appears at most once per query.
    term
        The ontology term that produced the score: the best-scoring synonym
        when the code has several.
    score
        Method-specific similarity. Higher is better; scores are **not**
        comparable across methods.
    rank
        One-based competition rank: tied candidates share a rank and the
        next rank is skipped (1, 1, 3).
    """

    code: str
    term: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class QueryResult:
    """All candidates kept for one input term, best first."""

    query_id: TermId
    query: str
    candidates: tuple[Candidate, ...]

    @property
    def matched(self) -> bool:
        """``True`` when at least one candidate passed the threshold."""
        return bool(self.candidates)

    @property
    def best(self) -> Candidate | None:
        """The top candidate, or ``None`` when nothing matched."""
        return self.candidates[0] if self.candidates else None

    def is_ambiguous(self, margin: float = 0.0) -> bool:
        """Whether the runner-up scores within ``margin`` of the best candidate.

        With the default ``margin=0.0`` this is ``True`` only for exact ties,
        such as a term that is a homonym of several codes in the ontology.
        """
        if len(self.candidates) < 2:
            return False
        return self.candidates[0].score - self.candidates[1].score <= margin


@dataclass(frozen=True)
class MappingParams:
    """The settings a mapping was produced with, kept for reproducibility."""

    method: str
    threshold: float | None
    top_k: int | None
    ties: Ties
    ontology: str
    normalizer: Normalizer | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters as plain JSON-serialisable values."""
        return asdict(self)


# ----------------------------------------------------------------------
# Candidate selection, shared by every method
# ----------------------------------------------------------------------


def validate_selection(threshold: float | None, top_k: int | None, ties: str) -> None:
    """Reject invalid selection parameters with a clear message.

    Raises
    ------
    ValueError
        If ``top_k`` is not a positive integer, ``threshold`` is not a finite
        number, or ``ties`` is not ``"keep"`` or ``"cut"``.
    """
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        raise ValueError(f"threshold must be a finite number or None, got {threshold!r}")
    if ties not in ("keep", "cut"):
        raise ValueError(f"ties must be 'keep' or 'cut', got {ties!r}")


def select_candidates(
    scored: Iterable[tuple[str, str, float]],
    *,
    threshold: float | None,
    top_k: int | None,
    ties: Ties,
) -> tuple[Candidate, ...]:
    """Turn raw ``(code, term, score)`` triples into ranked candidates.

    1. Drop triples scoring below ``threshold`` (the threshold is inclusive).
    2. Keep each code once, with its best-scoring term; equal scores are
       resolved by the alphabetically first term, so output is deterministic.
    3. Sort by score descending, then by code.
    4. Keep the first ``top_k``, plus any tied with the last kept one when
       ``ties="keep"``.
    5. Assign competition ranks (1, 1, 3).

    Parameters
    ----------
    scored
        Raw triples from a method, in any order.
    threshold
        Minimum score to keep, or ``None`` to keep everything.
    top_k
        Maximum number of distinct codes, or ``None`` for no limit.
    ties
        ``"keep"`` or ``"cut"``; see :data:`Ties`.
    """
    best_by_code: dict[str, tuple[float, str]] = {}
    for code, term, score in scored:
        score = float(score)
        if threshold is not None and score < threshold:
            continue
        current = best_by_code.get(code)
        if current is None or score > current[0] or (score == current[0] and term < current[1]):
            best_by_code[code] = (score, term)

    ordered = sorted(best_by_code.items(), key=lambda item: (-item[1][0], item[0]))

    if top_k is not None and len(ordered) > top_k:
        keep = top_k
        if ties == "keep":
            cutoff = ordered[top_k - 1][1][0]
            while keep < len(ordered) and ordered[keep][1][0] == cutoff:
                keep += 1
        ordered = ordered[:keep]

    candidates: list[Candidate] = []
    rank = 0
    previous: float | None = None
    for position, (code, (score, term)) in enumerate(ordered, start=1):
        if score != previous:
            rank = position
            previous = score
        candidates.append(Candidate(code=code, term=term, score=score, rank=rank))
    return tuple(candidates)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


class MappingResult:
    """The outcome of mapping a list of terms onto an ontology.

    There is one :class:`QueryResult` per input term, in input order,
    including terms with no candidates: no input is ever dropped.

    Views
    -----
    * :meth:`to_frame` — long table, one row per (query, candidate); queries
      with no candidate appear once with empty candidate columns.
    * :meth:`best` — wide table, one row per query with its top candidate.
    * :meth:`to_records`, :meth:`to_tsv`, :meth:`to_parquet`,
      :meth:`to_jsonl` — the same data for files and other tools.
    """

    def __init__(self, queries: Iterable[QueryResult], params: MappingParams) -> None:
        self._queries = tuple(queries)
        self.params = params

    # -- container protocol ---------------------------------------------

    def __len__(self) -> int:
        return len(self._queries)

    def __iter__(self) -> Iterator[QueryResult]:
        return iter(self._queries)

    def __getitem__(self, index: int) -> QueryResult:
        return self._queries[index]

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"MappingResult(method={self.params.method!r}, queries={summary['n_queries']}, "
            f"matched={summary['n_matched']}, unmatched={summary['n_unmatched']})"
        )

    # -- summaries -------------------------------------------------------

    @property
    def unmatched(self) -> list[QueryResult]:
        """Queries with no candidate above the threshold."""
        return [query for query in self._queries if not query.matched]

    def summary(self, ambiguity_margin: float = 0.0) -> dict[str, int]:
        """Count queries that are matched, unmatched and ambiguous."""
        n_matched = sum(query.matched for query in self._queries)
        return {
            "n_queries": len(self._queries),
            "n_matched": n_matched,
            "n_unmatched": len(self._queries) - n_matched,
            "n_ambiguous": sum(query.is_ambiguous(ambiguity_margin) for query in self._queries),
        }

    # -- tabular views ----------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Long-format rows as dictionaries; see :meth:`to_frame`."""
        method = self.params.method
        rows: list[dict[str, Any]] = []
        for query in self._queries:
            if not query.candidates:
                rows.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "rank": None,
                        "code": None,
                        "term": None,
                        "score": None,
                        "method": method,
                    }
                )
                continue
            for candidate in query.candidates:
                rows.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "rank": candidate.rank,
                        "code": candidate.code,
                        "term": candidate.term,
                        "score": candidate.score,
                        "method": method,
                    }
                )
        return rows

    def to_frame(self) -> pd.DataFrame:
        """Long table: one row per (query, candidate).

        Columns are ``query_id, query, rank, code, term, score, method``.
        Unmatched queries keep one row with null ``rank``, ``code``, ``term``
        and ``score``.
        """
        import pandas as pd

        frame = pd.DataFrame.from_records(self.to_records(), columns=list(LONG_COLUMNS))
        typed: pd.DataFrame = frame.astype(
            {"rank": "Int64", "score": "Float64", "code": "string", "term": "string"}
        )
        return typed

    def best(self, ambiguity_margin: float = 0.0) -> pd.DataFrame:
        """Wide table: one row per query with its top candidate.

        Columns are ``query_id, query, code, term, score, n_candidates,
        ambiguous, method``. ``ambiguous`` is ``True`` when the runner-up
        scores within ``ambiguity_margin`` of the best; inspect
        :meth:`to_frame` for the alternatives.
        """
        import pandas as pd

        rows = []
        for query in self._queries:
            top = query.best
            rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "code": top.code if top else None,
                    "term": top.term if top else None,
                    "score": top.score if top else None,
                    "n_candidates": len(query.candidates),
                    "ambiguous": query.is_ambiguous(ambiguity_margin),
                    "method": self.params.method,
                }
            )
        frame = pd.DataFrame.from_records(rows, columns=list(BEST_COLUMNS))
        typed: pd.DataFrame = frame.astype({"score": "Float64", "code": "string", "term": "string"})
        return typed

    # -- files -------------------------------------------------------------

    def to_tsv(self, path: str | os.PathLike[str]) -> None:
        """Write the long table (:meth:`to_frame`) as UTF-8 TSV."""
        self.to_frame().to_csv(path, sep="\t", index=False, encoding="utf-8")

    def to_parquet(self, path: str | os.PathLike[str]) -> None:
        """Write the long table (:meth:`to_frame`) as Parquet."""
        self.to_frame().to_parquet(path, index=False)

    def to_jsonl(self, path: str | os.PathLike[str]) -> None:
        """Write one JSON object per query, with its candidates nested.

        Each line has ``query_id``, ``query``, ``method`` and ``candidates``
        (a list of ``{rank, code, term, score}``; empty when unmatched).
        """
        with Path(path).open("w", encoding="utf-8") as handle:
            for query in self._queries:
                record = {
                    "query_id": query.query_id,
                    "query": query.query,
                    "method": self.params.method,
                    "candidates": [
                        {"rank": c.rank, "code": c.code, "term": c.term, "score": c.score}
                        for c in query.candidates
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
