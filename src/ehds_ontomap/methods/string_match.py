"""Exact matching after normalisation.

Adapted from ``ExactMatchGenerator`` in ``app/src/nel/candidates.py`` of
`dt4h-annotation-api <https://github.com/nlp4bia-bsc/dt4h-annotation-api>`_
(MIT License, Copyright (c) 2026 NLP4BIA). Unlike the original, a term shared
by several codes returns all of them instead of the first row only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..normalize import Normalizer
from ..ontology import Ontology
from .base import Matcher, ScoredTriple, TermIndex

__all__ = ["StringMatch"]


class StringMatch(Matcher):
    """Match a query only when it equals an ontology term after normalisation.

    Every match scores ``1.0``. A query that equals a homonym returns one
    candidate per code, all tied at rank 1.

    Parameters
    ----------
    normalizer
        Normalisation applied to queries and ontology terms; defaults to
        lowercasing and whitespace collapsing.
    """

    name = "string-match"

    def __init__(self, normalizer: Normalizer | None = None) -> None:
        self.normalizer = normalizer or Normalizer()
        self._index: TermIndex | None = None

    def fit(self, ontology: Ontology) -> StringMatch:
        """Build the normalised lookup table."""
        self._index = TermIndex(ontology, self.normalizer)
        return self

    def score(
        self,
        queries: Sequence[str],
        *,
        threshold: float | None,
        top_k: int | None,
    ) -> list[list[ScoredTriple]]:
        """Look every query up in the table."""
        if self._index is None:
            raise RuntimeError("call fit() before score()")
        index = self._index
        results: list[list[ScoredTriple]] = []
        for query in queries:
            position = index.positions.get(self.normalizer(query))
            if position is None:
                results.append([])
            else:
                results.append([(code, term, 1.0) for code, term in index.entries[position]])
        return results

    def describe(self) -> dict[str, Any]:
        """Record the normaliser."""
        return {"normalizer": self.normalizer}
