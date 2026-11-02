"""Normalised Levenshtein similarity.

Adapted from ``FuzzyMatchMethod`` in ``app/src/nel/fuzzy_match.py`` of
`dt4h-annotation-api <https://github.com/nlp4bia-bsc/dt4h-annotation-api>`_
(MIT License, Copyright (c) 2026 NLP4BIA). The scorer is the same
(:func:`rapidfuzz.distance.Levenshtein.normalized_similarity`); unlike the
original, it returns ranked candidates instead of a single best match and
reports no match as an empty list instead of a ``"NO_MAP"`` code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as _Levenshtein

from ..normalize import Normalizer
from ..ontology import Ontology
from .base import FloatMatrix, Matcher, ScoredTriple, TermIndex, rank_rows

__all__ = ["Levenshtein"]


class Levenshtein(Matcher):
    """Score queries by normalised Levenshtein similarity.

    The score is ``1 - distance / max(len(query), len(term))`` on normalised
    text, between ``0`` and ``1``; an exact match scores ``1.0``.

    Parameters
    ----------
    normalizer
        Normalisation applied to queries and ontology terms; defaults to
        lowercasing and whitespace collapsing.
    workers
        Threads used by RapidFuzz; ``-1`` uses every core.
    """

    name = "levenshtein"

    def __init__(self, normalizer: Normalizer | None = None, workers: int = -1) -> None:
        self.normalizer = normalizer or Normalizer()
        self.workers = workers
        self._index: TermIndex | None = None

    def fit(self, ontology: Ontology) -> Levenshtein:
        """Normalise and deduplicate the ontology terms."""
        self._index = TermIndex(ontology, self.normalizer)
        return self

    def score(
        self,
        queries: Sequence[str],
        *,
        threshold: float | None,
        top_k: int | None,
    ) -> list[list[ScoredTriple]]:
        """Compare every query with every distinct ontology term."""
        if self._index is None:
            raise RuntimeError("call fit() before score()")
        index = self._index

        normalized = [self.normalizer(query) for query in queries]

        def score_block(start: int, stop: int) -> FloatMatrix:
            return process.cdist(
                normalized[start:stop],
                index.keys,
                scorer=_Levenshtein.normalized_similarity,
                dtype=np.float32,
                workers=self.workers,
            )

        return rank_rows(
            len(queries), len(index), index.entries, score_block, threshold=threshold, top_k=top_k
        )

    def describe(self) -> dict[str, Any]:
        """Record the normaliser."""
        return {"normalizer": self.normalizer}
