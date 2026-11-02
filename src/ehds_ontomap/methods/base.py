"""Common interface and shared machinery for the matching methods."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..ontology import Ontology

__all__ = ["Matcher", "ScoredTriple", "TermIndex", "rank_rows"]

ScoredTriple = tuple[str, str, float]
FloatMatrix = npt.NDArray[np.float32]

# Upper bound on the number of cells in one (queries x ontology rows) score
# block: 2**25 float32 cells is 128 MiB, whatever the ontology size.
MAX_BLOCK_CELLS = 2**25


class TermIndex:
    """Ontology rows grouped by a comparison key.

    Several ontology rows may share a key: a homonym (one term, several
    codes) or two spellings that normalise to the same string. Each key is
    scored once, and its score is then attributed to every row behind it.

    Attributes
    ----------
    keys
        Distinct comparison keys, in first-seen order.
    entries
        For each key, the ``(code, original term)`` rows it stands for.
    """

    def __init__(self, ontology: Ontology, key: Callable[[str], str]) -> None:
        positions: dict[str, int] = {}
        self.keys: list[str] = []
        self.entries: list[list[tuple[str, str]]] = []
        for code, term in ontology:
            k = key(term)
            position = positions.get(k)
            if position is None:
                positions[k] = position = len(self.keys)
                self.keys.append(k)
                self.entries.append([])
            self.entries[position].append((code, term))
        self.positions = positions

    def __len__(self) -> int:
        return len(self.keys)


def _shortlist(
    scores: npt.NDArray[np.floating[Any]],
    entries: list[list[tuple[str, str]]],
    threshold: float | None,
    top_k: int | None,
) -> list[ScoredTriple]:
    """Return every row that can still appear in the final selection.

    The result is a superset of what :func:`~ehds_ontomap.result.select_candidates`
    keeps, which makes the final selection exact while only sorting a small
    part of the score row. Rows tied with the ``top_k``-th distinct code are
    included, so ``ties="keep"`` stays correct.
    """
    n = scores.shape[0]
    if top_k is None:
        idx = np.arange(n) if threshold is None else np.flatnonzero(scores >= threshold)
    else:
        m = min(n, max(4 * top_k, 32))
        while True:
            idx = np.arange(n) if m >= n else np.argpartition(-scores, m - 1)[:m]
            if m >= n:
                break
            best: dict[str, float] = {}
            for i in idx:
                score = float(scores[i])
                for code, _ in entries[i]:
                    if score > best.get(code, -math.inf):
                        best[code] = score
            # Every row left out scores <= the lowest score kept. If the k-th
            # best code scores strictly above that, nothing left out can reach
            # the top k or tie with it.
            if len(best) >= top_k:
                kth = sorted(best.values(), reverse=True)[top_k - 1]
                if kth > float(scores[idx].min()):
                    break
            m *= 4
        if threshold is not None:
            idx = idx[scores[idx] >= threshold]

    return [(code, term, float(scores[i])) for i in idx.tolist() for code, term in entries[i]]


def rank_rows(
    n_queries: int,
    index_size: int,
    entries: list[list[tuple[str, str]]],
    score_block: Callable[[int, int], FloatMatrix],
    *,
    threshold: float | None,
    top_k: int | None,
) -> list[list[ScoredTriple]]:
    """Score queries against every index row, block by block.

    ``score_block(start, stop)`` returns the ``(stop - start, index_size)``
    similarity matrix for queries ``start:stop``. Blocks are sized so that
    none exceeds :data:`MAX_BLOCK_CELLS`, keeping memory flat for large
    ontologies.
    """
    step = max(1, MAX_BLOCK_CELLS // max(1, index_size))
    results: list[list[ScoredTriple]] = []
    for start in range(0, n_queries, step):
        block = score_block(start, min(start + step, n_queries))
        for row in block:
            results.append(_shortlist(row, entries, threshold, top_k))
    return results


class Matcher(ABC):
    """A matching method: fitted once on an ontology, then scores queries.

    Subclasses produce raw ``(code, term, score)`` triples. Threshold,
    ``top_k``, deduplication and ranking are applied afterwards by
    :func:`~ehds_ontomap.result.select_candidates`; ``threshold`` and
    ``top_k`` are passed to :meth:`score` only so that a matcher can skip work
    that cannot affect the final selection.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(self, ontology: Ontology) -> Matcher:
        """Prepare the method for ``ontology`` and return ``self``."""

    @abstractmethod
    def score(
        self,
        queries: Sequence[str],
        *,
        threshold: float | None,
        top_k: int | None,
    ) -> list[list[ScoredTriple]]:
        """Return the scored triples for each query, aligned with ``queries``."""

    def describe(self) -> dict[str, Any]:
        """Return the settings to record in :class:`~ehds_ontomap.MappingParams`."""
        return {}
