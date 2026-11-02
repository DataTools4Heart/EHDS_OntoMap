"""The main entry point: map terms onto an ontology."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from .methods import Matcher, create_matcher
from .ontology import Ontology
from .result import (
    MappingParams,
    MappingResult,
    QueryResult,
    Ties,
    select_candidates,
    validate_selection,
)
from .terms import TermsLike, load_terms

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["OntologyMapper", "map_terms"]

OntologyLike: TypeAlias = Union[Ontology, str, "os.PathLike[str]", "pd.DataFrame"]


def _as_ontology(ontology: OntologyLike) -> Ontology:
    if isinstance(ontology, Ontology):
        return ontology
    if isinstance(ontology, (str, os.PathLike)):
        return Ontology.from_path(ontology)
    return Ontology.from_frame(ontology)


class OntologyMapper:
    """Map term lists onto one ontology, reusing prepared indexes across calls.

    Preparing a method (normalising the ontology, or embedding it with a
    bi-encoder) is done once per method configuration and reused by later
    calls to :meth:`map`.

    Parameters
    ----------
    ontology
        An :class:`Ontology`, a path to a ``code | term`` TSV or Parquet file,
        or a DataFrame with ``code`` and ``term`` columns.

    Examples
    --------
    >>> mapper = OntologyMapper("snomed_es.tsv")  # doctest: +SKIP
    >>> result = mapper.map(["IAM", "insuf. cardiaca"], method="levenshtein", top_k=3)
    >>> result.best()  # doctest: +SKIP
    """

    def __init__(self, ontology: OntologyLike) -> None:
        self.ontology = _as_ontology(ontology)
        self._matchers: dict[str, Matcher] = {}

    def __repr__(self) -> str:
        return f"OntologyMapper({self.ontology!r})"

    def _matcher(self, method: str | Matcher, options: dict[str, Any]) -> Matcher:
        if isinstance(method, Matcher):
            if options:
                raise TypeError("method options cannot be combined with a Matcher instance")
            key = f"instance:{id(method)}"
            if key not in self._matchers:
                self._matchers[key] = method.fit(self.ontology)
            return self._matchers[key]
        key = f"{method}:{sorted(options.items(), key=lambda item: item[0])!r}"
        if key not in self._matchers:
            self._matchers[key] = create_matcher(method, **options).fit(self.ontology)
        return self._matchers[key]

    def map(
        self,
        terms: TermsLike,
        *,
        method: str | Matcher = "string-match",
        top_k: int | None = 1,
        threshold: float | None = None,
        ties: Ties = "keep",
        column: str | None = None,
        id_column: str | None = None,
        **method_options: Any,
    ) -> MappingResult:
        """Map ``terms`` onto the ontology.

        Parameters
        ----------
        terms
            The terms to map, in any shape accepted by
            :func:`~ehds_ontomap.load_terms`.
        method
            ``"string-match"``, ``"levenshtein"`` or ``"biencoder"``, or a
            configured :class:`~ehds_ontomap.methods.Matcher` instance.
        top_k
            Maximum number of distinct codes per term; ``None`` for all.
        threshold
            Minimum score to keep (inclusive); ``None`` keeps everything.
            Scores are method-specific: ``1.0`` for string-match, ``0``-``1``
            for levenshtein, cosine ``-1``-``1`` for biencoder.
        ties
            ``"keep"`` also returns candidates tied with the last place;
            ``"cut"`` returns exactly ``top_k``.
        column, id_column
            For tabular inputs: the column holding the terms and the one
            holding their IDs.
        **method_options
            Forwarded to the method, e.g. ``normalizer=Normalizer(...)`` for
            the lexical methods, or ``language="es"``, ``model=...``,
            ``device=...``, ``cache_dir=...`` for the bi-encoder.

        Returns
        -------
        MappingResult
            One entry per input term, in input order.
        """
        validate_selection(threshold, top_k, ties)
        term_list = load_terms(terms, column=column, id_column=id_column)
        matcher = self._matcher(method, method_options)

        unique = term_list.unique_terms()
        scored = matcher.score(unique, threshold=threshold, top_k=top_k)
        candidates_by_term = {
            term: select_candidates(triples, threshold=threshold, top_k=top_k, ties=ties)
            for term, triples in zip(unique, scored, strict=True)
        }

        description = matcher.describe()
        params = MappingParams(
            method=matcher.name,
            threshold=threshold,
            top_k=top_k,
            ties=ties,
            ontology=self.ontology.name,
            normalizer=description.get("normalizer"),
            model=description.get("model"),
            extra=description.get("extra", {}),
        )
        queries = (
            QueryResult(query_id=term_id, query=term, candidates=candidates_by_term[term])
            for term_id, term in term_list
        )
        return MappingResult(queries, params)


def map_terms(
    terms: TermsLike,
    ontology: OntologyLike,
    *,
    method: str | Matcher = "string-match",
    top_k: int | None = 1,
    threshold: float | None = None,
    ties: Ties = "keep",
    column: str | None = None,
    id_column: str | None = None,
    **method_options: Any,
) -> MappingResult:
    """Map ``terms`` onto ``ontology`` in one call.

    A shortcut for ``OntologyMapper(ontology).map(terms, ...)``; see
    :meth:`OntologyMapper.map` for the parameters. Use :class:`OntologyMapper`
    directly to map several term lists without re-preparing the ontology.
    """
    return OntologyMapper(ontology).map(
        terms,
        method=method,
        top_k=top_k,
        threshold=threshold,
        ties=ties,
        column=column,
        id_column=id_column,
        **method_options,
    )
