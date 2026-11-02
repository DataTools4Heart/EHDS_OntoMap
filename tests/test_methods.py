from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehds_ontomap import Normalizer, Ontology, OntologyMapper, map_terms
from ehds_ontomap.methods.base import _shortlist
from ehds_ontomap.methods.biencoder import DEFAULT_MODELS, BiEncoder
from ehds_ontomap.result import select_candidates

ROWS = {
    "code": ["I21", "I21", "I50", "C1", "C2", "R07"],
    "term": [
        "Infarto agudo de miocardio",
        "IAM",
        "insuficiencia cardiaca",
        "frío",
        "frío",
        "dolor",
    ],
}


@pytest.fixture
def mapper() -> OntologyMapper:
    return OntologyMapper(pd.DataFrame(ROWS))


class TrigramEncoder:
    """Deterministic, offline stand-in for a bi-encoder."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(len(texts))
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - 2):
                vectors[row, hash(padded[i : i + 3]) % 256] += 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# -- string-match -------------------------------------------------------------


def test_string_match_normalises_and_keeps_original_term(mapper: OntologyMapper) -> None:
    best = mapper.map(["  iam ", "infarto AGUDO de miocardio"]).best()
    assert list(best["code"]) == ["I21", "I21"]
    assert best["term"].iloc[0] == "IAM"
    assert list(best["score"]) == [1.0, 1.0]


def test_string_match_returns_every_homonym(mapper: OntologyMapper) -> None:
    result = mapper.map(["Frío"])
    assert [c.code for c in result[0].candidates] == ["C1", "C2"]
    assert [c.rank for c in result[0].candidates] == [1, 1]
    assert result[0].is_ambiguous()


def test_string_match_accent_folding_is_opt_in(mapper: OntologyMapper) -> None:
    assert not mapper.map(["frio"])[0].matched
    folded = mapper.map(["frio"], normalizer=Normalizer(strip_accents=True))
    assert folded[0].matched


# -- levenshtein --------------------------------------------------------------


def test_levenshtein_ranks_close_terms(mapper: OntologyMapper) -> None:
    result = mapper.map(["insuficiencia cardiaka"], method="levenshtein", top_k=2)
    best = result[0].best
    assert best is not None
    assert best.code == "I50"
    assert 0.9 < best.score < 1.0
    assert len(result[0].candidates) == 2


def test_levenshtein_threshold_leaves_unmatched_rows(mapper: OntologyMapper) -> None:
    result = mapper.map(["zzzz", "dolr"], method="levenshtein", threshold=0.7)
    frame = result.to_frame()
    assert frame["code"].isna().tolist() == [True, False]
    assert result.summary()["n_unmatched"] == 1


def test_levenshtein_exact_match_scores_one(mapper: OntologyMapper) -> None:
    assert mapper.map(["IAM"], method="levenshtein")[0].best.score == 1.0  # type: ignore[union-attr]


# -- shared selection ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_shortlist_matches_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 300)
    codes = [f"C{rng.randint(0, n // 2)}" for _ in range(n)]
    entries = [[(code, f"t{i}")] for i, code in enumerate(codes)]
    scores = np.array([rng.randint(0, 10) / 10 for _ in range(n)], dtype=np.float32)
    top_k = rng.choice([1, 2, 5, None])
    threshold = rng.choice([None, 0.3])
    brute = [(code, term, float(scores[i])) for i, row in enumerate(entries) for code, term in row]
    for ties in ("keep", "cut"):
        expected = select_candidates(brute, threshold=threshold, top_k=top_k, ties=ties)
        fast = select_candidates(
            _shortlist(scores, entries, threshold, top_k),
            threshold=threshold,
            top_k=top_k,
            ties=ties,
        )
        assert fast == expected


# -- biencoder ----------------------------------------------------------------


def test_biencoder_with_injected_encoder(mapper: OntologyMapper) -> None:
    encoder = TrigramEncoder()
    result = mapper.map(["insuficiencia cardíaca"], method="biencoder", encoder=encoder, top_k=3)
    assert result[0].best.code == "I50"  # type: ignore[union-attr]
    assert result.params.method == "biencoder"
    assert result.params.normalizer is None


def test_biencoder_needs_language_or_model() -> None:
    with pytest.raises(ValueError, match="language"):
        BiEncoder()
    with pytest.raises(ValueError, match="no default model"):
        BiEncoder(language="xx")
    assert BiEncoder(language="es").model == DEFAULT_MODELS["es"] == "ICB-UMA/ClinLinker-KB-P"


def test_biencoder_embedding_cache(tmp_path: Path) -> None:
    onto = Ontology.from_frame(pd.DataFrame(ROWS))
    first = TrigramEncoder()
    BiEncoder("stub", encoder=first, cache_dir=tmp_path).fit(onto)
    second = TrigramEncoder()
    BiEncoder("stub", encoder=second, cache_dir=tmp_path).fit(onto)
    assert first.calls == [5]  # 5 distinct terms embedded once...
    assert second.calls == []  # ...and reloaded from disk the second time
    other = TrigramEncoder()
    BiEncoder("stub", encoder=other, cache_dir=tmp_path, max_length=64).fit(onto)
    assert other.calls == [5]  # different settings, different cache entry


# -- mapper -------------------------------------------------------------------


def test_duplicates_are_scored_once_but_reported_per_input(mapper: OntologyMapper) -> None:
    encoder = TrigramEncoder()
    result = mapper.map(
        {"a": "dolor", "b": "dolor", "c": "IAM"}, method="biencoder", encoder=encoder
    )
    assert encoder.calls[-1] == 2
    assert [q.query_id for q in result] == ["a", "b", "c"]


def test_prepared_methods_are_reused(mapper: OntologyMapper) -> None:
    mapper.map(["IAM"], method="levenshtein")
    mapper.map(["dolor"], method="levenshtein")
    mapper.map(["dolor"], method="levenshtein", normalizer=Normalizer(strip_accents=True))
    assert len(mapper._matchers) == 2


def test_unknown_method_and_bad_params(mapper: OntologyMapper) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        mapper.map(["x"], method="bm25")
    with pytest.raises(ValueError, match="top_k"):
        mapper.map(["x"], top_k=0)


def test_map_terms_from_files(ontology_tsv: Path, tmp_path: Path) -> None:
    terms = tmp_path / "terms.txt"
    terms.write_text("IAM\nfrío\nnada\n", encoding="utf-8")
    result = map_terms(terms, ontology_tsv)
    assert result.summary() == {"n_queries": 3, "n_matched": 2, "n_unmatched": 1, "n_ambiguous": 1}
    assert result.params.ontology == "onto.tsv"


# -- real models (network, large download) ------------------------------------


@pytest.mark.model
def test_clinlinker_end_to_end(mapper: OntologyMapper) -> None:
    result = mapper.map(["infarto de miocardio"], method="biencoder", language="es", device="cpu")
    assert result[0].best.code == "I21"  # type: ignore[union-attr]
