from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehds_ontomap import MappingParams, MappingResult, QueryResult
from ehds_ontomap.result import select_candidates, validate_selection

SCORED = [
    ("A", "a1", 0.9),
    ("A", "a2", 0.95),  # better synonym of A
    ("B", "b", 0.8),
    ("C", "c", 0.8),
    ("D", "d", 0.5),
]


def test_dedup_keeps_best_synonym() -> None:
    cands = select_candidates(SCORED, threshold=None, top_k=None, ties="keep")
    assert (cands[0].code, cands[0].term) == ("A", "a2")
    assert len({c.code for c in cands}) == len(cands) == 4


def test_threshold_is_inclusive() -> None:
    cands = select_candidates(SCORED, threshold=0.8, top_k=None, ties="keep")
    assert [c.code for c in cands] == ["A", "B", "C"]


def test_competition_ranking() -> None:
    cands = select_candidates(SCORED, threshold=None, top_k=None, ties="keep")
    assert [c.rank for c in cands] == [1, 2, 2, 4]


def test_ties_keep_extends_past_top_k() -> None:
    kept = select_candidates(SCORED, threshold=None, top_k=2, ties="keep")
    assert [c.code for c in kept] == ["A", "B", "C"]
    cut = select_candidates(SCORED, threshold=None, top_k=2, ties="cut")
    assert [c.code for c in cut] == ["A", "B"]


def test_equal_score_synonyms_resolve_deterministically() -> None:
    cands = select_candidates(
        [("A", "zz", 1.0), ("A", "aa", 1.0)], threshold=None, top_k=1, ties="cut"
    )
    assert cands[0].term == "aa"


@pytest.mark.parametrize(
    ("threshold", "top_k", "ties"),
    [(None, 0, "keep"), (None, True, "keep"), (float("nan"), None, "keep"), (None, None, "all")],
)
def test_invalid_selection(threshold: float | None, top_k: int | None, ties: str) -> None:
    with pytest.raises(ValueError, match="must be"):
        validate_selection(threshold, top_k, ties)


def _result() -> MappingResult:
    homonym = select_candidates(
        [("C1", "frío", 1.0), ("C2", "frío", 1.0)], threshold=None, top_k=1, ties="keep"
    )
    clear = select_candidates([("I21", "iam", 1.0)], threshold=None, top_k=1, ties="keep")
    queries = [
        QueryResult(query_id=0, query="frío", candidates=homonym),
        QueryResult(query_id=1, query="IAM", candidates=clear),
        QueryResult(query_id=2, query="xyz", candidates=()),
    ]
    params = MappingParams(
        method="string-match", threshold=None, top_k=1, ties="keep", ontology="t"
    )
    return MappingResult(queries, params)


def test_long_frame_keeps_unmatched_rows() -> None:
    frame = _result().to_frame()
    assert list(frame["query_id"]) == [0, 0, 1, 2]
    unmatched = frame[frame["query_id"] == 2].iloc[0]
    assert frame["code"].isna().sum() == 1
    assert unmatched["query"] == "xyz"


def test_best_view_flags_homonyms() -> None:
    best = _result().best()
    assert list(best["ambiguous"]) == [True, False, False]
    assert list(best["n_candidates"]) == [2, 1, 0]
    assert best["code"].iloc[0] == "C1"


def test_summary() -> None:
    assert _result().summary() == {
        "n_queries": 3,
        "n_matched": 2,
        "n_unmatched": 1,
        "n_ambiguous": 1,
    }


def test_file_exports(tmp_path: Path) -> None:
    result = _result()
    result.to_tsv(tmp_path / "r.tsv")
    result.to_parquet(tmp_path / "r.parquet")
    result.to_jsonl(tmp_path / "r.jsonl")
    lines = (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()
    assert [len(json.loads(line)["candidates"]) for line in lines] == [2, 1, 0]
    assert json.loads(lines[0])["query"] == "frío"
