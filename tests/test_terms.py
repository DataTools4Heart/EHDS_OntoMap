from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ehds_ontomap import InputError, TermList, load_terms


def test_list_gets_positional_ids() -> None:
    terms = load_terms(["a", "b"])
    assert terms.ids == (0, 1)
    assert terms.terms == ("a", "b")


def test_generator_and_tuple() -> None:
    assert load_terms(t for t in ("a", "b")).terms == ("a", "b")
    assert load_terms(("a",)).terms == ("a",)


def test_mapping_keeps_ids() -> None:
    assert list(load_terms({"p1": "a", "p2": "b"})) == [("p1", "a"), ("p2", "b")]


def test_series_uses_index() -> None:
    terms = load_terms(pd.Series(["a", "b"], index=["x", "y"]))
    assert terms.ids == ("x", "y")


def test_dataframe_needs_column_when_ambiguous() -> None:
    frame = pd.DataFrame({"id": ["r1", "r2"], "text": ["a", "b"]})
    with pytest.raises(InputError, match="pass column"):
        load_terms(frame)
    terms = load_terms(frame, column="text", id_column="id")
    assert list(terms) == [("r1", "a"), ("r2", "b")]


def test_missing_values_are_reported_with_their_id() -> None:
    with pytest.raises(InputError, match="not strings"):
        load_terms(pd.DataFrame({"text": ["a", None]}))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(InputError, match="unique"):
        TermList(ids=(1, 1), terms=("a", "b"))


def test_bare_string_is_not_silently_a_term() -> None:
    with pytest.raises(InputError, match="pass a list"):
        load_terms("infarto")


def test_unique_terms_preserve_order() -> None:
    assert load_terms(["b", "a", "b"]).unique_terms() == ["b", "a"]


@pytest.mark.parametrize("suffix", [".txt", ".csv", ".tsv", ".parquet", ".json", ".jsonl"])
def test_file_formats(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"terms{suffix}"
    values = ["infarto", "dolor torácico"]
    if suffix == ".txt":
        path.write_text("infarto\n\ndolor torácico\n", encoding="utf-8")
    elif suffix in (".csv", ".tsv"):
        pd.DataFrame({"term": values}).to_csv(
            path, sep="," if suffix == ".csv" else "\t", index=False
        )
    elif suffix == ".parquet":
        pd.DataFrame({"term": values}).to_parquet(path)
    elif suffix == ".json":
        path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text("\n".join(json.dumps({"term": v}) for v in values), encoding="utf-8")

    assert load_terms(path).terms == tuple(values)
    assert load_terms(str(path)).terms == tuple(values)


def test_json_object_maps_ids(tmp_path: Path) -> None:
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"k1": "a"}), encoding="utf-8")
    assert list(load_terms(path)) == [("k1", "a")]
