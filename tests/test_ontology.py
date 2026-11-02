from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ehds_ontomap import Ontology, OntologyError


def test_tsv_keeps_codes_and_terms_as_written(ontology_tsv: Path) -> None:
    onto = Ontology.from_path(ontology_tsv)
    assert len(onto) == 6
    assert onto.name == "onto.tsv"
    # leading zeros and the literal string "NA" survive
    assert ("0012", "NA") in set(onto)


def test_tsv_does_not_interpret_quotes(tmp_path: Path) -> None:
    path = tmp_path / "q.tsv"
    path.write_text('code\tterm\nX1\t"soplo" cardíaco\n', encoding="utf-8")
    assert Ontology.from_path(path).terms == ('"soplo" cardíaco',)


def test_parquet_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "onto.parquet"
    pd.DataFrame({"code": ["A", "B"], "term": ["uno", "dos"]}).to_parquet(path)
    assert Ontology.from_path(path).codes == ("A", "B")


def test_homonyms_and_synonyms_are_kept() -> None:
    frame = pd.DataFrame({"code": ["C1", "C2", "C1"], "term": ["frío", "frío", "cold"]})
    onto = Ontology.from_frame(frame)
    assert len(onto) == 3
    assert onto.n_codes == 2


def test_cleaning_drops_empty_and_duplicate_rows() -> None:
    frame = pd.DataFrame({"code": ["A", "A", None, "B", " "], "term": ["x", "x", "y", " z ", "w"]})
    onto = Ontology.from_frame(frame)
    assert list(onto) == [("A", "x"), ("B", "z")]


def test_missing_column_is_reported() -> None:
    with pytest.raises(OntologyError, match="missing column"):
        Ontology.from_frame(pd.DataFrame({"id": ["A"], "term": ["x"]}))


def test_empty_ontology_is_rejected() -> None:
    with pytest.raises(OntologyError, match="no usable rows"):
        Ontology.from_frame(pd.DataFrame({"code": [], "term": []}))


def test_unknown_suffix_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OntologyError, match="not found"):
        Ontology.from_path(tmp_path / "nope.tsv")
    bad = tmp_path / "onto.xlsx"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(OntologyError, match="unsupported"):
        Ontology.from_path(bad)
