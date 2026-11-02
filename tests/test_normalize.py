from __future__ import annotations

import unicodedata

from ehds_ontomap import Normalizer, normalize_text


def test_defaults_lowercase_and_collapse_whitespace_but_keep_accents() -> None:
    assert normalize_text("  Infarto   Agudo\tde Miocardio ") == "infarto agudo de miocardio"
    assert normalize_text("Cáncer") == "cáncer"


def test_composed_and_decomposed_forms_compare_equal() -> None:
    decomposed = unicodedata.normalize("NFD", "cáncer")
    assert normalize_text(decomposed) == normalize_text("cáncer")


def test_optional_steps() -> None:
    assert normalize_text("Cáncer", strip_accents=True) == "cancer"
    assert normalize_text("covid-19, grave", normalize_punct=True) == "covid 19 grave"
    assert Normalizer(lowercase=False)("IAM") == "IAM"
