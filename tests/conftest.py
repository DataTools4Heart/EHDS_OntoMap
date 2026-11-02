"""Shared fixtures. The default suite is offline and needs no model download."""

from __future__ import annotations

from pathlib import Path

import pytest

ONTOLOGY_ROWS = [
    ("I21", "infarto agudo de miocardio"),
    ("I21", "IAM"),
    ("I50", "insuficiencia cardiaca"),
    ("0012", "NA"),
    ("C1", "frío"),
    ("C2", "frío"),
]


@pytest.fixture
def ontology_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "onto.tsv"
    lines = ["code\tterm", *(f"{code}\t{term}" for code, term in ONTOLOGY_ROWS)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
