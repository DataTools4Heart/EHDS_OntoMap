"""Matching methods.

The bi-encoder is imported lazily so that the lexical methods work without
the optional ``biencoder`` extra (PyTorch, Transformers).
"""

from __future__ import annotations

from typing import Any

from .base import Matcher
from .levenshtein import Levenshtein
from .string_match import StringMatch

__all__ = ["METHODS", "Matcher", "available_methods", "create_matcher"]

METHODS = ("string-match", "levenshtein", "biencoder")


def available_methods() -> tuple[str, ...]:
    """Return the names accepted by ``method=``."""
    return METHODS


def create_matcher(method: str, **options: Any) -> Matcher:
    """Instantiate a matcher by name, forwarding ``options`` to its constructor."""
    if method == "string-match":
        return StringMatch(**options)
    if method == "levenshtein":
        return Levenshtein(**options)
    if method == "biencoder":
        from .biencoder import BiEncoder

        return BiEncoder(**options)
    raise ValueError(f"unknown method {method!r}; choose one of {list(METHODS)}")
