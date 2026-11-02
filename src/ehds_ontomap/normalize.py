"""Surface-form normalisation applied by the lexical methods.

Adapted from ``app/src/nel/normalize.py`` in
`dt4h-annotation-api <https://github.com/nlp4bia-bsc/dt4h-annotation-api>`_
(MIT License, Copyright (c) 2026 NLP4BIA).

Normalisation is applied to both the query terms and the ontology terms by the
``string-match`` and ``levenshtein`` methods. It is **not** applied to the
``biencoder`` method: the DT4H encoders were trained on cased terms, and
changing their input changes every embedding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["Normalizer", "normalize_text"]

_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_accents: bool = False,
    normalize_punct: bool = False,
) -> str:
    """Normalise a surface form for lexical comparison.

    The text is always converted to Unicode NFC, so that a precomposed ``"é"``
    and an ``"e"`` followed by a combining accent compare equal, and runs of
    whitespace are collapsed to single spaces.

    Parameters
    ----------
    text
        Input surface form.
    lowercase
        Fold to lowercase.
    strip_accents
        Drop combining marks, so ``"cáncer"`` matches ``"cancer"``. Off by
        default: in Spanish and other languages an accent can be the only
        difference between two distinct words.
    normalize_punct
        Replace punctuation with spaces. Off by default because it destroys
        forms such as ``"covid-19"`` or ``"5-fu"``.

    Returns
    -------
    str
        The normalised text.
    """
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = " ".join(normalized.split())
    if lowercase:
        normalized = normalized.lower()
    if strip_accents:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(character) != "Mn"
        )
        normalized = unicodedata.normalize("NFC", normalized)
    if normalize_punct:
        normalized = " ".join(_PUNCT.sub(" ", normalized).split())
    return normalized


@dataclass(frozen=True, slots=True)
class Normalizer:
    """A reusable, recorded normalisation configuration.

    The defaults (lowercase, collapse whitespace, keep accents and punctuation)
    are the library-wide defaults. The configuration is stored on every
    :class:`~ehds_ontomap.MappingResult` so that a mapping can be reproduced.
    """

    lowercase: bool = True
    strip_accents: bool = False
    normalize_punct: bool = False

    def __call__(self, text: str) -> str:
        """Normalise ``text`` with this configuration."""
        return normalize_text(
            text,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
            normalize_punct=self.normalize_punct,
        )
