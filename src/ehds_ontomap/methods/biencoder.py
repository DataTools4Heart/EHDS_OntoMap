"""Bi-encoder similarity: cosine between term embeddings.

Default models mirror the ``nel`` section of
``app/model_manager/default_registry.yaml`` in
`dt4h-annotation-api <https://github.com/nlp4bia-bsc/dt4h-annotation-api>`_
(MIT License, Copyright (c) 2026 NLP4BIA).

Both default models are plain ``transformers`` checkpoints whose authors
prescribe the ``[CLS]`` token as the term representation. Loading them with
``SentenceTransformer(name)`` silently applies *mean* pooling instead, so this
module encodes with ``transformers`` directly. Models that are genuine
sentence-transformers models can be used with ``pooling=None``.

Requires the ``biencoder`` extra: ``pip install "ehds_ontomap[biencoder]"``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from ..ontology import Ontology
from .base import FloatMatrix, Matcher, ScoredTriple, TermIndex, rank_rows

__all__ = ["DEFAULT_MODELS", "BiEncoder", "Encoder", "TransformerEncoder", "load_encoder"]

logger = logging.getLogger(__name__)

SAPBERT_XLMR = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"
CLINLINKER_KB_P = "ICB-UMA/ClinLinker-KB-P"

DEFAULT_MODELS: dict[str, str] = {
    "cs": SAPBERT_XLMR,
    "en": SAPBERT_XLMR,
    "es": CLINLINKER_KB_P,
    "it": SAPBERT_XLMR,
    "nl": SAPBERT_XLMR,
    "ro": SAPBERT_XLMR,
    "sv": SAPBERT_XLMR,
}
"""Default model per language, as in the DT4H annotation API registry."""

Pooling: TypeAlias = Literal["cls", "mean"]

_MISSING_EXTRA = (
    "the biencoder method needs PyTorch and Transformers; "
    'install them with: pip install "ehds_ontomap[biencoder]"'
)


class Encoder(Protocol):
    """Anything that turns texts into L2-normalised float32 row vectors."""

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Return a ``(len(texts), dim)`` array of unit-norm embeddings."""
        ...


def _pick_device(device: str | None) -> str:
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformerEncoder:
    """Encode texts with a Hugging Face ``AutoModel`` and explicit pooling.

    Parameters
    ----------
    model
        Hugging Face model ID or local path.
    pooling
        ``"cls"`` (the ``[CLS]`` token, as SapBERT and ClinLinker prescribe)
        or ``"mean"`` (attention-masked mean over tokens).
    max_length
        Truncation length in tokens. 25 is the length SapBERT was trained
        with; clinical terms rarely exceed it.
    batch_size
        Texts per forward pass.
    device
        ``"cpu"``, ``"cuda"``, ``"mps"``…; picked automatically when omitted.
    revision
        Model revision (branch, tag or commit) to pin for reproducibility.
    """

    def __init__(
        self,
        model: str,
        *,
        pooling: Pooling = "cls",
        max_length: int = 25,
        batch_size: int = 64,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(_MISSING_EXTRA) from exc
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls', 'mean' or None, got {pooling!r}")

        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = _pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed ``texts``, batching by length to minimise padding."""
        import torch

        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        order = np.argsort([len(text) for text in texts], kind="stable")
        chunks: list[FloatMatrix] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [texts[i] for i in order[start : start + self.batch_size]]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                chunks.append(pooled.cpu().numpy())
        embeddings = np.empty((len(texts), chunks[0].shape[1]), dtype=np.float32)
        embeddings[order] = np.concatenate(chunks)
        return embeddings


class _SentenceTransformerEncoder:
    """Adapter for models published as sentence-transformers models."""

    def __init__(
        self, model: str, *, batch_size: int, device: str | None, revision: str | None
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(_MISSING_EXTRA) from exc
        self.batch_size = batch_size
        self.model = SentenceTransformer(model, device=device, revision=revision)

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def load_encoder(
    model: str,
    *,
    pooling: Pooling | None = "cls",
    max_length: int = 25,
    batch_size: int = 64,
    device: str | None = None,
    revision: str | None = None,
) -> Encoder:
    """Load an encoder for ``model``.

    ``pooling="cls"`` or ``"mean"`` loads the checkpoint with ``transformers``
    and applies that pooling. ``pooling=None`` loads it as a
    sentence-transformers model with its own pooling configuration (and its
    own maximum length; ``max_length`` is then ignored).
    """
    if pooling is None:
        return _SentenceTransformerEncoder(
            model, batch_size=batch_size, device=device, revision=revision
        )
    return TransformerEncoder(
        model,
        pooling=pooling,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        revision=revision,
    )


class BiEncoder(Matcher):
    """Score queries by cosine similarity of bi-encoder embeddings.

    Scores lie in ``[-1, 1]``; in practice related clinical terms score
    roughly 0.6-1.0, so thresholds must be chosen per model. Ontology terms
    and queries are embedded as written: no lexical normalisation is applied,
    because the models were trained on cased text.

    Parameters
    ----------
    model
        Hugging Face model ID or local path. Overrides ``language``.
    language
        ISO 639-1 code selecting a default model from :data:`DEFAULT_MODELS`.
    pooling
        ``"cls"`` (default; what both default models prescribe), ``"mean"``,
        or ``None`` for a sentence-transformers model's own configuration.
    max_length
        Truncation length in tokens.
    batch_size
        Texts per forward pass.
    device
        Torch device; picked automatically when omitted.
    revision
        Model revision to pin, for reproducibility.
    cache_dir
        Directory in which to store the ontology embeddings. They are reused
        when the model, its settings and the ontology terms are unchanged.
    encoder
        A ready-made :class:`Encoder`, bypassing model loading (useful for
        tests or custom models). ``model`` then only labels the results.
    """

    name = "biencoder"

    def __init__(
        self,
        model: str | None = None,
        *,
        language: str | None = None,
        pooling: Pooling | None = "cls",
        max_length: int = 25,
        batch_size: int = 64,
        device: str | None = None,
        revision: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        if model is None:
            if language is None:
                if encoder is None:
                    raise ValueError(
                        "the biencoder method needs language=... "
                        f"(one of {sorted(DEFAULT_MODELS)}) or model=..."
                    )
                model = "custom-encoder"
            elif language not in DEFAULT_MODELS:
                raise ValueError(
                    f"no default model for language {language!r}; "
                    f"choose one of {sorted(DEFAULT_MODELS)} or pass model=..."
                )
            else:
                model = DEFAULT_MODELS[language]
        self.model = model
        self.language = language
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.revision = revision
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._encoder = encoder
        self._index: TermIndex | None = None
        self._embeddings: FloatMatrix | None = None

    @property
    def encoder(self) -> Encoder:
        """The encoder, loaded on first use."""
        if self._encoder is None:
            logger.info("loading bi-encoder %s", self.model)
            self._encoder = load_encoder(
                self.model,
                pooling=self.pooling,
                max_length=self.max_length,
                batch_size=self.batch_size,
                device=self.device,
                revision=self.revision,
            )
        return self._encoder

    def _cache_key(self, keys: list[str]) -> str:
        digest = hashlib.sha256()
        settings = [self.model, self.revision, self.pooling, self.max_length]
        digest.update(json.dumps(settings).encode())
        for key in keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:32]

    def fit(self, ontology: Ontology) -> BiEncoder:
        """Embed every distinct ontology term (or load them from the cache)."""
        index = TermIndex(ontology, lambda term: term)
        cache_file: Path | None = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self._cache_key(index.keys)}.npy"
            if cache_file.is_file():
                embeddings = np.load(cache_file)
                if embeddings.shape[0] == len(index):
                    logger.info("loaded %d ontology embeddings from %s", len(index), cache_file)
                    self._index, self._embeddings = index, embeddings.astype(np.float32)
                    return self
                logger.warning("ignoring stale embedding cache %s", cache_file)

        logger.info("embedding %d distinct ontology terms with %s", len(index), self.model)
        embeddings = self.encoder.encode(index.keys)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
        self._index, self._embeddings = index, embeddings
        return self

    def score(
        self,
        queries: Sequence[str],
        *,
        threshold: float | None,
        top_k: int | None,
    ) -> list[list[ScoredTriple]]:
        """Embed the queries and compare them with every ontology term."""
        if self._index is None or self._embeddings is None:
            raise RuntimeError("call fit() before score()")
        embeddings = self._embeddings
        query_vectors = self.encoder.encode(list(queries))

        def score_block(start: int, stop: int) -> FloatMatrix:
            block: FloatMatrix = query_vectors[start:stop] @ embeddings.T
            return block

        return rank_rows(
            len(queries),
            len(self._index),
            self._index.entries,
            score_block,
            threshold=threshold,
            top_k=top_k,
        )

    def describe(self) -> dict[str, Any]:
        """Record the model and its settings."""
        return {
            "model": self.model,
            "extra": {
                "revision": self.revision,
                "pooling": self.pooling,
                "max_length": self.max_length,
            },
        }
