from __future__ import annotations

import re
from typing import Any

from primary_reason.core.types import DistanceMetric

_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?;,]+$", "", text)
    return text


def exact_distance(a: str, b: str) -> float:
    """0.0 if normalized strings are equal, else 1.0."""
    return 0.0 if normalize(a) == normalize(b) else 1.0


def lexical_distance(a: str, b: str) -> float:
    """1 - Jaccard similarity over word tokens. Range [0, 1]."""
    ta = set(_WORD.findall(a.lower()))
    tb = set(_WORD.findall(b.lower()))
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    jaccard = len(ta & tb) / len(union)
    return 1.0 - jaccard


def distance(a: str, b: str, *, metric: DistanceMetric = "lexical") -> float:
    if metric == "exact":
        return exact_distance(a, b)
    if metric == "lexical":
        return lexical_distance(a, b)
    if metric == "embedding":
        return _embedding_distance(a, b)
    raise ValueError(f"Unknown distance metric: {metric!r}")


def _embedding_distance(a: str, b: str) -> float:
    try:
        from sentence_transformers import (
            SentenceTransformer,
            util,
        )
    except ImportError as e:
        raise RuntimeError(
            "embedding distance requires sentence-transformers extra: pip install primary-reason[embeddings]"
        ) from e
    model: Any = _get_cached_model()
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _set_cached_model(model)
    emb = model.encode([a, b], convert_to_tensor=True)
    sim = float(util.cos_sim(emb[0], emb[1]).item())
    return max(0.0, min(1.0, 1.0 - (sim + 1.0) / 2.0))


_CACHED_MODEL: Any = None


def _get_cached_model() -> Any:
    return _CACHED_MODEL


def _set_cached_model(m: Any) -> None:
    global _CACHED_MODEL
    _CACHED_MODEL = m
