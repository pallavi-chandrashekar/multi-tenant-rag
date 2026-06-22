"""Cross-encoder reranking.

A bi-encoder (embedding) retrieval stage is fast but approximate. A cross-encoder
scores the full ``(query, chunk)`` pair jointly and is far more precise, so the
pipeline becomes:

    query -> vector + keyword fusion (top N candidates)
          -> cross-encoder rerank
          -> top K chunks -> LLM

The heavy ``sentence-transformers`` CrossEncoder is loaded lazily on first use,
so importing this module stays cheap and the default stack (reranker disabled)
never pays for it. The ordering logic is factored into the pure ``apply_rerank``
so it is unit-testable without the model.
"""

from typing import Dict, List

from backend.config import settings

_model = None


def _get_model():
    """Load and cache the cross-encoder on first use."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        print(f"Loading Cross-Encoder Reranker ({settings.RAG_RERANKER_MODEL})...")
        _model = CrossEncoder(settings.RAG_RERANKER_MODEL)
    return _model


def apply_rerank(
    candidates: List[Dict], scores: List[float], top_k: int
) -> List[Dict]:
    """Attach ``rerank_score`` to each candidate, sort descending, truncate.

    Pure and deterministic so it can be tested with injected scores (no model).
    """
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(
        candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True
    )
    return ranked[:top_k]


def rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """Reorder ``candidates`` by cross-encoder relevance to ``query``."""
    if not candidates:
        return []
    pairs = [(query, c.get("content", "")) for c in candidates]
    scores = _get_model().predict(pairs)
    return apply_rerank(candidates, list(scores), top_k)
