"""Centralised application configuration.

All tunable behaviour for the RAG pipeline is exposed here as environment
variables so the reference architecture can be reconfigured without code
changes. Defaults are production-sensible and let the stack boot with zero
configuration for local evaluation.
"""

import os


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Runtime settings resolved from the environment."""

    # --- Models -----------------------------------------------------------
    # Chat/generation model used for grounded answers, routing and summaries.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Embedding configuration. The reference stack ships with a *local*
    # sentence-transformers model (no API key required, 384 dimensions) so it
    # is fully runnable offline. Set EMBEDDING_PROVIDER=openai to switch to the
    # hosted model below (requires re-ingestion + a matching Vector() column).
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    LOCAL_EMBEDDING_MODEL: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 384)

    # --- Retrieval --------------------------------------------------------
    RAG_TOP_K: int = _get_int("RAG_TOP_K", 5)
    RAG_MIN_CONFIDENCE_SCORE: float = _get_float("RAG_MIN_CONFIDENCE_SCORE", 0.55)
    RAG_ENABLE_HYBRID_RETRIEVAL: bool = _get_bool("RAG_ENABLE_HYBRID_RETRIEVAL", True)
    RAG_ENABLE_UNKNOWN_ANSWER: bool = _get_bool("RAG_ENABLE_UNKNOWN_ANSWER", True)
    RAG_ENABLE_CITATIONS: bool = _get_bool("RAG_ENABLE_CITATIONS", True)

    # Hybrid scoring weights: combined = VECTOR_WEIGHT*vec + KEYWORD_WEIGHT*kw
    RAG_VECTOR_WEIGHT: float = _get_float("RAG_VECTOR_WEIGHT", 0.7)
    RAG_KEYWORD_WEIGHT: float = _get_float("RAG_KEYWORD_WEIGHT", 0.3)

    # Minimum cosine similarity for a vector candidate to be considered.
    RAG_VECTOR_FLOOR: float = _get_float("RAG_VECTOR_FLOOR", 0.35)

    # --- Cross-encoder reranking ------------------------------------------
    # When enabled, retrieve RAG_RERANK_CANDIDATES fused candidates and reorder
    # them with a cross-encoder before truncating to RAG_TOP_K. Off by default
    # so the stack stays light; enabling downloads the reranker model on first
    # use (sentence-transformers CrossEncoder).
    RAG_ENABLE_RERANKER: bool = _get_bool("RAG_ENABLE_RERANKER", False)
    RAG_RERANKER_MODEL: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RAG_RERANK_CANDIDATES: int = _get_int("RAG_RERANK_CANDIDATES", 20)

    # --- Authentication (JWT) + RBAC --------------------------------------
    # When enabled, tenant identity is taken from a verified JWT claim instead
    # of the trusted X-Tenant-ID header, and role-based authorization is
    # enforced. Default off so the demo / tests keep using the header.
    AUTH_ENABLED: bool = _get_bool("AUTH_ENABLED", False)
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = _get_int("JWT_EXPIRE_MINUTES", 60)

    # --- Vector store backend ---------------------------------------------
    # Pluggable vector store. `pgvector` is the default; `hana` selects the
    # (stubbed) SAP HANA Vector Engine adapter.
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "pgvector")

    # --- OpenTelemetry tracing --------------------------------------------
    # Off by default; enabling requires the opentelemetry-* packages. When
    # disabled, tracing helpers are no-ops with zero runtime overhead.
    OTEL_ENABLED: bool = _get_bool("OTEL_ENABLED", False)
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "enterprise-rag")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    # Standard response message when the system cannot ground an answer.
    UNKNOWN_ANSWER_TEXT: str = os.getenv(
        "UNKNOWN_ANSWER_TEXT",
        "I don't know based on the available documents.",
    )


settings = Settings()
