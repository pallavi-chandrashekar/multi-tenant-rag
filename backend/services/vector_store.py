"""Pluggable vector store interface.

The retrieval pipeline depends only on the small ``VectorStore`` interface, so
the underlying nearest-neighbour engine can be swapped without touching the
retrieval, fusion, or grounding logic. ``pgvector`` is the default; adapters for
other engines (SAP HANA Vector Engine, Pinecone, Weaviate, Chroma, ...) implement
the same ``search`` contract.

Tenant isolation is part of the contract: every implementation MUST filter by
``tenant_id`` so there is no cross-tenant read path.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.config import settings


class VectorStore(ABC):
    """Tenant-scoped approximate nearest-neighbour search over chunk embeddings."""

    name: str = "abstract"

    @abstractmethod
    def search(
        self,
        db: Session,
        query_vector: List[float],
        tenant_id: str,
        limit: int,
        floor: float,
        document_ids: List[str] = None,
    ) -> List[Dict]:
        """Return up to ``limit`` candidates for ``tenant_id`` whose cosine
        similarity exceeds ``floor``.

        When ``document_ids`` is provided, the search is further restricted to
        that subset of the tenant's documents (per-file chat scoping).

        Each candidate is a dict with: ``id``, ``document_id``, ``content``,
        ``metadata``, ``vector_score`` and ``keyword_score`` (0.0 here; filled in
        by the hybrid fusion stage).
        """
        raise NotImplementedError


class PgVectorStore(VectorStore):
    """PostgreSQL + ``pgvector`` cosine search (the default backend)."""

    name = "pgvector"

    def search(
        self,
        db: Session,
        query_vector: List[float],
        tenant_id: str,
        limit: int,
        floor: float,
        document_ids: List[str] = None,
    ) -> List[Dict]:
        vector_str = str(query_vector)
        params = {
            "vec": vector_str,
            "tenant_id": tenant_id,
            "floor": floor,
            "limit": limit,
        }
        doc_clause = ""
        if document_ids:
            doc_clause = "AND document_id::text = ANY(:doc_ids)"
            params["doc_ids"] = list(document_ids)
        rows = db.execute(
            text(
                f"""
                SELECT id, document_id, content, metadata,
                       1 - (embedding <=> :vec) AS vector_score
                FROM chunks
                WHERE tenant_id = :tenant_id
                  AND (1 - (embedding <=> :vec)) > :floor
                  {doc_clause}
                ORDER BY embedding <=> :vec
                LIMIT :limit
                """
            ),
            params,
        ).fetchall()

        return [
            {
                "id": str(r.id),
                "document_id": str(r.document_id),
                "content": r.content,
                "metadata": r.metadata or {},
                "vector_score": float(r.vector_score),
                "keyword_score": 0.0,
            }
            for r in rows
        ]


class HanaVectorStore(VectorStore):
    """SAP HANA Vector Engine adapter (stub).

    Demonstrates the extension point for an enterprise vector backend. A real
    implementation would issue a ``COSINE_SIMILARITY`` query against a HANA
    table, still filtered by ``tenant_id``. It intentionally raises until wired
    to a HANA connection so misconfiguration fails loudly rather than silently
    falling back to another tenant's store.
    """

    name = "hana"

    def search(
        self,
        db: Session,
        query_vector: List[float],
        tenant_id: str,
        limit: int,
        floor: float,
        document_ids: List[str] = None,
    ) -> List[Dict]:
        raise NotImplementedError(
            "HanaVectorStore is a stub. Implement a tenant-scoped "
            "COSINE_SIMILARITY query against the SAP HANA Vector Engine and set "
            "VECTOR_STORE=hana."
        )


# Registry of available backends, keyed by VECTOR_STORE value.
_STORES = {
    PgVectorStore.name: PgVectorStore,
    HanaVectorStore.name: HanaVectorStore,
}


def get_vector_store(name: str = None) -> VectorStore:
    """Resolve the configured vector store (defaults to ``settings.VECTOR_STORE``)."""
    key = (name or settings.VECTOR_STORE or "pgvector").lower()
    store_cls = _STORES.get(key, PgVectorStore)
    return store_cls()
