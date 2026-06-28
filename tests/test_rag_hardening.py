"""Unit tests for the hardened RAG behaviour.

These are DB- and network-free: they exercise the pure retrieval scoring,
citation formatting, abstention path, tenant-isolation scoping, and the
evaluation harness. Live end-to-end checks live in `test-scenarios.py`
(marked `integration`).
"""

import pytest

from backend.config import settings
from backend.services.chunking import smart_split_text
from backend.services.retrieval_service import (
    RetrievalService,
    compute_keyword_score,
)
from backend.services import retrieval_service as rs_module
from backend.services.llm import generate_grounded_answer
from backend.services.reranker import apply_rerank
from backend.services.vector_store import (
    PgVectorStore,
    HanaVectorStore,
    get_vector_store,
)
from backend.observability.tracing import span, set_attributes
from backend.services.auth import role_allows, Principal, ROLE_PERMISSIONS
from backend.evaluation.evaluator import (
    EvalCase,
    Evaluator,
    relevance_score,
    groundedness_score,
    is_unknown,
)
from backend.evaluation.ragas_runner import build_samples
from backend.evaluation.deepeval_runner import build_test_cases


# --------------------------------------------------------------------------
# Hybrid retrieval scoring
# --------------------------------------------------------------------------
def test_compute_keyword_score_full_and_partial_overlap():
    assert compute_keyword_score("remote work policy", "the remote work policy") == 1.0
    # meaningful query terms: many, vacation, days -> 2 of 3 present
    score = compute_keyword_score("how many vacation days", "employees accrue vacation days")
    assert score == pytest.approx(2 / 3)
    assert compute_keyword_score("password reset", "billing and invoices") == 0.0


def test_compute_keyword_score_empty_query():
    assert compute_keyword_score("", "anything") == 0.0
    assert compute_keyword_score("the a an", "content") == 0.0  # all stopwords


def test_hybrid_fusion_uses_weighted_combination():
    svc = RetrievalService()
    candidates = [
        # moderate vector, no keyword overlap
        {"id": "1", "document_id": "d1", "content": "semantic only chunk",
         "metadata": {}, "vector_score": 0.6, "keyword_score": 0.0},
        # lower vector, perfect keyword overlap
        {"id": "2", "document_id": "d1", "content": "refund window policy",
         "metadata": {}, "vector_score": 0.4, "keyword_score": 0.0},
    ]
    ranked = svc._fuse("refund window policy", candidates)
    by_id = {c["id"]: c for c in ranked}
    # combined = 0.7*vec + 0.3*keyword (keyword recomputed inside _fuse)
    assert by_id["1"]["combined_score"] == pytest.approx(0.7 * 0.6 + 0.3 * 0.0)  # 0.42
    assert by_id["2"]["combined_score"] == pytest.approx(0.7 * 0.4 + 0.3 * 1.0)  # 0.58
    # strong keyword match overcomes the vector gap here
    assert ranked[0]["id"] == "2"


def test_confidence_is_best_combined_score():
    svc = RetrievalService()
    results = [{"combined_score": 0.3}, {"combined_score": 0.81}, {"combined_score": 0.5}]
    assert svc.confidence(results) == 0.81
    assert svc.confidence([]) == 0.0


# --------------------------------------------------------------------------
# Chunking -- sentence-aware, never splits a word (regression guard)
# --------------------------------------------------------------------------
def test_chunker_never_splits_words():
    # Regression: the previous fixed-window splitter cut words in half
    # (e.g. "receipt" -> "rec" + "eipt"), which silently broke keyword search
    # and fact grounding. Every emitted token must be a whole source token.
    text = (
        "Business expenses under seventy five dollars do not require a receipt. "
        "Expenses of seventy five dollars or more must include an itemized "
        "receipt and be submitted within thirty days of purchase. "
    ) * 6
    source_tokens = set(text.split())
    chunks = smart_split_text(text, chunk_size=120, overlap=30)

    assert len(chunks) > 1, "expected multiple chunks to exercise boundaries"
    for chunk in chunks:
        for token in chunk.split():
            assert token in source_tokens, f"chunk contains partial token {token!r}"
    # The split-prone word survives intact somewhere.
    assert any("receipt" in c for c in chunks)


def test_chunker_keeps_fact_sentence_intact():
    text = (
        "The remote work policy allows three days per week. "
        "Business expenses under $75 do not require a receipt. "
        "Sick leave is provided separately at 10 days per year."
    )
    chunks = smart_split_text(text, chunk_size=200, overlap=20)
    assert any(
        "Business expenses under $75 do not require a receipt." in c for c in chunks
    )


def test_chunker_multiple_bounded_chunks_no_data_loss():
    sentences = [f"This is fact number {i} about topic {i}." for i in range(30)]
    text = " ".join(sentences)
    chunks = smart_split_text(text, chunk_size=120, overlap=30)

    assert len(chunks) > 1
    # Bounded (a little slack for the carried overlap tail).
    for c in chunks:
        assert len(c) <= 120 + 60
    # Every original sentence is preserved in at least one chunk.
    for s in sentences:
        assert any(s in c for c in chunks)


def test_chunker_oversized_sentence_kept_whole():
    # A single sentence longer than chunk_size cannot be split on a boundary,
    # so it is emitted whole rather than cut mid-word.
    long_sentence = ("supercalifragilistic " * 20).strip() + "."
    chunks = smart_split_text(long_sentence, chunk_size=100, overlap=20)
    assert chunks == [long_sentence]


def test_chunker_edge_cases():
    assert smart_split_text("") == []
    assert smart_split_text("   \n  ") == []
    assert smart_split_text("Just one sentence.") == ["Just one sentence."]


# --------------------------------------------------------------------------
# Citations / format_sources
# --------------------------------------------------------------------------
def test_format_sources_builds_citation_fields():
    svc = RetrievalService()
    results = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Remote work is allowed three days per week.",
            "metadata": {"chunk_index": 4},
            "vector_score": 0.88,
            "keyword_score": 0.5,
            "combined_score": 0.766,
        }
    ]
    sources = svc.format_sources(results, "tenant-a", {"doc-1": "company_policy.md"})
    assert len(sources) == 1
    s = sources[0]
    assert s["document_id"] == "doc-1"
    assert s["filename"] == "company_policy.md"
    assert s["chunk_id"] == "chunk-1"
    assert s["chunk_index"] == 4
    assert s["tenant_id"] == "tenant-a"
    assert "Remote work" in s["text_snippet"]
    assert s["vector_score"] == 0.88
    assert s["keyword_score"] == 0.5
    assert s["combined_score"] == 0.766


def test_format_sources_snippet_truncated():
    svc = RetrievalService()
    long_text = "x" * 500
    results = [{"id": "c", "document_id": "d", "content": long_text, "metadata": {},
               "vector_score": 0.5, "keyword_score": 0.0, "combined_score": 0.35}]
    sources = svc.format_sources(results, "t", {"d": "f.md"})
    assert len(sources[0]["text_snippet"]) <= 301  # 300 chars + ellipsis


# --------------------------------------------------------------------------
# Unknown-answer / abstention
# --------------------------------------------------------------------------
def test_grounded_answer_abstains_without_sources():
    answer, usage = generate_grounded_answer("anything", sources=[])
    assert answer == settings.UNKNOWN_ANSWER_TEXT
    assert usage["total_tokens"] == 0


def test_low_confidence_triggers_abstention_decision():
    # Mirrors the gate in routes.search_rag.
    svc = RetrievalService()
    results = [{"combined_score": 0.2}]
    confidence = svc.confidence(results)
    should_abstain = (
        settings.RAG_ENABLE_UNKNOWN_ANSWER
        and confidence < settings.RAG_MIN_CONFIDENCE_SCORE
    )
    assert confidence < settings.RAG_MIN_CONFIDENCE_SCORE
    assert should_abstain is True


# --------------------------------------------------------------------------
# Tenant isolation -- every retrieval query is tenant-scoped
# --------------------------------------------------------------------------
class _FakeResult:
    def fetchall(self):
        return []


class _SpySession:
    """Captures SQL text + params passed to execute()."""

    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        return _FakeResult()


def test_retrieve_vector_is_tenant_scoped(monkeypatch):
    monkeypatch.setattr(rs_module, "get_embeddings", lambda texts: [[0.0] * 384])
    svc = RetrievalService()
    db = _SpySession()
    svc.retrieve_vector(db, "any question", "tenant-xyz")

    assert db.calls, "expected a SQL query to be issued"
    sql, params = db.calls[0]
    assert "tenant_id = :tenant_id" in sql
    assert params["tenant_id"] == "tenant-xyz"


def test_keyword_candidates_are_tenant_scoped():
    svc = RetrievalService()
    db = _SpySession()
    svc._retrieve_keyword_candidates(db, "refund policy", "tenant-xyz", limit=10)
    sql, params = db.calls[0]
    assert "tenant_id = :tenant_id" in sql
    assert params["tenant_id"] == "tenant-xyz"


# --------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------
def test_relevance_and_groundedness_scores():
    assert relevance_score("remote work allowed", ["remote", "work"]) == 1.0
    assert relevance_score("nothing here", ["remote"]) == 0.0
    grounded = groundedness_score(
        "remote work allowed",
        [{"text_snippet": "remote work is allowed three days a week"}],
    )
    assert grounded == pytest.approx(1.0)
    assert is_unknown("I don't know based on the available documents.")


def test_evaluator_run_summary():
    cases = [
        EvalCase("known q", expected_keywords=["alpha"]),
        EvalCase("out of scope", expect_unknown=True),
    ]

    def fake_search(query):
        if "out of scope" in query:
            return {"answer": "I don't know based on the available documents.", "sources": []}
        return {
            "answer": "alpha is the answer",
            "sources": [{"text_snippet": "alpha is the answer to everything"}],
        }

    report = Evaluator().run(cases, fake_search)
    summ = report["summary"]
    assert summ["cases"] == 2
    assert summ["unknown_answer_accuracy"] == 1.0  # both handled correctly
    assert summ["citation_rate"] == 1.0  # the one answered case had a citation
    assert summ["avg_relevance"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Cross-encoder reranking
# --------------------------------------------------------------------------
def test_apply_rerank_reorders_and_truncates():
    candidates = [
        {"id": "a", "content": "x", "combined_score": 0.9},
        {"id": "b", "content": "y", "combined_score": 0.5},
        {"id": "c", "content": "z", "combined_score": 0.7},
    ]
    # Cross-encoder disagrees with fusion: 'b' is most relevant.
    ranked = apply_rerank(candidates, scores=[0.1, 0.95, 0.4], top_k=2)
    assert [c["id"] for c in ranked] == ["b", "c"]
    assert ranked[0]["rerank_score"] == pytest.approx(0.95)
    assert len(ranked) == 2


# --------------------------------------------------------------------------
# Pluggable vector store
# --------------------------------------------------------------------------
def test_get_vector_store_selects_backend():
    assert isinstance(get_vector_store("pgvector"), PgVectorStore)
    assert isinstance(get_vector_store("hana"), HanaVectorStore)
    # Unknown backends fall back to the safe default rather than crashing.
    assert isinstance(get_vector_store("nope"), PgVectorStore)


def test_pgvector_store_search_is_tenant_scoped():
    store = PgVectorStore()
    db = _SpySession()
    store.search(db, [0.0] * 384, "tenant-xyz", limit=10, floor=0.35)
    sql, params = db.calls[0]
    assert "tenant_id = :tenant_id" in sql
    assert params["tenant_id"] == "tenant-xyz"


def test_hana_store_raises_until_implemented():
    with pytest.raises(NotImplementedError):
        HanaVectorStore().search(None, [0.0], "t", limit=5, floor=0.3)


# --------------------------------------------------------------------------
# OpenTelemetry tracing -- no-op when disabled
# --------------------------------------------------------------------------
def test_tracing_is_noop_when_disabled():
    # OTEL_ENABLED is False by default, so span() yields None and never raises.
    with span("rag.retrieval", tenant_id="t") as s:
        assert s is None
        set_attributes(s, retrieval_count=3)  # must be safe on a None span


# --------------------------------------------------------------------------
# Optional eval runners -- pure sample builders
# --------------------------------------------------------------------------
def _stub_search(query):
    if "scope" in query:
        return {"answer": "I don't know based on the available documents.", "sources": []}
    return {"answer": "alpha", "sources": [{"text_snippet": "alpha context"}]}


def test_ragas_build_samples_shape_and_excludes_unknown():
    cases = [
        EvalCase("known q", expected_keywords=["alpha"]),
        EvalCase("out of scope", expect_unknown=True),
    ]
    samples = build_samples(cases, _stub_search)
    assert len(samples) == 1  # abstention case excluded
    s = samples[0]
    assert set(s) == {"question", "answer", "contexts", "ground_truth"}
    assert s["contexts"] == ["alpha context"]
    assert s["ground_truth"] == "alpha"


def test_deepeval_build_test_cases_shape_and_excludes_unknown():
    cases = [
        EvalCase("known q", expected_keywords=["alpha"]),
        EvalCase("out of scope", expect_unknown=True),
    ]
    records = build_test_cases(cases, _stub_search)
    assert len(records) == 1
    r = records[0]
    assert r["input"] == "known q"
    assert r["actual_output"] == "alpha"
    assert r["retrieval_context"] == ["alpha context"]


# --------------------------------------------------------------------------
# RBAC role matrix
# --------------------------------------------------------------------------
def test_rbac_role_permissions():
    # viewer: read + eval only
    assert role_allows("viewer", "query")
    assert role_allows("viewer", "eval")
    assert not role_allows("viewer", "ingest")
    assert not role_allows("viewer", "delete")
    # editor: + ingest, still no delete
    assert role_allows("editor", "ingest")
    assert not role_allows("editor", "delete")
    # admin: everything
    assert all(role_allows("admin", a) for a in ["query", "eval", "ingest", "delete", "manage"])
    # unknown role / action -> denied
    assert not role_allows("nobody", "query")
    assert not role_allows("admin", "launch-missiles")


def test_principal_defaults():
    p = Principal(tenant_id="t1")
    assert p.tenant_id == "t1"
    assert p.role == "admin"  # fallback principal has full role


# --------------------------------------------------------------------------
# Per-file scoping -- document_ids restricts the tenant-scoped SQL
# --------------------------------------------------------------------------
def test_pgvector_store_search_respects_document_ids():
    store = PgVectorStore()
    db = _SpySession()
    store.search(db, [0.0] * 384, "tenant-xyz", limit=10, floor=0.35,
                 document_ids=["doc-1", "doc-2"])
    sql, params = db.calls[0]
    assert "tenant_id = :tenant_id" in sql
    assert "document_id::text = ANY(:doc_ids)" in sql
    assert params["doc_ids"] == ["doc-1", "doc-2"]


def test_keyword_candidates_respect_document_ids():
    svc = RetrievalService()
    db = _SpySession()
    svc._retrieve_keyword_candidates(db, "refund policy", "tenant-xyz", limit=10,
                                     document_ids=["doc-9"])
    sql, params = db.calls[0]
    assert "document_id::text = ANY(:doc_ids)" in sql
    assert params["doc_ids"] == ["doc-9"]


def test_no_document_ids_means_no_doc_filter():
    store = PgVectorStore()
    db = _SpySession()
    store.search(db, [0.0] * 384, "t", limit=5, floor=0.3)
    sql, params = db.calls[0]
    assert "document_id::text = ANY" not in sql
    assert "doc_ids" not in params


# --------------------------------------------------------------------------
# JWT create / decode (requires python-jose; skipped in lite CI)
# --------------------------------------------------------------------------
def test_jwt_encode_decode_round_trip():
    pytest.importorskip("jose")
    from backend.services.auth import create_access_token, decode_token

    token = create_access_token(tenant_id="acme", role="editor", username="alice")
    assert isinstance(token, str) and len(token) > 20
    principal = decode_token(token)
    assert principal.tenant_id == "acme"
    assert principal.role == "editor"
    assert principal.username == "alice"


def test_jwt_decode_raises_401_on_bad_token():
    pytest.importorskip("jose")
    from fastapi import HTTPException

    from backend.services.auth import decode_token

    with pytest.raises(HTTPException) as exc:
        decode_token("not.a.valid.token")
    assert exc.value.status_code == 401


# --------------------------------------------------------------------------
# Password hashing (requires passlib; skipped in lite CI)
# --------------------------------------------------------------------------
def test_password_hash_verify_round_trip():
    pytest.importorskip("passlib")
    from backend.services.auth import hash_password, verify_password

    hashed = hash_password("s3cr3t!")
    assert hashed != "s3cr3t!"
    assert verify_password("s3cr3t!", hashed)
    assert not verify_password("wrong-password", hashed)


# --------------------------------------------------------------------------
# get_principal / require() dependency behaviour (called directly, no DI)
# --------------------------------------------------------------------------
def test_get_principal_returns_admin_fallback_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    from backend.api.deps import get_principal

    p = get_principal(token=None, x_tenant_id="acme")
    assert p.tenant_id == "acme"
    assert p.role == "admin"


def test_get_principal_raises_401_when_no_token_and_auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    from fastapi import HTTPException

    from backend.api.deps import get_principal

    with pytest.raises(HTTPException) as exc:
        get_principal(token=None, x_tenant_id="acme")
    assert exc.value.status_code == 401


def test_require_raises_403_when_role_lacks_action(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    from fastapi import HTTPException

    from backend.api.deps import require

    dep = require("ingest")
    viewer = Principal(tenant_id="t", role="viewer")
    with pytest.raises(HTTPException) as exc:
        dep(principal=viewer)
    assert exc.value.status_code == 403


def test_require_passes_through_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    from backend.api.deps import require

    dep = require("delete")
    viewer = Principal(tenant_id="t", role="viewer")
    result = dep(principal=viewer)
    assert result.tenant_id == "t"
