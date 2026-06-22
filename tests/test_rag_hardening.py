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
from backend.evaluation.evaluator import (
    EvalCase,
    Evaluator,
    relevance_score,
    groundedness_score,
    is_unknown,
)


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
