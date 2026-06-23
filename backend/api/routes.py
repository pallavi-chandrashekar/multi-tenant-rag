from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import List, Dict, Optional
import datetime
import json
import logging
import time
import uuid

from backend.config import settings
from backend.database import get_db
from backend.models import ChatSession # <--- Import New Model
from backend.services.chat import rewrite_query, generate_multi_queries, decompose_query
from backend.services.ingestion import process_document
from backend.services.router import route_query
from backend.services.llm import chat_with_llm, generate_grounded_answer, stream_grounded_answer
from backend.services.retrieval_service import retrieval_service
from backend.services.auth import Principal
from backend.api.deps import require
from backend.evaluation.evaluator import Evaluator, DEFAULT_CASES
from backend.observability.tracing import span, set_attributes

logger = logging.getLogger("rag.api")

router = APIRouter()


def _tenant_of(principal: Principal, fallback: str) -> str:
    """Tenant id: from the verified token when auth is on, else the request value."""
    return principal.tenant_id if settings.AUTH_ENABLED else fallback


def _embedding_model_name() -> str:
    """Resolve the active embedding model for observability."""
    if settings.EMBEDDING_PROVIDER == "openai":
        return settings.OPENAI_EMBEDDING_MODEL
    return settings.LOCAL_EMBEDDING_MODEL

class SearchPayload(BaseModel):
    text: str
    tenant_id: str
    session_id: Optional[str] = None # <--- New Field
    chat_history: List[Dict[str, str]] = []
    # Optional per-file scoping: restrict retrieval to these document ids.
    document_ids: Optional[List[str]] = None

class RenameChatPayload(BaseModel):
    title: str

# --- ROUTING + RETRIEVAL (shared by JSON + streaming endpoints) ---
def _route(query_text: str) -> str:
    q_lower = query_text.lower()
    summary_triggers = ["summarize", "summary", "tldr", "overview"]
    if q_lower in summary_triggers or any(q_lower.startswith(s) for s in summary_triggers):
        return "summary"
    if any(x in q_lower for x in ["hi", "hello", "hey", "how are you"]):
        return "llm_only"
    return route_query(query_text)


def _resolve_rag(payload: SearchPayload, tenant_id: str, db: Session) -> Dict:
    """Route + retrieve. Returns everything needed to produce an answer, leaving
    grounded generation to the caller so JSON and SSE can share this logic.

    Keys: ``mode``, ``sources``, ``confidence``, ``standalone_query``,
    ``generate`` (whether grounded generation is needed) and ``static_answer``
    (the final answer when no streaming generation applies).
    """
    query_text = payload.text.strip()
    mode = _route(query_text)

    if mode == "llm_only":
        msgs = [{"role": ("assistant" if m["role"] == "ai" else "user"), "content": m["content"]} for m in payload.chat_history]
        msgs.append({"role": "user", "content": query_text})
        return {"mode": mode, "sources": [], "confidence": 0.0,
                "standalone_query": query_text, "generate": False,
                "static_answer": chat_with_llm(msgs)}

    if mode == "summary":
        docs = db.execute(
            text("SELECT content FROM chunks WHERE tenant_id = :tid LIMIT 15"),
            {"tid": tenant_id},
        ).fetchall()
        if docs:
            combined = "\n\n".join([r.content for r in docs])
            answer = chat_with_llm([{"role": "user", "content": f"Summarize:\n{combined}"}])
        else:
            answer = settings.UNKNOWN_ANSWER_TEXT if settings.RAG_ENABLE_UNKNOWN_ANSWER else "No documents found to summarize."
        return {"mode": mode, "sources": [], "confidence": 0.0,
                "standalone_query": query_text, "generate": False,
                "static_answer": answer}

    mode = "hybrid" if settings.RAG_ENABLE_HYBRID_RETRIEVAL else "vector"
    standalone_query = query_text
    if payload.chat_history:
        standalone_query = rewrite_query(query_text, payload.chat_history)

    with span("rag.retrieval", tenant_id=tenant_id) as sp:
        retrieved = retrieval_service.retrieve(
            db, standalone_query, tenant_id, document_ids=payload.document_ids
        )
        confidence = retrieval_service.confidence(retrieved)
        set_attributes(
            sp, retrieval_count=len(retrieved), confidence=confidence,
            reranker=settings.RAG_ENABLE_RERANKER,
        )

    grounded = (
        retrieved
        and not (
            settings.RAG_ENABLE_UNKNOWN_ANSWER
            and confidence < settings.RAG_MIN_CONFIDENCE_SCORE
        )
    )
    if grounded:
        filenames = retrieval_service.lookup_filenames(
            db, [r["document_id"] for r in retrieved]
        )
        sources = retrieval_service.format_sources(retrieved, tenant_id, filenames)
        return {"mode": mode, "sources": sources, "confidence": confidence,
                "standalone_query": standalone_query, "generate": True,
                "static_answer": None}

    answer = (
        settings.UNKNOWN_ANSWER_TEXT
        if settings.RAG_ENABLE_UNKNOWN_ANSWER
        else "No relevant information was found in your documents."
    )
    return {"mode": "unknown", "sources": [], "confidence": confidence,
            "standalone_query": standalone_query, "generate": False,
            "static_answer": answer}


def _log_query(tenant_id, mode, sources, confidence, latency_ms, tokens):
    logger.info(
        "rag_query tenant=%s mode=%s retrieval_count=%d selected_sources=%d "
        "confidence=%.3f model=%s embedding_model=%s latency_ms=%d tokens=%d",
        tenant_id, mode, len(sources), len(sources), confidence,
        settings.OPENAI_CHAT_MODEL, _embedding_model_name(), latency_ms, tokens,
    )


def _save_history(db, payload, tenant_id, answer, mode, confidence, sources):
    if not payload.session_id:
        return
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        session = ChatSession(
            id=payload.session_id, tenant_id=tenant_id,
            title=payload.text.strip()[:30] + "...", history=[],
        )
        db.add(session)
    new_history = list(session.history) if session.history else []
    new_history.append({"role": "user", "content": payload.text.strip()})
    new_history.append({
        "role": "ai",
        "content": answer or "Here are the search results.",
        "strategy": mode,
        "confidence": confidence,
        "sources": [{"filename": s["filename"], "chunk_index": s["chunk_index"]} for s in sources],
    })
    session.history = new_history
    session.updated_at = datetime.datetime.utcnow()
    db.commit()


# --- 1. SEARCH (Grounded RAG + history persistence) ---
@router.post("/api/v1/search")
async def search_rag(
    payload: SearchPayload,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require("query")),
):
    """Grounded, tenant-scoped RAG.

    Modes: ``LLM_ONLY``, ``SUMMARY``, ``SEARCH`` (hybrid retrieval). Returns a
    citation-rich response with confidence + observability metadata.
    """
    started = time.perf_counter()
    tenant_id = _tenant_of(principal, payload.tenant_id)

    r = _resolve_rag(payload, tenant_id, db)
    mode, sources, confidence = r["mode"], r["sources"], r["confidence"]
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if r["generate"]:
        with span("rag.generate", source_count=len(sources)) as sp:
            answer, token_usage = generate_grounded_answer(
                r["standalone_query"], sources, payload.chat_history
            )
            set_attributes(sp, tokens=token_usage.get("total_tokens", 0))
        # Drop citations if the model abstained anyway.
        if answer.strip() == settings.UNKNOWN_ANSWER_TEXT:
            sources = []
            mode = "unknown"
    else:
        answer = r["static_answer"]

    latency_ms = int((time.perf_counter() - started) * 1000)
    _log_query(tenant_id, mode, sources, confidence, latency_ms, token_usage.get("total_tokens", 0))

    legacy_results = [
        {"id": s["chunk_id"], "content": s["text_snippet"], "score": s["combined_score"]}
        for s in sources
    ]
    _save_history(db, payload, tenant_id, answer, mode, confidence, sources)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "mode": mode,
        "tenant_id": tenant_id,
        "latency_ms": latency_ms,
        "model": settings.OPENAI_CHAT_MODEL,
        "embedding_model": _embedding_model_name(),
        "token_usage": token_usage,
        # --- Backwards-compatible fields ---
        "results": legacy_results,
        "strategy_used": mode,
    }


# --- 1b. STREAMING SEARCH (Server-Sent Events) ---
@router.post("/api/v1/search/stream")
async def search_rag_stream(
    payload: SearchPayload,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require("query")),
):
    """Same grounded RAG as ``/search`` but streamed as SSE: ``token`` events
    carry answer deltas, then a final ``metadata`` event carries sources,
    confidence, mode and latency."""
    started = time.perf_counter()
    tenant_id = _tenant_of(principal, payload.tenant_id)
    r = _resolve_rag(payload, tenant_id, db)

    def event_stream():
        mode, sources, confidence = r["mode"], r["sources"], r["confidence"]
        collected = []
        if r["generate"]:
            for delta in stream_grounded_answer(
                r["standalone_query"], sources, payload.chat_history
            ):
                collected.append(delta)
                yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"
            answer = "".join(collected).strip()
            if answer == settings.UNKNOWN_ANSWER_TEXT:
                sources, mode = [], "unknown"
        else:
            answer = r["static_answer"]
            yield f"event: token\ndata: {json.dumps({'text': answer})}\n\n"

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log_query(tenant_id, mode, sources, confidence, latency_ms, 0)
        _save_history(db, payload, tenant_id, answer, mode, confidence, sources)
        meta = {
            "mode": mode, "sources": sources, "confidence": confidence,
            "tenant_id": tenant_id, "latency_ms": latency_ms,
            "model": settings.OPENAI_CHAT_MODEL,
            "embedding_model": _embedding_model_name(),
        }
        yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- EVALUATION: run the offline harness against the live stack ---
class EvalRunPayload(BaseModel):
    tenant_id: str


@router.post("/eval/run")
async def run_eval(
    payload: EvalRunPayload,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require("eval")),
):
    """Run the bundled evaluation suite against the live RAG pipeline for a
    tenant. Each labelled question is sent through the real search path, then
    scored for citations, abstention correctness, relevance and groundedness."""
    tenant_id = _tenant_of(principal, payload.tenant_id)
    # Drive the real search path per case (async), caching responses by query.
    responses: Dict[str, Dict] = {}
    for case in DEFAULT_CASES:
        sp = SearchPayload(text=case.query, tenant_id=tenant_id, chat_history=[])
        responses[case.query] = await search_rag(sp, db, principal)

    report = Evaluator().run(DEFAULT_CASES, lambda q: responses[q])
    report["tenant_id"] = tenant_id
    return report

# ---  LIST CHATS ---
@router.get("/api/v1/chats")
async def list_chats(db: Session = Depends(get_db), principal: Principal = Depends(require("query"))):
    chats = db.query(ChatSession).filter(ChatSession.tenant_id == principal.tenant_id).order_by(ChatSession.updated_at.desc()).all()
    return [{"id": str(c.id), "title": c.title, "updated_at": str(c.updated_at)} for c in chats]

# ---  GET CHAT ---
@router.get("/api/v1/chats/{session_id}")
async def get_chat(session_id: str, db: Session = Depends(get_db), principal: Principal = Depends(require("query"))):
    chat = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.tenant_id == principal.tenant_id).first()
    if not chat: raise HTTPException(404, "Chat not found")
    return {"history": chat.history}

# ---  DELETE CHAT ---
@router.delete("/api/v1/chats/{session_id}")
async def delete_chat(session_id: str, db: Session = Depends(get_db), principal: Principal = Depends(require("delete"))):
    db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.tenant_id == principal.tenant_id).delete()
    db.commit()
    return {"status": "success"}

# --- INGEST & DELETE DOCS ---
@router.post("/api/v1/ingest")
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db), principal: Principal = Depends(require("ingest"))):
    doc_id = await process_document(file, principal.tenant_id, db)
    return {"status": "success", "doc_id": doc_id}

@router.get("/api/v1/documents")
async def list_documents(db: Session = Depends(get_db), principal: Principal = Depends(require("query"))):
    results = db.execute(text("SELECT id, filename, created_at FROM documents WHERE tenant_id = :tid ORDER BY created_at DESC"), {"tid": principal.tenant_id}).fetchall()
    return {"documents": [{"id": str(r.id), "filename": r.filename, "created_at": str(r.created_at).split(" ")[0]} for r in results]}

@router.delete("/api/v1/documents/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db), principal: Principal = Depends(require("delete"))):
    result = db.execute(text("DELETE FROM documents WHERE id = :id AND tenant_id = :tid"), {"id": doc_id, "tid": principal.tenant_id})
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success"}

# --- RENAME CHAT ---
@router.put("/api/v1/chats/{session_id}")
async def rename_chat(
    session_id: str,
    payload: RenameChatPayload,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require("query")),
):
    chat = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.tenant_id == principal.tenant_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat.title = payload.title
    db.commit()
    return {"status": "success", "title": chat.title}