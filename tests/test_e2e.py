"""End-to-end integration tests — require a live Docker Compose stack.

    docker-compose up -d
    pytest tests/test_e2e.py -v -m integration

Optional env vars:
    RAG_BASE_URL   backend base URL (default: http://localhost:8000)

Scenarios covered:
  - Server health + auth/config endpoint
  - Auth flow: register → login → use token → bad token → duplicate register
  - RBAC: viewer blocked from ingest; editor blocked from delete; admin can all
  - Grounded answer with citations
  - Abstention on out-of-scope query
  - Tenant isolation: cross-tenant query abstains
  - Per-file scoping: document_ids filters retrieval
  - SSE streaming: event shape and metadata fields
  - Document list and 404 on delete-nonexistent
"""
import json
import os
import uuid
from typing import Optional

import pytest
import requests

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
UNKNOWN = "I don't know based on the available documents."


# ─── low-level helpers ────────────────────────────────────────────────────────

def _headers(tenant_id: str, token: Optional[str] = None) -> dict:
    h = {"X-Tenant-ID": tenant_id}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _ingest(tenant_id: str, content: str, filename: str, token: Optional[str] = None) -> requests.Response:
    return requests.post(
        f"{BASE_URL}/api/v1/ingest",
        files={"file": (filename, content.encode(), "text/plain")},
        headers=_headers(tenant_id, token),
    )


def _search(
    tenant_id: str,
    text: str,
    document_ids: Optional[list] = None,
    token: Optional[str] = None,
) -> requests.Response:
    payload: dict = {"text": text, "tenant_id": tenant_id, "chat_history": []}
    if document_ids:
        payload["document_ids"] = document_ids
    return requests.post(
        f"{BASE_URL}/api/v1/search",
        json=payload,
        headers={**_headers(tenant_id, token), "Content-Type": "application/json"},
    )


def _register(username: str, password: str, tenant_id: str, role: str = "viewer") -> requests.Response:
    return requests.post(
        f"{BASE_URL}/auth/register",
        json={"username": username, "password": password, "tenant_id": tenant_id, "role": role},
    )


def _login(username: str, password: str) -> requests.Response:
    return requests.post(
        f"{BASE_URL}/auth/token",
        data={"username": username, "password": password},
    )


# ─── session fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tenant_id() -> str:
    return f"e2e-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def tenant_b_id() -> str:
    return f"e2e-b-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def auth_enabled() -> bool:
    r = requests.get(f"{BASE_URL}/auth/config")
    assert r.status_code == 200
    return r.json()["auth_enabled"]


@pytest.fixture(scope="session")
def admin_token(auth_enabled: bool, tenant_id: str) -> Optional[str]:
    """JWT for an admin user (None when AUTH_ENABLED=false)."""
    if not auth_enabled:
        return None
    u = f"admin-{uuid.uuid4().hex[:6]}"
    _register(u, "pass123", tenant_id, role="admin")
    r = _login(u, "pass123")
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def ingested_docs(tenant_id: str, admin_token: Optional[str]) -> dict:
    """Ingest two topic-distinct documents once per session; returns {name: doc_id}."""
    r1 = _ingest(
        tenant_id,
        "Remote work policy: employees may work from home up to three days per week. "
        "All remote work days must be pre-approved by the direct manager.",
        "remote_policy.txt",
        token=admin_token,
    )
    assert r1.status_code == 200, r1.text

    r2 = _ingest(
        tenant_id,
        "Expense policy: business expenses under $75 do not require a receipt. "
        "Expenses of $75 or more must include an itemized receipt submitted within 30 days.",
        "expense_policy.txt",
        token=admin_token,
    )
    assert r2.status_code == 200, r2.text

    return {
        "remote": r1.json()["doc_id"],
        "expense": r2.json()["doc_id"],
    }


# ─── server health ────────────────────────────────────────────────────────────

def test_server_is_up():
    r = requests.get(f"{BASE_URL}/docs")
    assert r.status_code == 200


# ─── auth/config endpoint ─────────────────────────────────────────────────────

def test_auth_config_is_public():
    r = requests.get(f"{BASE_URL}/auth/config")
    assert r.status_code == 200
    data = r.json()
    assert "auth_enabled" in data
    assert isinstance(data["auth_enabled"], bool)


# ─── auth flow ────────────────────────────────────────────────────────────────

def test_register_and_login_returns_token(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"user-{uuid.uuid4().hex[:6]}"
    r = _register(u, "pw", tenant_id, role="viewer")
    assert r.status_code == 200
    assert r.json()["username"] == u
    assert r.json()["role"] == "viewer"

    r = _login(u, "pw")
    assert r.status_code == 200
    assert "access_token" in r.json()
    assert r.json()["role"] == "viewer"


def test_wrong_password_returns_401(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"user-{uuid.uuid4().hex[:6]}"
    _register(u, "correct", tenant_id)
    r = _login(u, "wrong-password")
    assert r.status_code == 401


def test_bad_token_returns_401(auth_enabled):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    r = requests.post(
        f"{BASE_URL}/api/v1/search",
        json={"text": "hello", "tenant_id": "x", "chat_history": []},
        headers={"Authorization": "Bearer garbage.token.here", "Content-Type": "application/json"},
    )
    assert r.status_code == 401


def test_duplicate_register_returns_409(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"dup-{uuid.uuid4().hex[:6]}"
    _register(u, "pw", tenant_id)
    r = _register(u, "pw", tenant_id)
    assert r.status_code == 409


def test_register_with_invalid_role_returns_400(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    r = _register(f"u-{uuid.uuid4().hex[:6]}", "pw", tenant_id, role="superuser")
    assert r.status_code == 400


# ─── RBAC ─────────────────────────────────────────────────────────────────────

def test_viewer_cannot_ingest(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"viewer-{uuid.uuid4().hex[:6]}"
    _register(u, "pw", tenant_id, role="viewer")
    token = _login(u, "pw").json()["access_token"]
    r = _ingest(tenant_id, "some content", "test.txt", token=token)
    assert r.status_code == 403


def test_editor_cannot_delete(auth_enabled, tenant_id, ingested_docs):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"editor-{uuid.uuid4().hex[:6]}"
    _register(u, "pw", tenant_id, role="editor")
    token = _login(u, "pw").json()["access_token"]
    r = requests.delete(
        f"{BASE_URL}/api/v1/documents/{ingested_docs['remote']}",
        headers=_headers(tenant_id, token),
    )
    assert r.status_code == 403


def test_editor_can_ingest(auth_enabled, tenant_id):
    if not auth_enabled:
        pytest.skip("AUTH_ENABLED=false")
    u = f"editor-{uuid.uuid4().hex[:6]}"
    _register(u, "pw", tenant_id, role="editor")
    token = _login(u, "pw").json()["access_token"]
    r = _ingest(tenant_id, "editor-uploaded content", "editor_doc.txt", token=token)
    assert r.status_code == 200


# ─── grounded answer + citations ──────────────────────────────────────────────

def test_grounded_answer_with_citations(tenant_id, ingested_docs, admin_token):
    r = _search(
        tenant_id,
        "How many days per week can employees work from home?",
        token=admin_token,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] != UNKNOWN
    assert len(data["sources"]) >= 1
    assert any("remote_policy" in s["filename"] for s in data["sources"])
    assert data["confidence"] >= 0.4


def test_abstention_on_out_of_scope_query(tenant_id, ingested_docs, admin_token):
    r = _search(tenant_id, "What were the 2019 cloud revenue figures?", token=admin_token)
    assert r.status_code == 200
    assert r.json()["answer"] == UNKNOWN


def test_response_includes_observability_fields(tenant_id, ingested_docs, admin_token):
    r = _search(tenant_id, "What is the expense reimbursement limit?", token=admin_token)
    assert r.status_code == 200
    data = r.json()
    assert "confidence" in data
    assert "latency_ms" in data
    assert "model" in data
    assert "embedding_model" in data
    assert "token_usage" in data


# ─── tenant isolation ─────────────────────────────────────────────────────────

def test_tenant_b_cannot_see_tenant_a_documents(auth_enabled, tenant_b_id, ingested_docs, admin_token):
    if auth_enabled:
        u = f"b-{uuid.uuid4().hex[:6]}"
        _register(u, "pw", tenant_b_id, role="admin")
        token_b = _login(u, "pw").json()["access_token"]
    else:
        token_b = None
    r = _search(
        tenant_b_id,
        "How many days per week can employees work from home?",
        token=token_b,
    )
    assert r.status_code == 200
    assert r.json()["answer"] == UNKNOWN, (
        f"Tenant isolation broken: tenant_b saw tenant_a's document. "
        f"Answer: {r.json()['answer']}"
    )


# ─── per-file scoping ─────────────────────────────────────────────────────────

def test_per_file_scoping_finds_correct_doc(tenant_id, ingested_docs, admin_token):
    r = _search(
        tenant_id,
        "What is the remote work policy?",
        document_ids=[ingested_docs["remote"]],
        token=admin_token,
    )
    assert r.status_code == 200
    assert r.json()["answer"] != UNKNOWN


def test_per_file_scoping_abstains_when_scoped_to_wrong_doc(tenant_id, ingested_docs, admin_token):
    # Remote-work question scoped to the expense doc only → should abstain.
    r = _search(
        tenant_id,
        "How many days per week can employees work from home?",
        document_ids=[ingested_docs["expense"]],
        token=admin_token,
    )
    assert r.status_code == 200
    assert r.json()["answer"] == UNKNOWN


def test_per_file_scoping_finds_expense_doc(tenant_id, ingested_docs, admin_token):
    r = _search(
        tenant_id,
        "What is the expense reimbursement receipt threshold?",
        document_ids=[ingested_docs["expense"]],
        token=admin_token,
    )
    assert r.status_code == 200
    assert r.json()["answer"] != UNKNOWN


# ─── SSE streaming ────────────────────────────────────────────────────────────

def _stream_events(tenant_id: str, text: str, token: Optional[str]) -> list:
    """Collect all SSE event names from the streaming endpoint."""
    payload = {"text": text, "tenant_id": tenant_id, "chat_history": []}
    h = {**_headers(tenant_id, token), "Content-Type": "application/json"}
    r = requests.post(
        f"{BASE_URL}/api/v1/search/stream",
        json=payload,
        headers=h,
        stream=True,
        timeout=60,
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")

    events: list = []
    current_event: Optional[str] = None
    data_lines: dict = {}

    for raw in r.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith("event: "):
            current_event = line[len("event: "):]
            events.append(current_event)
            data_lines[current_event] = None
        elif line.startswith("data: ") and current_event:
            data_lines[current_event] = line[len("data: "):]
        if "done" in events:
            break

    return events, data_lines


def test_sse_stream_emits_token_metadata_done(tenant_id, ingested_docs, admin_token):
    events, _ = _stream_events(
        tenant_id,
        "What is the remote work policy?",
        admin_token,
    )
    assert "token" in events, f"missing token events; got: {events}"
    assert "metadata" in events, f"missing metadata event; got: {events}"
    assert "done" in events, f"missing done event; got: {events}"


def test_sse_metadata_frame_has_required_fields(tenant_id, ingested_docs, admin_token):
    _, data_lines = _stream_events(
        tenant_id,
        "What is the remote work policy?",
        admin_token,
    )
    assert "metadata" in data_lines, "no metadata frame received"
    meta = json.loads(data_lines["metadata"])
    for field in ("mode", "confidence", "latency_ms", "sources", "model"):
        assert field in meta, f"metadata missing field: {field}"


# ─── document list + delete ───────────────────────────────────────────────────

def test_list_documents_returns_ingested_docs(tenant_id, ingested_docs, admin_token):
    r = requests.get(
        f"{BASE_URL}/api/v1/documents",
        headers=_headers(tenant_id, admin_token),
    )
    assert r.status_code == 200
    ids = [d["id"] for d in r.json()["documents"]]
    assert ingested_docs["remote"] in ids
    assert ingested_docs["expense"] in ids


def test_delete_nonexistent_document_returns_404(tenant_id, admin_token):
    fake_id = str(uuid.uuid4())
    r = requests.delete(
        f"{BASE_URL}/api/v1/documents/{fake_id}",
        headers=_headers(tenant_id, admin_token),
    )
    assert r.status_code == 404
