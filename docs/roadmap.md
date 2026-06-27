# Roadmap

This reference architecture is intentionally scoped to the *durable* concerns of
enterprise RAG — isolation, grounding, evaluation, observability — rather than a
single application. The roadmap below extends it toward a production deployment.

## Near term

*All near-term items shipped.*

## Shipped

- [x] **Frontend auth flow** — `LoginGate` component with login/register forms; token +
      role persisted in `localStorage`; `apiFetch()` attaches `Authorization: Bearer`
      header; identity card in sidebar; graceful 401 → logout.
- [x] **JWT authentication** (optional) — `/auth/token` issues a tenant + role
      claim; when `AUTH_ENABLED`, tenant identity comes from the verified token
      instead of the trusted `X-Tenant-ID` header.
- [x] **RBAC** — `viewer` / `editor` / `admin` roles enforced per action
      (query, eval, ingest, delete) via a `require(action)` dependency.
- [x] **Streaming responses (SSE)** — `POST /api/v1/search/stream` streams answer
      tokens then a final metadata frame (sources, confidence, latency).
- [x] **Per-file chat scoping** — optional `document_ids` restricts retrieval to a
      chosen subset of a tenant's documents.
- [x] **Pluggable vector store interface** — `VectorStore` abstraction with a
      `pgvector` implementation and a HANA adapter stub; swap engines without
      touching retrieval/fusion/grounding.
- [x] **Cross-encoder reranking** (optional) — reorder fused candidates with a
      cross-encoder for higher retrieval precision.
- [x] **OpenTelemetry tracing** (optional) — spans for the retrieval and
      generation stages; no-op when disabled.
- [x] **RAGAS & DeepEval runners** — richer LLM-judged offline evaluation
      alongside the built-in harness.

## Platform & scale

- [ ] **Kubernetes manifests + Helm chart** — production deployment with health
      probes, autoscaling, and secrets management.
- [ ] **HANA Vector adapter implementation** — wire the stubbed `HanaVectorStore`
      to a live SAP HANA Vector Engine connection.
- [ ] **Background ingestion workers** — move embedding/ingestion off the request
      path onto a queue for large documents.

## Responsible AI & quality

- [ ] **LLM-as-judge groundedness** — replace the lexical groundedness proxy with
      an NLI/LLM judge for higher-fidelity scoring.
- [ ] **PII detection & redaction** at ingestion time.

## Status legend

`[x]` shipped · `[ ]` planned. Shipped capabilities (hybrid retrieval, citations,
abstention, tenant isolation, evaluation harness, structured observability) are
documented in [`../README.md`](../README.md) and [`architecture.md`](architecture.md).
