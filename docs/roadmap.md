# Roadmap

This reference architecture is intentionally scoped to the *durable* concerns of
enterprise RAG — isolation, grounding, evaluation, observability — rather than a
single application. The roadmap below extends it toward a production deployment.

## Near term

- [ ] **JWT authentication** — bind users to tenants and remove the trust placed
      in the raw `X-Tenant-ID` header. Tenant identity becomes a verified claim.
- [ ] **RBAC** — document- and action-level authorization (who may ingest,
      delete, query, or run evaluation per tenant).
- [ ] **Streaming responses (SSE)** — token-by-token answer streaming for lower
      perceived latency.
- [ ] **Per-file chat scoping** — restrict a conversation to a chosen subset of a
      tenant's documents.

## Shipped

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
