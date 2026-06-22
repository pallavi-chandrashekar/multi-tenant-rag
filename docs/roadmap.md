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

## Platform & scale

- [ ] **Kubernetes manifests + Helm chart** — production deployment with health
      probes, autoscaling, and secrets management.
- [ ] **Pluggable vector store interface** — adapter layer so pgvector can be
      swapped for a managed vector database (e.g. a HANA Vector adapter) without
      touching the retrieval service.
- [ ] **Background ingestion workers** — move embedding/ingestion off the request
      path onto a queue for large documents.

## Responsible AI & quality

- [ ] **OpenTelemetry** — export the existing `rag_query` signals (latency,
      retrieval counts, confidence, tokens, model, embedding model) as spans and
      metrics for dashboards and alerting.
- [ ] **RAGAS integration** — richer offline evaluation (faithfulness, answer
      relevancy, context precision/recall) alongside the built-in harness.
- [ ] **LLM-as-judge groundedness** — replace the lexical groundedness proxy with
      an NLI/LLM judge for higher-fidelity scoring.
- [ ] **PII detection & redaction** at ingestion time.

## Status legend

`[x]` shipped · `[ ]` planned. Shipped capabilities (hybrid retrieval, citations,
abstention, tenant isolation, evaluation harness, structured observability) are
documented in [`../README.md`](../README.md) and [`architecture.md`](architecture.md).
