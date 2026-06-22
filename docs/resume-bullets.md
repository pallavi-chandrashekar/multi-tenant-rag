# Resume Bullets

Reusable, quantifiable bullets describing this project. Adjust scope/seniority
framing to the target role.

## Senior ML Engineer / AI Platform Engineer

- Designed and built a **multi-tenant RAG reference architecture** (FastAPI,
  React, PostgreSQL/pgvector) with **strict per-tenant data isolation enforced at
  the database layer** and asserted by an automated test suite.
- Implemented **hybrid retrieval** (weighted vector + keyword fusion) with a
  **confidence-gated abstention** path, eliminating ungrounded answers on
  out-of-scope or low-confidence queries.
- Engineered **grounded generation with citations** and a structured **response
  contract** (answer, sources, confidence, latency, model, embedding model, token
  usage) for full per-answer auditability.
- Built an **offline evaluation harness** (citation rate, abstention accuracy,
  relevance, groundedness) exposed via a `POST /eval/run` API and wired into
  **CI**, enabling regression-tested RAG quality.
- Added **structured observability** (latency, retrieval counts, confidence,
  model/embedding-model, token usage) designed to map directly onto
  OpenTelemetry spans and metrics.

## Shorter variants

- Built a production-style **multi-tenant RAG platform** with grounded answers,
  citations, hybrid retrieval, abstention, and an offline evaluation harness.
- Shipped **tenant-isolated retrieval** and a **grounded, citation-backed answer
  contract** with built-in evaluation and observability.

## Impact framing

- Turned an unverifiable LLM chatbot into an **auditable, tenant-isolated
  assistant** that abstains instead of hallucinating — the controls enterprises
  need to safely point LLMs at internal knowledge.
