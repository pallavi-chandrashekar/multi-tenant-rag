# National Interest / Impact Statement

This document frames the project's relevance to the broader national interest in
trustworthy, safely-adopted enterprise AI. It is written to support an EB-2 NIW
portfolio while remaining an accurate description of the engineering work.

## Problem

Enterprises are adopting large language models faster than they are adopting the
controls that make them safe to use on internal knowledge:

- **Hallucination** — ungrounded models answer confidently from parametric memory,
  producing plausible but unsupported claims.
- **No provenance** — answers cannot be traced to a source document, so they
  cannot be audited or trusted for decisions.
- **Weak data isolation** — multi-tenant systems risk leaking one organization's
  (or department's) data into another's answers.
- **No measurement** — teams ship RAG systems with no offline metric for
  citation coverage, abstention correctness, or groundedness.

## Solution

A reusable reference architecture that codifies the controls rather than just
documenting them:

- **Grounded generation** — answers are produced only from retrieved, tenant-scoped
  context; the model is instructed to use nothing else.
- **Citations** — every answer carries the document, chunk, and retrieval scores
  it relied on, making answers auditable.
- **Abstention** — on no or low-confidence retrieval the system replies
  *"I don't know based on the available documents."* instead of guessing.
- **Strict multi-tenancy** — every retrieval and read query is filtered by
  `tenant_id` at the SQL layer; there is no cross-tenant read path.
- **Hybrid retrieval** — weighted vector + keyword fusion so exact-term matches
  are not lost to embedding drift.
- **Built-in evaluation & observability** — an offline harness scores citation
  rate, abstention accuracy, relevance and groundedness; each query emits
  structured signals (latency, confidence, model, embedding model, tokens).

## Impact

- **Trustworthy AI** — grounding + citations + abstention turn an unverifiable
  chatbot into an auditable assistant.
- **Safer AI adoption** — tenant isolation and minimal logging make it defensible
  to point LLMs at sensitive internal knowledge.
- **Enterprise productivity** — employees reach authoritative answers from their
  own documents instead of searching manually.
- **Knowledge accessibility** — the same patterns generalize across industries
  (policy, runbooks, product knowledge), lowering the barrier to responsible
  adoption broadly rather than for a single firm.

## Why a reference architecture (not a product)

By codifying the *durable* engineering concerns — isolation, grounding,
evaluation, observability, abstention — the project is reusable across
organizations adopting LLM systems, amplifying its impact beyond any single
deployment.
