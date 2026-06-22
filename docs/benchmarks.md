# Benchmark Report

A recorded run of the built-in evaluation harness and per-query latency against
the local stack with the bundled sample documents. Reproduce with the commands
below; absolute latency depends on hardware and the OpenAI endpoint.

## Setup

| Parameter | Value |
| --- | --- |
| Date | 2026-06-22 |
| Documents | `data/sample_documents/` (company policy, incident runbook, product FAQ) |
| Tenant | `demo-corp` |
| Generation model | `gpt-4o-mini` |
| Embedding model | `all-MiniLM-L6-v2` (local, 384-dim) |
| Retrieval | hybrid (0.7·vector + 0.3·keyword), `RAG_TOP_K=5` |
| Abstention threshold | `RAG_MIN_CONFIDENCE_SCORE=0.40` |
| Reranker | disabled (default) |

## Quality (built-in harness, `POST /eval/run`)

| Metric | Score |
| --- | --- |
| citation_rate | 1.00 |
| avg_source_count | 3.6 |
| unknown_answer_accuracy | 1.00 |
| avg_relevance | 1.00 |
| avg_groundedness | 0.86 |

All five answerable questions returned grounded, cited answers; the one
out-of-scope question correctly abstained.

## Latency (end-to-end `/api/v1/search`, server-reported `latency_ms`)

Warm-state (the cold-start request that warms the local embedding model and LLM
connection is excluded):

| Statistic | ms |
| --- | --- |
| p50 | ~1,630 |
| avg | ~1,640 |
| min | ~1,210 |
| max | ~2,050 |

Steady-state answerable queries land in the **~1.2–2.0 s** range; latency is
dominated by the LLM generation call, not retrieval. The first request after a
cold start is an outlier (~8–12 s) while the embedding model loads. Enabling the
cross-encoder reranker adds the rerank pass (and a one-time model download)
without materially changing these headline quality metrics on this corpus.

## Methodology

```bash
# Quality
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" -d '{"tenant_id":"demo-corp"}'

# Latency: the search response includes a server-side latency_ms field
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the remote work policy?","tenant_id":"demo-corp","chat_history":[]}'
```

For LLM-judged metrics (faithfulness, answer relevancy, context precision/recall)
see [`evaluation.md`](evaluation.md) (RAGAS / DeepEval).
