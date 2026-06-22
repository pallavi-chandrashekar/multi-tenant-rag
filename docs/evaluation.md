# Evaluation

The project ships three complementary evaluation paths. The **built-in harness**
is dependency-free and runs in CI; **RAGAS** and **DeepEval** are optional,
industry-standard, LLM-judged frameworks for deeper quality signals.

---

## 1. Built-in harness (default, no extra deps)

Pure metric functions in [`../backend/evaluation/metrics.py`](../backend/evaluation/metrics.py),
driven by [`../backend/evaluation/evaluator.py`](../backend/evaluation/evaluator.py)
over the labelled suite in
[`../backend/evaluation/sample_questions.json`](../backend/evaluation/sample_questions.json).

| Metric | Meaning |
| --- | --- |
| `citation_rate` | answered queries carrying ≥ 1 source |
| `avg_source_count` | citations per answer |
| `unknown_answer_accuracy` | correct abstention on out-of-scope queries |
| `avg_relevance` | expected-keyword coverage |
| `avg_groundedness` | answer terms supported by cited sources |

```bash
# CLI
RAG_BASE_URL=http://localhost:8000 RAG_EVAL_TENANT=demo-corp \
  python -m backend.evaluation.evaluator

# API
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" -d '{"tenant_id":"demo-corp"}'
```

---

## 2. RAGAS (optional)

[`../backend/evaluation/ragas_runner.py`](../backend/evaluation/ragas_runner.py)
scores **faithfulness**, **answer_relevancy**, **context_precision**, and
**context_recall**. RAGAS uses an LLM judge and makes OpenAI calls.

```bash
pip install -r backend/requirements-eval.txt
RAG_BASE_URL=http://localhost:8000 RAG_EVAL_TENANT=demo-corp \
  python -m backend.evaluation.ragas_runner
```

---

## 3. DeepEval (optional)

[`../backend/evaluation/deepeval_runner.py`](../backend/evaluation/deepeval_runner.py)
scores **faithfulness**, **answer_relevancy**, and **contextual_precision** per
case (also LLM-judged).

```bash
pip install -r backend/requirements-eval.txt
RAG_BASE_URL=http://localhost:8000 RAG_EVAL_TENANT=demo-corp \
  python -m backend.evaluation.deepeval_runner
```

---

## Which to use

- **CI / fast regression gate** → built-in harness (deterministic, no API cost).
- **Deep quality analysis / reports** → RAGAS or DeepEval (richer, LLM-judged).

All three consume the same labelled cases and the same live `/api/v1/search`
path, so results are directly comparable. See
[`benchmarks.md`](benchmarks.md) for a recorded run of the built-in harness.
