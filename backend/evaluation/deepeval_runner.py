"""DeepEval integration -- optional.

Runs DeepEval metrics over the live RAG pipeline:

* faithfulness / hallucination -- is the answer grounded in the contexts?
* answer_relevancy             -- does the answer address the question?
* contextual_precision         -- are the retrieved contexts relevant / ranked?

DeepEval is an optional dependency (LLM-judged, makes OpenAI calls). Heavy
imports are deferred; the pure ``build_test_cases`` helper is unit-testable
without DeepEval installed.

Install:  ``pip install -r backend/requirements-eval.txt``
Run:      ``RAG_BASE_URL=... RAG_EVAL_TENANT=... python -m backend.evaluation.deepeval_runner``
"""

import os
from typing import Callable, Dict, List

from backend.evaluation.evaluator import DEFAULT_CASES, EvalCase


def _require_deepeval():
    try:
        import deepeval  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the lib
        raise RuntimeError(
            "DeepEval is not installed. Install it with:\n"
            "    pip install -r backend/requirements-eval.txt"
        ) from exc


def build_test_cases(
    cases: List[EvalCase], search_fn: Callable[[str], Dict]
) -> List[Dict]:
    """Build DeepEval-shaped records (input/actual_output/retrieval_context)
    from the live pipeline. Abstention cases are excluded.

    Pure aside from ``search_fn``; testable with a stub search function. Returns
    plain dicts so the shape can be asserted without importing DeepEval.
    """
    records = []
    for case in cases:
        if case.expect_unknown:
            continue
        resp = search_fn(case.query)
        contexts = [
            s.get("text_snippet") or s.get("content", "")
            for s in resp.get("sources", [])
        ]
        records.append(
            {
                "input": case.query,
                "actual_output": resp.get("answer", ""),
                "retrieval_context": contexts or [""],
                "expected_keywords": case.expected_keywords,
            }
        )
    return records


def run_deepeval(
    search_fn: Callable[[str], Dict],
    cases: List[EvalCase] = None,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate the pipeline with DeepEval and return per-case metric scores."""
    _require_deepeval()
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    cases = cases or DEFAULT_CASES
    records = build_test_cases(cases, search_fn)

    results = []
    for rec in records:
        tc = LLMTestCase(
            input=rec["input"],
            actual_output=rec["actual_output"],
            retrieval_context=rec["retrieval_context"],
            expected_output=" ".join(rec["expected_keywords"]) or None,
        )
        scores = {}
        for metric in (
            FaithfulnessMetric(threshold=threshold),
            AnswerRelevancyMetric(threshold=threshold),
            ContextualPrecisionMetric(threshold=threshold),
        ):
            metric.measure(tc)
            scores[type(metric).__name__] = metric.score
        results.append({"input": rec["input"], "scores": scores})
    return {"cases": results}


if __name__ == "__main__":  # pragma: no cover - live smoke run
    import json
    import requests

    base = os.getenv("RAG_BASE_URL", "http://localhost:8000")
    tenant = os.getenv("RAG_EVAL_TENANT", "demo-corp")

    def live_search(query: str) -> Dict:
        r = requests.post(
            f"{base}/api/v1/search",
            json={"text": query, "tenant_id": tenant, "chat_history": []},
            timeout=120,
        )
        return r.json()

    print(json.dumps(run_deepeval(live_search), indent=2))
