"""RAGAS integration -- optional.

Runs the industry-standard RAGAS metrics over the live RAG pipeline:

* faithfulness        -- is the answer supported by the retrieved contexts?
* answer_relevancy    -- does the answer address the question?
* context_precision   -- are the retrieved contexts relevant / well-ranked?
* context_recall      -- do the contexts cover the reference answer?

RAGAS is an optional dependency (it pulls in an LLM judge and makes OpenAI
calls). The heavy imports are deferred so this module imports cheaply; the pure
``build_samples`` helper is unit-testable without RAGAS installed.

Install:  ``pip install -r backend/requirements-eval.txt``
Run:      ``RAG_BASE_URL=... RAG_EVAL_TENANT=... python -m backend.evaluation.ragas_runner``
"""

import os
from typing import Callable, Dict, List

from backend.evaluation.evaluator import DEFAULT_CASES, EvalCase


def _require_ragas():
    try:
        import ragas  # noqa: F401
        from datasets import Dataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the lib
        raise RuntimeError(
            "RAGAS is not installed. Install it with:\n"
            "    pip install -r backend/requirements-eval.txt"
        ) from exc


def build_samples(
    cases: List[EvalCase], search_fn: Callable[[str], Dict]
) -> List[Dict]:
    """Build RAGAS samples (question/answer/contexts/ground_truth) from the live
    pipeline. Abstention cases are excluded -- RAGAS scores answered queries.

    Pure aside from ``search_fn``; testable with a stub search function.
    """
    samples = []
    for case in cases:
        if case.expect_unknown:
            continue
        resp = search_fn(case.query)
        contexts = [
            s.get("text_snippet") or s.get("content", "")
            for s in resp.get("sources", [])
        ]
        samples.append(
            {
                "question": case.query,
                "answer": resp.get("answer", ""),
                "contexts": contexts or [""],
                # Reference proxy: the expected keywords (or the answer itself
                # when no labels are available) so context_recall can score.
                "ground_truth": " ".join(case.expected_keywords)
                or resp.get("answer", ""),
            }
        )
    return samples


def run_ragas(
    search_fn: Callable[[str], Dict],
    cases: List[EvalCase] = None,
    metrics: List = None,
) -> Dict:
    """Evaluate the pipeline with RAGAS and return per-metric scores."""
    _require_ragas()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    cases = cases or DEFAULT_CASES
    samples = build_samples(cases, search_fn)
    dataset = Dataset.from_list(samples)
    metrics = metrics or [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    result = evaluate(dataset, metrics=metrics)
    # `result` is dict-like; coerce to a plain dict of metric -> score.
    return {k: float(v) for k, v in dict(result).items()}


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

    print(json.dumps(run_ragas(live_search), indent=2))
