"""Lightweight RAG evaluation harness.

Computes offline quality metrics over a set of labelled cases without requiring
any third-party eval service. The metric functions live in ``metrics.py`` (pure
and unit-testable); ``Evaluator.run`` drives them against any ``search_fn`` (the
live API, a stub, or a mock) so the same harness works in CI, against a deployed
stack, and behind the ``POST /eval/run`` endpoint.

Metrics (see ``metrics.py``):
* citation_rate          -- answers that carry >= 1 source.
* avg_source_count       -- mean citations per answered query.
* unknown_answer_accuracy-- correct abstention on out-of-scope queries.
* relevance              -- expected keyword coverage in the answer.
* groundedness           -- proxy: answer terms supported by cited sources.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

# Re-exported so existing imports (`from ...evaluator import relevance_score`)
# and tests keep working after the metrics were extracted into metrics.py.
from backend.evaluation.metrics import (  # noqa: F401
    relevance_score,
    groundedness_score,
    is_unknown,
    summarize,
)

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "sample_questions.json")


@dataclass
class EvalCase:
    query: str
    # Keywords expected in a correct grounded answer.
    expected_keywords: List[str] = field(default_factory=list)
    # True when the system *should* abstain ("I don't know...").
    expect_unknown: bool = False


def load_cases(path: str = _QUESTIONS_PATH) -> List[EvalCase]:
    """Load labelled evaluation cases from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        EvalCase(
            query=item["query"],
            expected_keywords=item.get("expected_keywords", []),
            expect_unknown=item.get("expect_unknown", False),
        )
        for item in raw
    ]


class Evaluator:
    def __init__(self, unknown_text: str = "I don't know based on the available documents."):
        self.unknown_text = unknown_text

    def run(self, cases: List[EvalCase], search_fn: Callable[[str], Dict]) -> Dict:
        """Run all cases. ``search_fn(query)`` must return a response dict with
        ``answer`` and ``sources`` keys (matching the API contract)."""
        per_case = []
        for case in cases:
            resp = search_fn(case.query)
            answer = resp.get("answer", "")
            sources = resp.get("sources", []) or []
            abstained = is_unknown(answer, self.unknown_text)

            if case.expect_unknown:
                unknown_correct = abstained
                relevance = 1.0 if abstained else 0.0
                grounded = 1.0 if abstained else 0.0
            else:
                unknown_correct = not abstained
                relevance = relevance_score(answer, case.expected_keywords)
                grounded = groundedness_score(answer, sources)

            per_case.append(
                {
                    "query": case.query,
                    "answered": not abstained,
                    "source_count": len(sources),
                    "has_citations": len(sources) > 0,
                    "unknown_correct": unknown_correct,
                    "relevance": round(relevance, 3),
                    "groundedness": round(grounded, 3),
                }
            )

        return {"summary": summarize(per_case), "cases": per_case}


# Default suite, loaded from sample_questions.json (aligned with the bundled
# sample documents). Falls back to an empty list if the file is unreadable.
try:
    DEFAULT_CASES = load_cases()
except (OSError, ValueError):  # pragma: no cover - defensive
    DEFAULT_CASES = []


if __name__ == "__main__":  # pragma: no cover - live smoke run
    import requests

    base = os.getenv("RAG_BASE_URL", "http://localhost:8000")
    tenant = os.getenv("RAG_EVAL_TENANT", "demo-corp")

    def live_search(query: str) -> Dict:
        r = requests.post(
            f"{base}/api/v1/search",
            json={"text": query, "tenant_id": tenant, "chat_history": []},
            timeout=60,
        )
        return r.json()

    report = Evaluator().run(DEFAULT_CASES, live_search)
    print(json.dumps(report, indent=2))
