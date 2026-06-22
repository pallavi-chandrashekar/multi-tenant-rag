"""Pure, dependency-free RAG evaluation metrics.

These functions are deterministic and DB-/network-free so they can be unit
tested in isolation and reused by ``evaluator.py`` (offline harness) and the
``/eval/run`` API. Keeping them here separates *what we measure* from *how the
suite is driven*.

Metrics:
* ``relevance_score``      -- expected-keyword coverage in the answer.
* ``groundedness_score``   -- share of answer terms supported by cited sources.
* ``is_unknown``           -- whether an answer is the abstention response.
* ``summarize``            -- aggregate per-case results into headline metrics.
"""

from typing import Dict, List
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set:
    """Lower-cased alphanumeric token set."""
    return set(_TOKEN_RE.findall((text or "").lower()))


def relevance_score(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    if not expected_keywords:
        return 1.0
    a = (answer or "").lower()
    hits = sum(1 for k in expected_keywords if k.lower() in a)
    return hits / len(expected_keywords)


def groundedness_score(answer: str, sources: List[Dict]) -> float:
    """Proxy groundedness: share of answer content terms found in sources.

    A high score means the answer's vocabulary is supported by the retrieved
    context (a cheap stand-in for an LLM/NLI groundedness judge).
    """
    answer_terms = tokens(answer)
    if not answer_terms:
        return 0.0
    source_terms = set()
    for s in sources:
        source_terms |= tokens(s.get("text_snippet") or s.get("content", ""))
    if not source_terms:
        return 0.0
    supported = answer_terms & source_terms
    return len(supported) / len(answer_terms)


def is_unknown(answer: str, unknown_text: str = "I don't know") -> bool:
    """True when the answer is the configured abstention response."""
    return unknown_text.lower() in (answer or "").lower()


def summarize(per_case: List[Dict]) -> Dict:
    """Aggregate per-case rows into the headline evaluation summary."""
    answered = [c for c in per_case if c["answered"]]
    n = len(per_case) or 1
    n_answered = len(answered) or 1
    return {
        "cases": len(per_case),
        "citation_rate": round(sum(c["has_citations"] for c in answered) / n_answered, 3),
        "avg_source_count": round(sum(c["source_count"] for c in answered) / n_answered, 3),
        "unknown_answer_accuracy": round(sum(c["unknown_correct"] for c in per_case) / n, 3),
        "avg_relevance": round(sum(c["relevance"] for c in per_case) / n, 3),
        "avg_groundedness": round(sum(c["groundedness"] for c in per_case) / n, 3),
    }
