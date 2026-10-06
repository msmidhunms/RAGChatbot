"""Evaluation metrics and what each one needs in order to be computed.

Selecting metrics (``evaluation.metrics`` or ``rag eval run --metrics``) decides
how much work an evaluation does: retrieval-only metrics never generate an
answer, deterministic answer metrics need an answer but no judge, and judge
metrics add LLM calls per question.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rag_chatbot.evaluation.metrics.answer import (
    answer_scores,
    exact_match,
    is_refusal,
    keyword_coverage,
    normalize_answer,
    token_f1,
)
from rag_chatbot.evaluation.metrics.judge import judge_scores
from rag_chatbot.evaluation.metrics.performance import latency_summary, percentile, usage_totals
from rag_chatbot.evaluation.metrics.retrieval import (
    contains_evidence,
    is_relevant,
    retrieval_scores,
    source_matches,
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str  # retrieval | answer | judge
    description: str
    higher_is_better: bool = True

    @property
    def needs_answer(self) -> bool:
        return self.group in ("answer", "judge")

    @property
    def needs_judge(self) -> bool:
        return self.group == "judge"


_SPECS = [
    MetricSpec("hit_rate", "retrieval", "a relevant chunk is in the top k"),
    MetricSpec("mrr", "retrieval", "1 / rank of the first relevant chunk"),
    MetricSpec("recall", "retrieval", "share of expected sources retrieved"),
    MetricSpec("precision", "retrieval", "share of retrieved chunks that are relevant"),
    MetricSpec("ndcg", "retrieval", "rank-weighted relevance (binary nDCG)"),
    MetricSpec("evidence_recall", "retrieval", "share of evidence snippets present in the context"),
    MetricSpec("filter_compliance", "retrieval", "share of retrieved chunks that satisfy the filter"),
    MetricSpec("exact_match", "answer", "normalised answer equals the reference"),
    MetricSpec("token_f1", "answer", "token overlap F1 with the reference"),
    MetricSpec("keyword_coverage", "answer", "share of required keywords in the answer"),
    MetricSpec("citation_validity", "answer", "share of [n] citations that point at a retrieved chunk"),
    MetricSpec("citation_precision", "answer", "share of cited chunks that are relevant"),
    MetricSpec("refusal_accuracy", "answer", "refuses unanswerable questions and answers the rest"),
    MetricSpec("false_refusal_rate", "answer", "answerable questions that were refused", False),
    MetricSpec("missed_refusal_rate", "answer", "unanswerable questions that were answered", False),
    MetricSpec("faithfulness", "judge", "answer claims are supported by the context"),
    MetricSpec("answer_relevance", "judge", "answer addresses the question"),
    MetricSpec("correctness", "judge", "answer agrees with the reference"),
    MetricSpec("context_recall", "judge", "reference facts are present in the context"),
    MetricSpec("context_precision", "judge", "share of retrieved chunks the judge finds relevant"),
]
METRICS: dict[str, MetricSpec] = {s.name: s for s in _SPECS}
# metrics computed together with another selected metric
DERIVED = {"false_refusal_rate": "refusal_accuracy", "missed_refusal_rate": "refusal_accuracy"}


def expand(selected: Iterable[str]) -> list[str]:
    """Selected metrics plus the ones derived from them, in registry order."""
    wanted = set(selected)
    wanted |= {d for d, parent in DERIVED.items() if parent in wanted}
    unknown = wanted - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; available: {', '.join(METRICS)}")
    return [m for m in METRICS if m in wanted]


__all__ = [
    "DERIVED",
    "METRICS",
    "MetricSpec",
    "answer_scores",
    "contains_evidence",
    "exact_match",
    "expand",
    "is_refusal",
    "is_relevant",
    "judge_scores",
    "keyword_coverage",
    "latency_summary",
    "normalize_answer",
    "percentile",
    "retrieval_scores",
    "source_matches",
    "token_f1",
    "usage_totals",
]
