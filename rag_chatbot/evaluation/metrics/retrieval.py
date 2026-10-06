"""Retrieval metrics computed without an LLM.

A retrieved chunk counts as relevant when its source matches one of the
item's ``expected_sources`` (path suffix) or it contains one of the item's
``evidence`` snippets (case- and whitespace-insensitive).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.evaluation.dataset import QAItem
from rag_chatbot.stores.filters import matches

_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def source_matches(source: str, expected: str) -> bool:
    """Expected sources match as a path suffix: 'sample.md' matches '/docs/sample.md'."""
    source, expected = source.replace("\\", "/"), expected.replace("\\", "/").lstrip("./")
    return source == expected or source.endswith("/" + expected)


def contains_evidence(text: str, evidence: Sequence[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(e) in norm for e in evidence)


def is_relevant(chunk: RetrievedChunk, item: QAItem) -> bool:
    source = str(chunk.metadata.get("source", ""))
    if any(source_matches(source, e) for e in item.expected_sources):
        return True
    return bool(item.evidence) and contains_evidence(chunk.text, item.evidence)


def retrieval_scores(chunks: Sequence[RetrievedChunk], item: QAItem, k: int) -> dict[str, float]:
    """hit_rate, mrr, precision, recall, ndcg, evidence_recall@k and filter_compliance."""
    top = list(chunks)[:k]
    scores: dict[str, float] = {}

    if item.filter:
        ok = [matches(c.metadata, item.filter) for c in top]
        scores["filter_compliance"] = sum(ok) / len(ok) if ok else 1.0

    if not item.answerable or not (item.expected_sources or item.evidence):
        return scores  # nothing to retrieve

    relevant = [is_relevant(c, item) for c in top]
    first = next((rank for rank, rel in enumerate(relevant, start=1) if rel), None)
    scores["hit_rate"] = 1.0 if first else 0.0
    scores["mrr"] = 1.0 / first if first else 0.0
    scores["precision"] = sum(relevant) / len(top) if top else 0.0

    if item.expected_sources:
        sources = [str(c.metadata.get("source", "")) for c in top]
        found = [e for e in item.expected_sources if any(source_matches(s, e) for s in sources)]
        scores["recall"] = len(found) / len(item.expected_sources)

    # binary nDCG; the ideal ranking has one relevant chunk per expected source / evidence
    ideal_hits = min(k, max(len(item.expected_sources), len(item.evidence), 1))
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(relevant, start=1) if rel)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    scores["ndcg"] = min(dcg / idcg, 1.0) if idcg else 0.0

    if item.evidence:
        context = " ".join(c.text for c in top)
        found_ev = [e for e in item.evidence if contains_evidence(context, [e])]
        scores["evidence_recall"] = len(found_ev) / len(item.evidence)
    return scores
