"""Answer metrics computed without an LLM: overlap with the reference, required
keywords, citation checks and refusal behaviour."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.evaluation.dataset import QAItem
from rag_chatbot.evaluation.metrics.retrieval import is_relevant
from rag_chatbot.generation.generator import NO_ANSWER
from rag_chatbot.generation.schemas import RAGAnswer

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_CITE = re.compile(r"\[(\d+)\]")
_THOUSANDS = re.compile(r"(?<=\d)[,  ](?=\d{3}\b)")
REFUSAL_PHRASES = (
    "couldn't find",
    "could not find",
    "cannot find",
    "can't find",
    "not found in the",
    "no information",
    "not mentioned",
    "not contain",
    "doesn't contain",
    "does not contain",
    "don't know",
    "do not know",
    "not provided in",
    "unable to find",
    "not available in the",
)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles and extra spaces."""
    text = _CITE.sub(" ", text.lower())
    text = _THOUSANDS.sub("", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    norm = normalize_answer(answer)
    found = [k for k in keywords if normalize_answer(k) in norm]
    return len(found) / len(keywords) if keywords else 1.0


def is_refusal(answer: RAGAnswer) -> bool:
    text = answer.answer.lower()
    if NO_ANSWER.lower() in text:
        return True
    return answer.confidence == "low" and any(p in text for p in REFUSAL_PHRASES)


def citation_scores(answer: RAGAnswer, chunks: Sequence[RetrievedChunk], item: QAItem) -> dict[str, float]:
    """citation_validity: share of [n] markers that point at a retrieved chunk;
    citation_precision: share of cited chunks that are relevant to the item."""
    markers = [int(m) for m in _CITE.findall(answer.answer)]
    cited = sorted({c.id for c in answer.citations} | {m for m in markers if 1 <= m <= len(chunks)})
    scores: dict[str, float] = {}
    if markers:
        scores["citation_validity"] = sum(1 <= m <= len(chunks) for m in markers) / len(markers)
    if cited and item.answerable and (item.expected_sources or item.evidence):
        scores["citation_precision"] = sum(is_relevant(chunks[i - 1], item) for i in cited) / len(cited)
    return scores


def answer_scores(answer: RAGAnswer, chunks: Sequence[RetrievedChunk], item: QAItem) -> dict[str, float]:
    refused = is_refusal(answer)
    scores: dict[str, float] = {"refusal_accuracy": float(refused != item.answerable)}
    if item.answerable:
        scores["false_refusal_rate"] = float(refused)
        if item.ground_truth:
            scores["exact_match"] = exact_match(answer.answer, item.ground_truth)
            scores["token_f1"] = token_f1(answer.answer, item.ground_truth)
        if item.answer_must_contain:
            scores["keyword_coverage"] = keyword_coverage(answer.answer, item.answer_must_contain)
    else:
        scores["missed_refusal_rate"] = float(not refused)
    scores.update(citation_scores(answer, chunks, item))
    return scores
