"""Retrieval metrics (no LLM) and LLM-as-judge answer metrics."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from rag.core.types import RetrievedChunk
from rag.generation import prompts
from rag.generation.context import format_context
from rag.generation.schemas import JudgeScores

RETRIEVAL_METRICS = ("hit_rate", "mrr", "recall")
JUDGE_METRICS = ("faithfulness", "answer_relevance", "correctness")


def source_matches(source: str, expected: str) -> bool:
    """Expected sources match as a path suffix: 'sample.md' matches 'docs/sample.md'."""
    source, expected = source.replace("\\", "/"), expected.replace("\\", "/").lstrip("./")
    return source == expected or source.endswith("/" + expected)


def retrieval_scores(chunks: Sequence[RetrievedChunk], expected: Sequence[str]) -> dict[str, float]:
    sources = [str(c.metadata.get("source", "")) for c in chunks]
    first = next(
        (i for i, s in enumerate(sources, start=1) if any(source_matches(s, e) for e in expected)), None
    )
    found = {e for e in expected if any(source_matches(s, e) for s in sources)}
    return {
        "hit_rate": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "recall": len(found) / len(expected),
    }


def judge(
    llm: BaseChatModel,
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    reference: str | None,
    max_context_tokens: int = 6000,
) -> dict[str, float]:
    context, _ = format_context(chunks, max_context_tokens)
    chain = prompts.JUDGE | llm.with_structured_output(JudgeScores)
    scores: JudgeScores = chain.invoke(  # type: ignore[assignment]
        {
            "question": question,
            "context": context or "(none)",
            "answer": answer,
            "reference": reference or "(none)",
        }
    )
    out = {"faithfulness": scores.faithfulness, "answer_relevance": scores.answer_relevance}
    if reference:
        out["correctness"] = scores.correctness
    return out
