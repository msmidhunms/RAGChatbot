"""LLM-as-judge metrics. Only run when one of them is selected."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.evaluation.dataset import QAItem
from rag_chatbot.generation import prompts
from rag_chatbot.generation.context import format_context
from rag_chatbot.generation.schemas import GradeResult, JudgeScores
from rag_chatbot.retrieval.base import excerpts


def judge_scores(
    llm: BaseChatModel,
    item: QAItem,
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    wanted: set[str],
    max_context_tokens: int = 6000,
) -> dict[str, float]:
    """faithfulness, answer_relevance, correctness, context_recall (one call) and
    context_precision (one grading call over the numbered chunks)."""
    scores: dict[str, float] = {}
    if wanted & {"faithfulness", "answer_relevance", "correctness", "context_recall"}:
        context, _ = format_context(chunks, max_context_tokens)
        chain = prompts.JUDGE | llm.with_structured_output(JudgeScores)
        result: JudgeScores = chain.invoke(  # type: ignore[assignment]
            {
                "question": question,
                "context": context or "(none)",
                "answer": answer,
                "reference": item.ground_truth or "(none)",
            }
        )
        scores["faithfulness"] = result.faithfulness
        scores["answer_relevance"] = result.answer_relevance
        if item.ground_truth:
            scores["correctness"] = result.correctness
            scores["context_recall"] = result.context_recall
    if "context_precision" in wanted and chunks:
        chain = prompts.GRADE | llm.with_structured_output(GradeResult)
        graded: GradeResult = chain.invoke(  # type: ignore[assignment]
            {"question": question, "context": excerpts(chunks)}
        )
        relevant = {i for i in graded.relevant if 1 <= i <= len(chunks)}
        scores["context_precision"] = len(relevant) / len(chunks)
    return {k: v for k, v in scores.items() if k in wanted}
