"""Pydantic schemas for every structured LLM output.

``LLMAnswer`` is what the model fills in; ``RAGAnswer`` is what callers get,
with citations resolved to their source and page from chunk metadata
(the model only names excerpt numbers, it never invents source names).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]


class LLMAnswer(BaseModel):
    """Answer to the user's question based on the numbered context excerpts."""

    title: str = Field(description="A short title (max 8 words) for the answer")
    answer: str = Field(description="The answer, citing excerpts like [1] where they support a claim")
    citations: list[int] = Field(
        default_factory=list, description="Numbers of the excerpts the answer relies on"
    )
    confidence: Confidence = Field(description="How well the context supports the answer")


class Citation(BaseModel):
    id: int
    source: str
    page: int | None = None
    section: str | None = None
    snippet: str = ""


class RAGAnswer(BaseModel):
    title: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = "medium"
    grounded: bool | None = None  # set when generation.self_check is enabled


class QueryList(BaseModel):
    """Alternative search queries."""

    queries: list[str] = Field(description="Distinct standalone search queries")


class ExcerptScore(BaseModel):
    id: int = Field(description="Excerpt number")
    score: float = Field(description="Relevance from 0 to 10")


class RelevanceScores(BaseModel):
    """Relevance score for each excerpt."""

    scores: list[ExcerptScore]


class GradeResult(BaseModel):
    """Which excerpts are relevant to the question."""

    relevant: list[int] = Field(default_factory=list, description="Numbers of the relevant excerpts")


class GroundednessResult(BaseModel):
    """Whether the answer is supported by the context."""

    grounded: bool = Field(description="True if every claim is supported by the context")
    issues: str = Field(default="", description="Unsupported claims, if any")


class JudgeScores(BaseModel):
    """Evaluation scores between 0 and 1."""

    faithfulness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    context_recall: float = Field(
        1.0, ge=0, le=1, description="Share of the reference answer's facts present in the context"
    )
