"""Graph state. Values are plain JSON so any checkpointer can persist them."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from rag.core.types import RetrievedChunk


class RAGState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # whole conversation
    question: str
    standalone_question: str
    filter: dict[str, Any]
    queries: list[str]
    chunks: list[dict[str, Any]]  # serialized RetrievedChunk
    answer: dict[str, Any] | None  # serialized RAGAnswer
    retries: int  # query rewrites after failed grading
    rewrite: bool  # set by grade when the query should be rewritten and retried
    check_retries: int  # regenerations after failed self-check
    feedback: str
    summary: str
    summarized_upto: int
    timings: dict[str, float]


def dump_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.document.id,
            "text": c.text,
            "metadata": c.metadata,
            "score": c.score,
            "rank": c.rank,
            "source": c.source,
        }
        for c in chunks
    ]


def load_chunks(data: list[dict[str, Any]] | None) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            Document(page_content=d["text"], metadata=d["metadata"], id=d.get("id")),
            d["score"],
            d["rank"],
            d["source"],
        )
        for d in data or []
    ]
