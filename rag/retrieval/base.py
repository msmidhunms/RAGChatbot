"""Retriever protocol and helpers shared by retrieval stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from rag.core.types import RetrievedChunk

Filter = Mapping[str, Any] | None
LLMGetter = Callable[[], BaseChatModel]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]: ...


def history_block(history: str) -> str:
    """Prefix for prompts that optionally include the conversation."""
    return f"Conversation so far:\n{history}\n\n" if history.strip() else ""


def excerpts(chunks: Sequence[RetrievedChunk], max_chars: int = 1500) -> str:
    """Numbered excerpts ([1] ...) for grading and reranking prompts."""
    return "\n\n".join(f"[{i}] {c.text[:max_chars]}" for i, c in enumerate(chunks, start=1))
