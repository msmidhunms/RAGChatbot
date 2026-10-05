"""Conversation memory: which past messages the model sees, and where sessions persist."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.config.schema import MemoryConfig
from rag.core.registry import require
from rag.generation import prompts
from rag.retrieval.base import LLMGetter


def select_history(past: Sequence[BaseMessage], cfg: MemoryConfig) -> list[BaseMessage]:
    """Past messages (excluding the current question) to show the model."""
    if cfg.type == "none":
        return []
    if cfg.type == "buffer":
        return list(past)
    return list(past[-2 * cfg.window_size :])  # window and summary keep the last N turns verbatim


def format_messages(messages: Sequence[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else m.type
        lines.append(f"{role}: {m.text}")
    return "\n".join(lines)


def format_history(past: Sequence[BaseMessage], cfg: MemoryConfig, summary: str = "") -> str:
    text = format_messages(select_history(past, cfg))
    if cfg.type == "summary" and summary:
        text = f"Summary of earlier conversation: {summary}\n{text}".strip()
    return text


def update_summary(
    messages: Sequence[BaseMessage], cfg: MemoryConfig, summary: str, summarized_upto: int, get_llm: LLMGetter
) -> tuple[str, int]:
    """Fold messages that left the window into the running summary."""
    keep = 2 * cfg.window_size
    cutoff = len(messages) - keep
    if cfg.type != "summary" or cutoff <= summarized_upto:
        return summary, summarized_upto
    chain = prompts.SUMMARY | get_llm() | StrOutputParser()
    new = chain.invoke(
        {"summary": summary or "(none)", "history": format_messages(messages[summarized_upto:cutoff])}
    )
    return new.strip(), cutoff


def build_checkpointer(cfg: MemoryConfig) -> BaseCheckpointSaver:
    if cfg.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    mod = require("langgraph.checkpoint.sqlite")
    path = Path(cfg.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return mod.SqliteSaver(sqlite3.connect(path, check_same_thread=False))
