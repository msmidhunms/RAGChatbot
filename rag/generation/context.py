"""Format retrieved chunks into a numbered, token-budgeted context block."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from rag.core.types import RetrievedChunk


@lru_cache(maxsize=1)
def _encoder() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # encoding files are downloaded on first use; work offline too
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc is not None else (len(text) + 3) // 4


def chunk_label(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    label = str(meta.get("source", "unknown"))
    if meta.get("page") is not None:
        label += f", page {meta['page']}"
    if meta.get("section"):
        label += f", section: {meta['section']}"
    return label


def short_label(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    label = str(meta.get("source", "unknown")).rsplit("/", 1)[-1]
    return f"{label} p.{meta['page']}" if meta.get("page") is not None else label


def format_context(chunks: Sequence[RetrievedChunk], max_tokens: int) -> tuple[str, list[RetrievedChunk]]:
    """Return (context text, chunks actually included), best-ranked first.

    Chunks are added until the budget is reached; if even the first chunk does
    not fit, it is truncated so the model always sees something.
    """
    parts: list[str] = []
    used: list[RetrievedChunk] = []
    budget = max_tokens
    for chunk in chunks:
        block = f"[{len(used) + 1}] ({chunk_label(chunk)})\n{chunk.text.strip()}"
        cost = count_tokens(block)
        if cost > budget:
            if used:
                break
            block = block[: max(budget, 1) * 4]
            cost = budget
        parts.append(block)
        used.append(chunk)
        budget -= cost
    return "\n\n".join(parts), used
