"""Data types passed between pipeline stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_core.documents import Document


def chunk_key(doc: Document) -> str:
    """Stable identity of a chunk: its chunk_id, else its id, else a content hash."""
    key = doc.metadata.get("chunk_id") or doc.id
    return key or hashlib.sha256(doc.page_content.encode()).hexdigest()[:32]


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    rank: int = 0
    source: str = ""  # retriever that produced it: dense, bm25, rrf, rerank:llm, ...

    @property
    def chunk_id(self) -> str:
        return chunk_key(self.document)

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata


def rerank(chunks: list[RetrievedChunk], source: str | None = None) -> list[RetrievedChunk]:
    """Renumber ranks 1..n in the given order, optionally relabelling the source."""
    for i, chunk in enumerate(chunks, start=1):
        chunk.rank = i
        if source:
            chunk.source = source
    return chunks


@dataclass
class IngestReport:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors
