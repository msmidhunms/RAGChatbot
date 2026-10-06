"""Small-to-big retrieval: match small child chunks, return their parent chunks."""

from __future__ import annotations

from rag_chatbot.core.types import RetrievedChunk, rerank
from rag_chatbot.retrieval.base import Filter, Retriever
from rag_chatbot.stores.docstore import SQLiteDocStore


class ParentRetriever:
    def __init__(self, child: Retriever, docstore: SQLiteDocStore, child_factor: int = 4) -> None:
        self.child = child
        self.docstore = docstore
        self.child_factor = child_factor

    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]:
        children = self.child.retrieve(query, k * self.child_factor, flt)
        best: dict[str, float] = {}
        for c in children:  # children arrive best-first; keep each parent's best score
            pid = c.metadata.get("parent_id")
            if pid and pid not in best:
                best[pid] = c.score
        order = list(best)[:k]
        parents = {p.metadata.get("chunk_id"): p for p in self.docstore.get_many(order)}
        return rerank(
            [RetrievedChunk(parents[pid], best[pid], source="parent") for pid in order if pid in parents]
        )
