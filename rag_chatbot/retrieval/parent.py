"""Small-to-big retrieval: match small child chunks, return their parent chunks."""

from __future__ import annotations

from rag_chatbot.core.logging import get_logger
from rag_chatbot.core.types import RetrievedChunk, rerank
from rag_chatbot.retrieval.base import Filter, Retriever
from rag_chatbot.stores.docstore import SQLiteDocStore

log = get_logger("retrieval")


class ParentRetriever:
    def __init__(self, child: Retriever, docstore: SQLiteDocStore, child_factor: int = 4) -> None:
        self.child = child
        self.docstore = docstore
        self.child_factor = child_factor
        self._warned = False

    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]:
        children = self.child.retrieve(query, k * self.child_factor, flt)
        parent_ids = list(
            dict.fromkeys(c.metadata["parent_id"] for c in children if c.metadata.get("parent_id"))
        )
        parents = {p.metadata.get("chunk_id"): p for p in self.docstore.get_many(parent_ids)}
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        orphans = 0
        for c in children:  # best-first: each parent takes its best child's score
            pid = c.metadata.get("parent_id")
            if pid is not None and pid in parents:
                if pid not in seen:
                    seen.add(pid)
                    results.append(RetrievedChunk(parents[pid], c.score, source="parent"))
            else:
                # indexed without parent chunks (e.g. strategy switched without re-ingesting)
                orphans += 1
                results.append(RetrievedChunk(c.document, c.score, source=c.source or "dense"))
            if len(results) >= k:
                break
        if orphans and not self._warned:
            self._warned = True
            log.warning(
                "retrieval.strategy is 'parent' but the index has no parent chunks for some results; "
                "run `rag ingest --reset` to rebuild it for parent retrieval"
            )
        return rerank(results)
