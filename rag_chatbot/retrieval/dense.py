"""Vector search: similarity, MMR or score-threshold."""

from __future__ import annotations

from rag_chatbot.config.schema import RetrievalConfig
from rag_chatbot.core.types import RetrievedChunk, rerank
from rag_chatbot.retrieval.base import Filter
from rag_chatbot.stores.base import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, cfg: RetrievalConfig) -> None:
        self.store = store
        self.cfg = cfg

    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]:
        cfg = self.cfg
        if cfg.search_type == "mmr":
            hits = self.store.mmr_search(query, k, max(cfg.fetch_k, k), cfg.mmr_lambda, flt)
        else:
            hits = self.store.search(query, k, flt)
            if cfg.search_type == "threshold" and cfg.score_threshold is not None:
                hits = [h for h in hits if h.score >= cfg.score_threshold]
        return rerank([RetrievedChunk(h.document, h.score, source="dense") for h in hits])
