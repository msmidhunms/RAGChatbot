"""Assemble the configured retrieval pipeline.

``RetrievalPipeline.run`` = transform -> search each query -> fuse ->
compress -> rerank. The stages are also callable one by one so the LangGraph
nodes can time and branch on them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.embeddings import Embeddings

from rag_chatbot.config.schema import RAGConfig
from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.retrieval.base import Filter, LLMGetter, Retriever
from rag_chatbot.retrieval.compression import build_compressor
from rag_chatbot.retrieval.dense import DenseRetriever
from rag_chatbot.retrieval.hybrid import HybridRetriever, fuse
from rag_chatbot.retrieval.parent import ParentRetriever
from rag_chatbot.retrieval.query_transform import Transform, build_transform
from rag_chatbot.retrieval.rerankers import build_reranker
from rag_chatbot.retrieval.sparse import BM25Retriever
from rag_chatbot.stores.base import VectorStore
from rag_chatbot.stores.docstore import SQLiteDocStore


def build_base_retriever(cfg: RAGConfig, store: VectorStore, docstore: SQLiteDocStore) -> Retriever:
    r = cfg.retrieval
    dense = DenseRetriever(store, r)
    if r.strategy == "dense":
        return dense
    if r.strategy == "sparse":
        return BM25Retriever(docstore)
    if r.strategy == "hybrid":
        return HybridRetriever(dense, BM25Retriever(docstore), r.hybrid)
    return ParentRetriever(dense, docstore)


class RetrievalPipeline:
    def __init__(
        self,
        cfg: RAGConfig,
        store: VectorStore,
        docstore: SQLiteDocStore,
        embeddings: Embeddings,
        get_llm: LLMGetter,
    ) -> None:
        self.cfg = cfg
        self.base = build_base_retriever(cfg, store, docstore)
        self._get_llm = get_llm
        self.transformer: Transform = build_transform(cfg.query_transform, get_llm)
        self.compressor = build_compressor(cfg.compression, embeddings, get_llm)
        self.reranker = build_reranker(cfg.reranker, get_llm)

    @property
    def has_postprocess(self) -> bool:
        return self.cfg.compression.type != "none" or self.cfg.reranker.type != "none"

    def filter_for(self, flt: Filter) -> dict[str, Any]:
        return {**self.cfg.retrieval.filters, **(flt or {})}

    def transform(self, question: str, history: str = "", *, mode: str | None = None) -> list[str]:
        if mode:
            return build_transform(self.cfg.query_transform, self._get_llm, mode)(question, history)
        return self.transformer(question, history)

    def search(self, queries: list[str], flt: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        k = self.cfg.retrieval.k
        merged = self.filter_for(flt) or None
        results = [self.base.retrieve(q, k, merged) for q in queries]
        if len(results) == 1:
            return results[0]
        return fuse(results, method="rrf", rrf_k=self.cfg.retrieval.hybrid.rrf_k, k=k)

    def postprocess(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return self.reranker(question, self.compressor(question, chunks))

    def run(
        self, question: str, history: str = "", flt: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return self.postprocess(question, self.search(self.transform(question, history), flt))
