"""BM25 keyword search over the docstore (rank_bm25)."""

from __future__ import annotations

import re
import threading

from langchain_core.documents import Document

from rag.core.registry import require
from rag.core.types import RetrievedChunk, rerank
from rag.retrieval.base import Filter
from rag.stores.docstore import SQLiteDocStore
from rag.stores.filters import matches

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Index is rebuilt lazily whenever the docstore changes."""

    def __init__(self, docstore: SQLiteDocStore) -> None:
        self.docstore = docstore
        self._lock = threading.Lock()
        self._version = -1
        self._docs: list[Document] = []
        self._terms: list[set[str]] = []
        self._bm25: object | None = None

    def _ensure_index(self) -> None:
        version = self.docstore.version()
        with self._lock:
            if version == self._version:
                return
            self._docs = self.docstore.all()
            tokens = [tokenize(d.page_content) for d in self._docs]
            self._terms = [set(t) for t in tokens]
            if self._docs:
                bm25 = require("rank_bm25")
                self._bm25 = bm25.BM25Okapi(tokens)
            else:
                self._bm25 = None
            self._version = version

    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]:
        self._ensure_index()
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)  # type: ignore[attr-defined]
        wanted = set(terms)
        # require a shared term rather than score > 0: Okapi IDF is 0 for a term in half the corpus
        ranked = sorted(
            (
                i
                for i in range(len(self._docs))
                if wanted & self._terms[i] and matches(self._docs[i].metadata, flt)
            ),
            key=lambda i: -scores[i],
        )[:k]
        return rerank([RetrievedChunk(self._docs[i], float(scores[i]), source="bm25") for i in ranked])
