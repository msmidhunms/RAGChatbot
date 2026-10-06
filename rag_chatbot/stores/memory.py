"""Non-persistent in-process store; handy for tests and quick experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.config.schema import VectorStoreConfig
from rag_chatbot.stores.base import STORES, Hit, VectorStore
from rag_chatbot.stores.filters import matches


class MemoryStore(VectorStore):
    kind = "memory"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._docs: dict[str, tuple[Document, list[float]]] = {}
        self._meta: dict[str, Any] = {}

    # metadata lives in memory too, so nothing outlives the process
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def _record_meta(self, dimension: int) -> None:
        if self._meta.get("dimension") not in (None, dimension):
            super()._record_meta(dimension)  # raises the dimension error
        self._meta = {"embedding": self.namespace, "dimension": dimension}

    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        for doc, id_, vec in zip(docs, ids, vectors, strict=True):
            self._docs[id_] = (
                Document(page_content=doc.page_content, metadata=dict(doc.metadata), id=id_),
                vec,
            )

    def delete(self, ids: Sequence[str]) -> None:
        for id_ in ids:
            self._docs.pop(id_, None)

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        items = [(d, v) for d, v in self._docs.values() if matches(d.metadata, flt)]
        if not items:
            return []
        mat = np.asarray([v for _, v in items], dtype=float)
        q = np.asarray(vector, dtype=float)
        if self.cfg.distance == "l2":
            scores = 1.0 / (1.0 + np.sqrt(((mat - q) ** 2).sum(axis=1)))
        elif self.cfg.distance == "ip":
            scores = mat @ q
        else:
            norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
            scores = (mat @ q) / np.where(norms == 0, 1, norms)
        order = np.argsort(-scores)[:k]
        return [Hit(items[i][0], float(scores[i]), items[i][1] if with_vectors else None) for i in order]

    def count(self) -> int:
        return len(self._docs)

    def _reset(self) -> None:
        self._docs.clear()
        self._meta = {}


@STORES.register("memory")
def _memory(cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path) -> VectorStore:
    return MemoryStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
