"""Chroma backend (persistent local client)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.config.schema import VectorStoreConfig
from rag_chatbot.core.registry import require
from rag_chatbot.stores.base import STORES, Hit, VectorStore, from_cosine_distance, from_l2
from rag_chatbot.stores.filters import to_chroma


class ChromaStore(VectorStore):
    kind = "chroma"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        chromadb = require("chromadb")
        path = Path(self.cfg.chroma.persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._collection()

    def _collection(self) -> Any:
        return self._client.get_or_create_collection(
            self.collection, metadata={"hnsw:space": self.cfg.distance}
        )

    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        self._col.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[d.page_content for d in docs],
            metadatas=[d.metadata or {"_": ""} for d in docs],  # chroma rejects empty metadata
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._col.delete(ids=list(ids))

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        if self._col.count() == 0:
            return []
        include = ["documents", "metadatas", "distances"] + (["embeddings"] if with_vectors else [])
        res = self._col.query(
            query_embeddings=[vector],
            n_results=min(k, self._col.count()),
            where=to_chroma(flt),
            include=include,
        )
        hits = []
        for i, id_ in enumerate(res["ids"][0]):
            doc = Document(page_content=res["documents"][0][i], metadata=res["metadatas"][0][i] or {}, id=id_)
            vec = list(res["embeddings"][0][i]) if with_vectors else None
            hits.append(Hit(doc, self._score(res["distances"][0][i]), vec))
        return hits

    def _score(self, distance: float) -> float:
        if self.cfg.distance == "l2":
            return from_l2(distance, squared=True)  # chroma reports squared L2
        return from_cosine_distance(distance)  # cosine: 1 - cos; ip: 1 - dot

    def count(self) -> int:
        return self._col.count()

    def _reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._col = self._collection()


@STORES.register("chroma")
def _chroma(cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path) -> VectorStore:
    return ChromaStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
