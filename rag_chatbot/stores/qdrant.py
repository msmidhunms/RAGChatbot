"""Qdrant backend (server via ``url`` or embedded local mode via ``path``).

Points use the same payload layout as ``langchain_qdrant``
(``page_content`` + ``metadata``), so collections are interchangeable.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.config.schema import VectorStoreConfig
from rag_chatbot.core.registry import require
from rag_chatbot.stores.base import STORES, Hit, VectorStore, from_l2
from rag_chatbot.stores.filters import to_qdrant

CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUID or int ids; chunk ids are 32 hex chars, i.e. a UUID."""
    try:
        return str(uuid.UUID(hex=chunk_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore(VectorStore):
    kind = "qdrant"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        qdrant_client = require("qdrant_client", "qdrant")
        self._models = qdrant_client.models
        q = self.cfg.qdrant
        if q.url:
            api_key = os.environ.get(q.api_key_env) if q.api_key_env else None
            self._client = qdrant_client.QdrantClient(url=q.url, api_key=api_key or None)
        else:
            assert q.path is not None  # guaranteed by config validation
            Path(q.path).mkdir(parents=True, exist_ok=True)
            self._client = qdrant_client.QdrantClient(path=str(q.path))

    def _distance(self) -> Any:
        d = self._models.Distance
        return {"cosine": d.COSINE, "l2": d.EUCLID, "ip": d.DOT}[self.cfg.distance]

    def _exists(self) -> bool:
        return bool(self._client.collection_exists(self.collection))

    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        if not self._exists():
            self._client.create_collection(
                self.collection,
                vectors_config=self._models.VectorParams(size=len(vectors[0]), distance=self._distance()),
            )
        points = [
            self._models.PointStruct(
                id=point_id(id_),
                vector=vec,
                payload={CONTENT_KEY: doc.page_content, METADATA_KEY: {**doc.metadata, "chunk_id": id_}},
            )
            for doc, id_, vec in zip(docs, ids, vectors, strict=True)
        ]
        self._client.upsert(self.collection, points=points)

    def delete(self, ids: Sequence[str]) -> None:
        if ids and self._exists():
            self._client.delete(
                self.collection, points_selector=self._models.PointIdsList(points=[point_id(i) for i in ids])
            )

    def _score(self, raw: float) -> float:
        if self.cfg.distance == "l2":
            return from_l2(raw, squared=False)  # qdrant reports plain euclidean distance
        return raw  # cosine similarity / dot product as-is

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        if not self._exists():
            return []
        res = self._client.query_points(
            self.collection,
            query=vector,
            limit=k,
            query_filter=to_qdrant(flt, METADATA_KEY),
            with_payload=True,
            with_vectors=with_vectors,
        )
        hits = []
        for p in res.points:
            meta = dict(p.payload.get(METADATA_KEY) or {})
            doc = Document(
                page_content=p.payload.get(CONTENT_KEY, ""), metadata=meta, id=meta.get("chunk_id")
            )
            vec = list(p.vector) if with_vectors and isinstance(p.vector, list) else None
            hits.append(Hit(doc, self._score(float(p.score)), vec))
        return hits

    def count(self) -> int:
        return int(self._client.count(self.collection).count) if self._exists() else 0

    def _reset(self) -> None:
        if self._exists():
            self._client.delete_collection(self.collection)


@STORES.register("qdrant")
def _qdrant(cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path) -> VectorStore:
    return QdrantStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
