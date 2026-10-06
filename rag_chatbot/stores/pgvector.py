"""Postgres + pgvector backend via ``langchain_postgres.PGVector``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.config.schema import VectorStoreConfig
from rag_chatbot.core.exceptions import MissingCredentialsError
from rag_chatbot.core.registry import require
from rag_chatbot.stores.base import STORES, Hit, VectorStore, from_cosine_distance, from_l2
from rag_chatbot.stores.filters import to_pgvector


class PGVectorStore(VectorStore):
    kind = "pgvector"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mod = require("langchain_postgres", "pgvector")
        env = self.cfg.pgvector.connection_env
        conn = os.environ.get(env)
        if not conn:
            raise MissingCredentialsError(f"{env} is not set but required by vector_store.type=pgvector")
        strategy = {"cosine": "cosine", "l2": "l2", "ip": "inner"}[self.cfg.distance]
        self._vs = mod.PGVector(
            embeddings=self.embeddings,
            connection=conn,
            collection_name=self.collection,
            use_jsonb=True,
            distance_strategy=strategy,
        )

    def _row_id(self, chunk_id: str) -> str:
        # langchain_postgres keys rows by id across *all* collections, so an upsert of
        # the same chunk into a second collection would move it; namespace the ids.
        return f"{self.collection}::{chunk_id}"

    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        self._vs.add_embeddings(
            texts=[d.page_content for d in docs],
            embeddings=vectors,
            metadatas=[{**d.metadata, "chunk_id": id_} for d, id_ in zip(docs, ids, strict=True)],
            ids=[self._row_id(i) for i in ids],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._vs.delete(ids=[self._row_id(i) for i in ids])

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        results = self._vs.similarity_search_with_score_by_vector(vector, k=k, filter=to_pgvector(flt))
        # PGVector cannot return stored vectors; MMR re-embeds via the embedding cache
        hits = []
        for doc, dist in results:
            doc.id = doc.metadata.get("chunk_id", doc.id)
            hits.append(Hit(doc, self._score(dist)))
        return hits

    def _score(self, distance: float) -> float:
        if self.cfg.distance == "l2":
            return from_l2(distance, squared=False)
        if self.cfg.distance == "ip":
            return -float(distance)  # pgvector <#> is the negative inner product
        return from_cosine_distance(distance)

    def count(self) -> int | None:
        return None  # not exposed by langchain_postgres; see docstore counts

    def _reset(self) -> None:
        self._vs.delete_collection()
        self._vs.create_collection()


@STORES.register("pgvector")
def _pgvector(cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path) -> VectorStore:
    return PGVectorStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
