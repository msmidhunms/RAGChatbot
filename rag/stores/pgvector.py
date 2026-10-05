"""Postgres + pgvector backend via ``langchain_postgres.PGVector``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config.schema import VectorStoreConfig
from rag.core.exceptions import MissingCredentialsError
from rag.core.registry import require
from rag.stores.base import STORES, Hit, VectorStore, similarity_from_distance
from rag.stores.filters import to_pgvector


class PGVectorStore(VectorStore):
    kind = "pgvector"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mod = require("langchain_postgres", "pgvector")
        env = self.cfg.pgvector.connection_env
        conn = os.environ.get(env)
        if not conn:
            raise MissingCredentialsError(f"{env} is not set but required by vector_store.type=pgvector")
        strategy = {"cosine": "cosine", "l2": "euclidean", "ip": "inner"}[self.cfg.distance]
        self._vs = mod.PGVector(
            embeddings=self.embeddings,
            connection=conn,
            collection_name=self.collection,
            use_jsonb=True,
            distance_strategy=strategy,
        )

    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        self._vs.add_embeddings(
            texts=[d.page_content for d in docs],
            embeddings=vectors,
            metadatas=[d.metadata for d in docs],
            ids=ids,
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._vs.delete(ids=list(ids))

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        results = self._vs.similarity_search_with_score_by_vector(vector, k=k, filter=to_pgvector(flt))
        # PGVector cannot return stored vectors; MMR re-embeds via the embedding cache
        return [Hit(doc, similarity_from_distance(self.cfg.distance, dist)) for doc, dist in results]

    def count(self) -> int | None:
        return None  # not exposed by langchain_postgres; see docstore counts

    def _reset(self) -> None:
        self._vs.delete_collection()
        self._vs.create_collection()


@STORES.register("pgvector")
def _pgvector(cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path) -> VectorStore:
    return PGVectorStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
