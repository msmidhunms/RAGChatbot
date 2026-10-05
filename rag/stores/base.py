"""Vector store abstraction shared by every backend.

Backends implement ``_upsert``, ``delete``, ``_query``, ``count`` and
``_reset``. The base class embeds text exactly once per call, records which
embedding model built the collection (so a later model switch fails loudly
instead of returning garbage), and implements MMR for every backend.
Scores returned by ``search`` are always "higher is better".
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config.schema import VectorStoreConfig
from rag.core.exceptions import ConfigError
from rag.core.registry import Registry
from rag.core.types import chunk_key


@dataclass
class Hit:
    document: Document
    score: float
    vector: list[float] | None = None


def similarity_from_distance(distance: str, value: float) -> float:
    """Map a backend distance to a similarity where higher is better."""
    if distance == "cosine":
        return 1.0 - value  # cosine distance = 1 - cos
    if distance == "ip":
        return 1.0 - value  # chroma/pg inner-product distance = 1 - dot
    return 1.0 / (1.0 + value)  # l2


def mmr_select(query: np.ndarray, candidates: np.ndarray, k: int, lambda_mult: float) -> list[int]:
    """Maximal marginal relevance over cosine similarity; returns candidate indexes."""
    if len(candidates) == 0:
        return []

    def unit(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    cands = unit(candidates)
    rel = cands @ unit(query)
    selected = [int(np.argmax(rel))]
    while len(selected) < min(k, len(cands)):
        redundancy = (cands @ cands[selected].T).max(axis=1)
        score = lambda_mult * rel - (1 - lambda_mult) * redundancy
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


class VectorStore(ABC):
    kind = "base"

    def __init__(
        self, cfg: VectorStoreConfig, embeddings: Embeddings, *, namespace: str, meta_dir: Path
    ) -> None:
        self.cfg = cfg
        self.embeddings = embeddings
        self.namespace = namespace
        self.collection = cfg.collection
        self._meta_path = Path(meta_dir) / f"{self.kind}__{cfg.collection}.json"
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    # ---------------------------------------------------------- metadata
    def meta(self) -> dict[str, Any]:
        if self._meta_path.is_file():
            return json.loads(self._meta_path.read_text())
        return {}

    def check_compatible(self) -> None:
        built_with = self.meta().get("embedding")
        if built_with and built_with != self.namespace:
            raise ConfigError(
                f"collection '{self.collection}' ({self.kind}) was built with embeddings '{built_with}' "
                f"but the config uses '{self.namespace}'. Run `rag store reset` to rebuild it, "
                "or set vector_store.collection to a new name."
            )

    def _record_meta(self, dimension: int) -> None:
        meta = self.meta()
        if meta.get("dimension") not in (None, dimension):
            raise ConfigError(
                f"collection '{self.collection}' holds {meta['dimension']}-d vectors, got {dimension}-d"
            )
        if meta.get("embedding") != self.namespace or meta.get("dimension") != dimension:
            self._meta_path.parent.mkdir(parents=True, exist_ok=True)
            self._meta_path.write_text(json.dumps({"embedding": self.namespace, "dimension": dimension}))

    # --------------------------------------------------------------- api
    def add(self, docs: Sequence[Document]) -> list[str]:
        if not docs:
            return []
        self.check_compatible()
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        self._record_meta(len(vectors[0]))
        ids = [chunk_key(d) for d in docs]
        self._upsert(list(docs), ids, vectors)
        return ids

    def embed_query(self, query: str) -> list[float]:
        if query in self._query_cache:
            self._query_cache.move_to_end(query)
            return self._query_cache[query]
        vec = self.embeddings.embed_query(query)
        self._query_cache[query] = vec
        if len(self._query_cache) > 256:
            self._query_cache.popitem(last=False)
        return vec

    def search(self, query: str, k: int, flt: Mapping[str, Any] | None = None) -> list[Hit]:
        self.check_compatible()
        if not self.meta():
            return []  # nothing ingested yet
        return self._query(self.embed_query(query), k, flt)[:k]

    def mmr_search(
        self, query: str, k: int, fetch_k: int, lambda_mult: float, flt: Mapping[str, Any] | None = None
    ) -> list[Hit]:
        self.check_compatible()
        if not self.meta():
            return []
        qvec = self.embed_query(query)
        hits = self._query(qvec, max(fetch_k, k), flt, with_vectors=True)
        missing = [h for h in hits if h.vector is None]
        if missing:  # backends that cannot return vectors: re-embed (served by the embedding cache)
            for h, v in zip(
                missing,
                self.embeddings.embed_documents([h.document.page_content for h in missing]),
                strict=True,
            ):
                h.vector = v
        order = mmr_select(np.asarray(qvec), np.asarray([h.vector for h in hits]), k, lambda_mult)
        return [hits[i] for i in order]

    def reset(self) -> None:
        self._reset()
        self._query_cache.clear()
        self._meta_path.unlink(missing_ok=True)

    # ---------------------------------------------------------- backend
    @abstractmethod
    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...

    @abstractmethod
    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int | None:
        """Number of stored vectors, or None if the backend cannot tell cheaply."""

    @abstractmethod
    def _reset(self) -> None: ...


StoreFactory = Callable[[VectorStoreConfig, Embeddings, str, Path], VectorStore]
STORES: Registry[StoreFactory] = Registry("vector store")


def build_store(
    cfg: VectorStoreConfig, embeddings: Embeddings, *, namespace: str, data_dir: Path
) -> VectorStore:
    return STORES.get(cfg.type)(cfg, embeddings, namespace, Path(data_dir) / "store_meta")
