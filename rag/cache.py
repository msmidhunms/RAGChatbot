"""Persistent embedding cache.

``CachedEmbeddings`` wraps any LangChain ``Embeddings`` and stores document
vectors in a SQLite file keyed by ``sha256(namespace + text)``. The namespace
is the provider and model, so switching embedding models never returns stale
vectors. Re-ingesting unchanged text then costs no API calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag.core.logging import get_logger

log = get_logger("cache")


class SQLiteEmbeddingStore:
    """Tiny key -> vector store backed by one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._conn.commit()

    def mget(self, keys: Sequence[str]) -> list[list[float] | None]:
        found: dict[str, list[float]] = {}
        with self._lock:
            # stay under SQLite's bound-parameter limit
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                marks = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", batch
                ).fetchall()
                found.update((k, json.loads(v)) for k, v in rows)
        return [found.get(k) for k in keys]

    def mset(self, items: Iterable[tuple[str, list[float]]]) -> None:
        rows = [(k, json.dumps(v)) for k, v in items]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", rows)
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbeddings(Embeddings):
    """Embed documents through a persistent cache, in batches of ``batch_size``."""

    def __init__(
        self,
        base: Embeddings,
        store: SQLiteEmbeddingStore,
        namespace: str,
        *,
        batch_size: int = 64,
        cache_queries: bool = False,
    ) -> None:
        self.base = base
        self.store = store
        self.namespace = namespace
        self.batch_size = batch_size
        self.cache_queries = cache_queries

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        vectors = self.store.mget(keys)

        # embed each distinct missing text once
        missing: dict[str, str] = {}
        for key, text, vec in zip(keys, texts, vectors, strict=True):
            if vec is None:
                missing.setdefault(key, text)
        if missing:
            log.debug("embedding cache: %d hits, %d misses", len(texts) - len(missing), len(missing))
            miss_keys = list(missing)
            new: dict[str, list[float]] = {}
            for start in range(0, len(miss_keys), self.batch_size):
                batch = miss_keys[start : start + self.batch_size]
                embedded = self.base.embed_documents([missing[k] for k in batch])
                batch_items = list(zip(batch, embedded, strict=True))
                self.store.mset(batch_items)
                new.update(batch_items)
            vectors = [vec if vec is not None else new[key] for key, vec in zip(keys, vectors, strict=True)]
        return vectors  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        if not self.cache_queries:
            return self.base.embed_query(text)
        key = self._key("query\x00" + text)
        cached = self.store.mget([key])[0]
        if cached is not None:
            return cached
        vec = self.base.embed_query(text)
        self.store.mset([(key, vec)])
        return vec


class NormalizedEmbeddings(Embeddings):
    """Scale every vector to unit length (cosine == dot product)."""

    def __init__(self, base: Embeddings) -> None:
        self.base = base

    @staticmethod
    def _unit(vec: list[float]) -> list[float]:
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else list(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._unit(v) for v in self.base.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._unit(self.base.embed_query(text))
