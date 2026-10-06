"""SQLite store for chunk text and metadata.

It is the corpus for BM25, holds parent chunks for parent-document
retrieval, and is the source of truth for ``rag store stats``. A version
counter changes on every write so in-memory indexes know when to rebuild.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag_chatbot.core.types import chunk_key
from rag_chatbot.stores.filters import matches

CHUNK = "chunk"
PARENT = "parent"


class SQLiteDocStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
            CREATE INDEX IF NOT EXISTS chunks_kind ON chunks (kind);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '0');
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------ writes
    def _bump(self) -> None:
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'version'")

    def put(self, docs: Iterable[Document], kind: str = CHUNK) -> int:
        rows = [
            (chunk_key(d), d.metadata.get("doc_id"), kind, d.page_content, json.dumps(d.metadata))
            for d in docs
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, kind, content, metadata) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            self._bump()
            self._conn.commit()
        return len(rows)

    def delete_doc(self, doc_id: str) -> list[str]:
        """Delete every chunk and parent of a document; return the deleted chunk ids (kind=chunk)."""
        with self._lock:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ? AND kind = ?", (doc_id, CHUNK)
                )
            ]
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._bump()
            self._conn.commit()
        return ids

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._bump()
            self._conn.commit()

    # ------------------------------------------------------------- reads
    @staticmethod
    def _doc(row: Sequence[Any]) -> Document:
        chunk_id, content, metadata = row
        return Document(page_content=content, metadata=json.loads(metadata), id=chunk_id)

    def get_many(self, ids: Sequence[str]) -> list[Document]:
        """Documents for ``ids`` in the given order; unknown ids are skipped."""
        found: dict[str, Document] = {}
        with self._lock:
            for start in range(0, len(ids), 500):
                batch = list(ids[start : start + 500])
                marks = ",".join("?" * len(batch))
                for row in self._conn.execute(
                    f"SELECT chunk_id, content, metadata FROM chunks WHERE chunk_id IN ({marks})", batch
                ):
                    found[row[0]] = self._doc(row)
        return [found[i] for i in ids if i in found]

    def all(self, kind: str = CHUNK, flt: Mapping[str, Any] | None = None) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, content, metadata FROM chunks WHERE kind = ? ORDER BY rowid", (kind,)
            ).fetchall()
        docs = [self._doc(r) for r in rows]
        return [d for d in docs if matches(d.metadata, flt)] if flt else docs

    def count(self, kind: str = CHUNK) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks WHERE kind = ?", (kind,)).fetchone()[0]

    def sources(self) -> dict[str, int]:
        """Chunk count per source."""
        out: dict[str, int] = {}
        for doc in self.all():
            src = str(doc.metadata.get("source", "?"))
            out[src] = out.get(src, 0) + 1
        return out

    def version(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
