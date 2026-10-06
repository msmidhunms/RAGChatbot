"""Identity, hashing and metadata enrichment for documents and chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag_chatbot.ingestion.loaders import is_url


def sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def normalize_source(source: str) -> str:
    """Canonical source string: URLs as-is, paths relative to cwd when possible."""
    if is_url(source):
        return source
    path = Path(source).expanduser().resolve()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def doc_id_for(source: str) -> str:
    return sha256(source)[:16]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def texts_hash(docs: Iterable[Document]) -> str:
    return sha256("\x00".join(d.page_content for d in docs))


def sanitize_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Vector stores only accept flat scalar metadata: drop None, JSON-encode the rest."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        out[str(key)] = (
            value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str)
        )
    return out


def enrich_chunks(
    chunks: list[Document],
    *,
    doc_id: str,
    content_hash: str,
    extra: Mapping[str, Any] | None = None,
    prefix: str = "",
) -> list[Document]:
    """Assign chunk ids and common metadata; ids are stable for identical content."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for index, chunk in enumerate(chunks):
        chunk_id = sha256(f"{doc_id}|{prefix}{index}|{chunk.page_content}")[:32]
        meta = {
            **chunk.metadata,
            **(extra or {}),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "content_hash": content_hash,
            "ingested_at": now,
        }
        out.append(Document(page_content=chunk.page_content, metadata=sanitize_metadata(meta), id=chunk_id))
    return out


def dedupe(chunks: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out = []
    for chunk in chunks:
        key = sha256(chunk.page_content.strip())
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out
