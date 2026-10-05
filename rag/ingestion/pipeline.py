"""Ingestion: sources -> load -> clean -> split -> enrich -> dedupe -> store.

Per source, unchanged content is skipped (incremental mode), changed content
replaces the previous chunks, and failures are recorded without stopping
the run. In ``retrieval.strategy: parent`` mode, large parent chunks go to
the docstore and small child chunks (linked by ``parent_id``) are embedded.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config.schema import RAGConfig
from rag.core.logging import get_logger
from rag.core.types import IngestReport
from rag.ingestion.cleaners import clean_documents, drop_short
from rag.ingestion.loaders import discover_sources, is_url, load_source
from rag.ingestion.manifest import Manifest
from rag.ingestion.metadata import (
    dedupe,
    doc_id_for,
    enrich_chunks,
    file_hash,
    normalize_source,
    texts_hash,
)
from rag.ingestion.splitters import Splitter, build_splitter, recursive
from rag.stores.base import VectorStore
from rag.stores.docstore import PARENT, SQLiteDocStore

log = get_logger("ingestion")

ADD_BATCH = 256


class IngestionPipeline:
    def __init__(
        self,
        cfg: RAGConfig,
        store: VectorStore,
        docstore: SQLiteDocStore,
        manifest: Manifest,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.docstore = docstore
        self.manifest = manifest
        self.parent_mode = cfg.retrieval.strategy == "parent"
        if self.parent_mode:
            p = cfg.retrieval.parent
            self.parent_splitter: Splitter = recursive(p.parent_chunk_size, 0, cfg.splitter.separators)
            child_overlap = min(cfg.splitter.chunk_overlap, p.child_chunk_size // 5)
            self.splitter: Splitter = recursive(p.child_chunk_size, child_overlap, cfg.splitter.separators)
        else:
            self.splitter = build_splitter(cfg.splitter, embeddings)

    # ------------------------------------------------------------------ run
    def run(
        self,
        sources: Iterable[str] | None = None,
        *,
        tags: Mapping[str, Any] | None = None,
        reset: bool = False,
        incremental: bool | None = None,
    ) -> IngestReport:
        start = time.perf_counter()
        ing = self.cfg.ingestion
        incremental = ing.incremental if incremental is None else incremental
        report = IngestReport()
        if reset:
            self.reset()

        paths = discover_sources(sources or ing.sources, ing.glob, ing.exclude, cfg=ing)
        report.files_seen = len(paths)
        for raw in paths:
            source = normalize_source(raw)
            try:
                self._ingest_one(raw, source, tags or {}, incremental, report)
            except Exception as exc:  # one bad file must not stop the run
                log.warning("failed to ingest %s: %s", source, exc)
                report.errors[source] = f"{type(exc).__name__}: {exc}"
        self.manifest.save()
        report.seconds = round(time.perf_counter() - start, 3)
        log.info(
            "ingested %d/%d sources (%d skipped, %d failed), %d chunks in %.1fs",
            report.files_ingested,
            report.files_seen,
            report.files_skipped,
            len(report.errors),
            report.chunks_added,
            report.seconds,
        )
        return report

    def _ingest_one(
        self, raw: str, source: str, tags: Mapping[str, Any], incremental: bool, report: IngestReport
    ) -> None:
        content_hash = None
        if not is_url(raw):
            content_hash = file_hash(raw)  # raises FileNotFoundError for missing paths
            if incremental and self.manifest.is_current(source, content_hash):
                report.files_skipped += 1
                return

        docs = load_source(raw, self.cfg.ingestion)
        if content_hash is None:
            content_hash = texts_hash(docs)
            if incremental and self.manifest.is_current(source, content_hash):
                report.files_skipped += 1
                return
        for doc in docs:
            doc.metadata["source"] = source
        docs = clean_documents(docs, self.cfg.ingestion.cleaning)
        if not docs:
            raise ValueError("no text could be extracted")

        doc_id = doc_id_for(source)
        extra = {
            **tags,
            "splitter": "parent" if self.parent_mode else self.cfg.splitter.type,
            "embedding_model": self.store.namespace,
        }
        parents: list[Document] = []
        if self.parent_mode:
            parents = enrich_chunks(
                self.parent_splitter.split_documents(docs),
                doc_id=doc_id,
                content_hash=content_hash,
                extra=extra,
                prefix="p",
            )
            children: list[Document] = []
            for parent in parents:
                for child in self.splitter.split_documents([parent]):
                    child.metadata["parent_id"] = parent.metadata["chunk_id"]
                    children.append(child)
            chunks = children
        else:
            chunks = self.splitter.split_documents(docs)

        chunks = drop_short(chunks, self.cfg.ingestion.cleaning.min_chars)
        if self.cfg.ingestion.dedupe.enabled and self.cfg.ingestion.dedupe.strategy == "hash":
            chunks = dedupe(chunks)
        if not chunks:
            raise ValueError("no chunks left after cleaning (all shorter than cleaning.min_chars?)")
        chunks = enrich_chunks(chunks, doc_id=doc_id, content_hash=content_hash, extra=extra)

        report.chunks_deleted += self._remove(source, doc_id)
        for start in range(0, len(chunks), ADD_BATCH):
            self.store.add(chunks[start : start + ADD_BATCH])
        self.docstore.put(chunks)
        if parents:
            self.docstore.put(parents, kind=PARENT)
        self.manifest.set(
            source,
            content_hash=content_hash,
            doc_id=doc_id,
            chunk_ids=[c.metadata["chunk_id"] for c in chunks],
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        report.files_ingested += 1
        report.chunks_added += len(chunks)

    # ---------------------------------------------------------- maintenance
    def _remove(self, source: str, doc_id: str) -> int:
        entry = self.manifest.get(source)
        ids = set(entry.get("chunk_ids", [])) if entry else set()
        ids.update(self.docstore.delete_doc(doc_id))
        if ids:
            self.store.delete(sorted(ids))
        return len(ids)

    def delete_source(self, source: str) -> int:
        """Remove a source from the store, docstore and manifest; returns deleted chunk count."""
        source = normalize_source(source)
        deleted = self._remove(source, doc_id_for(source))
        self.manifest.remove(source)
        self.manifest.save()
        return deleted

    def reset(self) -> None:
        self.store.reset()
        self.docstore.reset()
        self.manifest.clear()
        self.manifest.save()
