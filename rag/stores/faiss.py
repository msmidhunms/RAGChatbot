"""FAISS backend: exact flat index plus a JSON sidecar for documents.

Implemented directly on ``faiss`` (the LangChain wrapper lives in the
retiring langchain-community package). Vectors get int64 ids through an
``IndexIDMap2`` so chunks can be deleted and replaced. Filters are applied
in Python over an over-fetched candidate set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config.schema import VectorStoreConfig
from rag.core.registry import require
from rag.stores.base import STORES, Hit, VectorStore, from_l2
from rag.stores.filters import matches


class FAISSStore(VectorStore):
    kind = "faiss"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._faiss = require("faiss", "faiss")
        self._dir = Path(self.cfg.faiss.index_dir)
        self._index_path = self._dir / f"{self.collection}.faiss"
        self._docs_path = self._dir / f"{self.collection}.json"
        self._index: Any = None
        self._next_id = 0
        self._id_of: dict[str, int] = {}  # chunk id -> faiss id
        self._docs: dict[int, dict[str, Any]] = {}  # faiss id -> {id, content, metadata}
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self._index_path.is_file() and self._docs_path.is_file():
            self._index = self._faiss.read_index(str(self._index_path))
            state = json.loads(self._docs_path.read_text(encoding="utf-8"))
            self._next_id = state["next_id"]
            self._docs = {int(k): v for k, v in state["docs"].items()}
            self._id_of = {v["id"]: k for k, v in self._docs.items()}

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._faiss.write_index(self._index, str(self._index_path))
        tmp = self._docs_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"next_id": self._next_id, "docs": self._docs}), encoding="utf-8")
        os.replace(tmp, self._docs_path)

    def _prepare(self, vectors: list[list[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        if self.cfg.distance == "cosine":
            self._faiss.normalize_L2(arr)
        return arr

    # -------------------------------------------------------------- backend
    def _upsert(self, docs: list[Document], ids: list[str], vectors: list[list[float]]) -> None:
        arr = self._prepare(vectors)
        if self._index is None:
            flat = self._faiss.IndexFlatL2 if self.cfg.distance == "l2" else self._faiss.IndexFlatIP
            self._index = self._faiss.IndexIDMap2(flat(arr.shape[1]))
        self._remove([i for i in ids if i in self._id_of])
        faiss_ids = np.arange(self._next_id, self._next_id + len(ids), dtype="int64")
        self._next_id += len(ids)
        self._index.add_with_ids(arr, faiss_ids)
        for fid, id_, doc in zip(faiss_ids.tolist(), ids, docs, strict=True):
            self._docs[fid] = {"id": id_, "content": doc.page_content, "metadata": doc.metadata}
            self._id_of[id_] = fid
        self._save()

    def _remove(self, ids: Sequence[str]) -> None:
        fids = [self._id_of.pop(i) for i in ids if i in self._id_of]
        if fids and self._index is not None:
            self._index.remove_ids(np.asarray(fids, dtype="int64"))
        for fid in fids:
            self._docs.pop(fid, None)

    def delete(self, ids: Sequence[str]) -> None:
        self._remove(ids)
        self._save()

    def _score(self, raw: float) -> float:
        if self.cfg.distance == "l2":
            return from_l2(raw, squared=True)  # IndexFlatL2 reports squared L2
        return float(raw)  # inner product; equals cosine for the normalized vectors

    def _query(
        self, vector: list[float], k: int, flt: Mapping[str, Any] | None, with_vectors: bool = False
    ) -> list[Hit]:
        if self._index is None or self._index.ntotal == 0:
            return []
        n = self._index.ntotal if flt else min(k, self._index.ntotal)
        scores, fids = self._index.search(self._prepare([vector]), n)
        hits = []
        for raw, fid in zip(scores[0], fids[0], strict=True):
            if fid < 0 or int(fid) not in self._docs:
                continue
            entry = self._docs[int(fid)]
            if flt and not matches(entry["metadata"], flt):
                continue
            vec = self._index.reconstruct(int(fid)).tolist() if with_vectors else None
            doc = Document(page_content=entry["content"], metadata=entry["metadata"], id=entry["id"])
            hits.append(Hit(doc, self._score(raw), vec))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def _reset(self) -> None:
        self._index = None
        self._next_id = 0
        self._docs.clear()
        self._id_of.clear()
        self._index_path.unlink(missing_ok=True)
        self._docs_path.unlink(missing_ok=True)


@STORES.register("faiss")
def _faiss_store(
    cfg: VectorStoreConfig, embeddings: Embeddings, namespace: str, meta_dir: Path
) -> VectorStore:
    return FAISSStore(cfg, embeddings, namespace=namespace, meta_dir=meta_dir)
