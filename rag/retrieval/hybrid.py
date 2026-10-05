"""Result fusion (reciprocal rank or weighted scores) and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from rag.config.schema import HybridConfig
from rag.core.types import RetrievedChunk, rerank
from rag.retrieval.base import Filter, Retriever


def fuse(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    weights: Sequence[float] | None = None,
    method: str = "rrf",
    rrf_k: int = 60,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Merge ranked lists into one, deduplicating by chunk id.

    ``rrf``: score = sum(w / (rrf_k + rank)). ``weighted``: each list's scores
    are min-max normalised to [0, 1], then summed with the weights.
    """
    weights = list(weights) if weights is not None else [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    first: dict[str, RetrievedChunk] = {}
    for results, weight in zip(result_lists, weights, strict=True):
        if not results:
            continue
        if method == "weighted":
            lo = min(c.score for c in results)
            hi = max(c.score for c in results)
            span = (hi - lo) or 1.0
        for rank, chunk in enumerate(results, start=1):
            key = chunk.chunk_id
            first.setdefault(key, chunk)
            if method == "weighted":
                contribution = weight * ((chunk.score - lo) / span if hi != lo else 1.0)
            else:
                contribution = weight / (rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
    order = sorted(scores, key=lambda key: -scores[key])[: k or None]
    label = "rrf" if method == "rrf" else "weighted"
    return rerank([replace(first[key], score=scores[key]) for key in order], source=label)


class HybridRetriever:
    """Dense + BM25, each over-fetched, then fused."""

    def __init__(self, dense: Retriever, sparse: Retriever, cfg: HybridConfig, fetch_factor: int = 2) -> None:
        self.dense = dense
        self.sparse = sparse
        self.cfg = cfg
        self.fetch_factor = fetch_factor

    def retrieve(self, query: str, k: int, flt: Filter = None) -> list[RetrievedChunk]:
        n = k * self.fetch_factor
        lists = [self.dense.retrieve(query, n, flt), self.sparse.retrieve(query, n, flt)]
        return fuse(lists, self.cfg.weights, self.cfg.fusion, self.cfg.rrf_k, k)
