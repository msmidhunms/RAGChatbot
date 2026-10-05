"""Post-retrieval compression: drop irrelevant/redundant chunks or extract relevant sentences."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser

from rag.config.schema import CompressionConfig
from rag.core.registry import Registry
from rag.core.types import RetrievedChunk, rerank
from rag.generation import prompts
from rag.retrieval.base import LLMGetter

Compressor = Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]]
CompressorFactory = Callable[[CompressionConfig, Embeddings, LLMGetter], Compressor]
COMPRESSORS: Registry[CompressorFactory] = Registry("compressor")


def build_compressor(cfg: CompressionConfig, embeddings: Embeddings, get_llm: LLMGetter) -> Compressor:
    return COMPRESSORS.get(cfg.type)(cfg, embeddings, get_llm)


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


@COMPRESSORS.register("none")
def _none(cfg: CompressionConfig, embeddings: Embeddings, get_llm: LLMGetter) -> Compressor:
    return lambda question, chunks: chunks


@COMPRESSORS.register("embeddings_filter")
def _embeddings_filter(cfg: CompressionConfig, embeddings: Embeddings, get_llm: LLMGetter) -> Compressor:
    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        q = _unit(np.asarray(embeddings.embed_query(question)))
        docs = _unit(np.asarray(embeddings.embed_documents([c.text for c in chunks])))
        sims = docs @ q
        return rerank([c for c, s in zip(chunks, sims, strict=True) if s >= cfg.similarity_threshold])

    return run


@COMPRESSORS.register("redundant_filter")
def _redundant_filter(cfg: CompressionConfig, embeddings: Embeddings, get_llm: LLMGetter) -> Compressor:
    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if len(chunks) < 2:
            return chunks
        vecs = _unit(np.asarray(embeddings.embed_documents([c.text for c in chunks])))
        kept: list[int] = []
        for i in range(len(chunks)):  # chunks are best-first, so the better duplicate survives
            if not kept or float((vecs[kept] @ vecs[i]).max()) < cfg.similarity_threshold:
                kept.append(i)
        return rerank([chunks[i] for i in kept])

    return run


@COMPRESSORS.register("llm_extract")
def _llm_extract(cfg: CompressionConfig, embeddings: Embeddings, get_llm: LLMGetter) -> Compressor:
    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        chain = prompts.EXTRACT | get_llm() | StrOutputParser()
        outputs = chain.batch([{"question": question, "text": c.text} for c in chunks])
        kept = []
        for chunk, text in zip(chunks, outputs, strict=True):
            text = text.strip()
            if text and "NO_OUTPUT" not in text:
                doc = Document(page_content=text, metadata=chunk.metadata, id=chunk.document.id)
                kept.append(replace(chunk, document=doc))
        return rerank(kept)

    return run
