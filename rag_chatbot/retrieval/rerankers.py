"""Rerankers: reorder retrieved chunks by a stronger relevance model and keep top_n."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import Any

from rag_chatbot.config.schema import RerankerConfig
from rag_chatbot.core.exceptions import MissingCredentialsError
from rag_chatbot.core.registry import Registry, require
from rag_chatbot.core.types import RetrievedChunk, rerank
from rag_chatbot.generation import prompts
from rag_chatbot.generation.schemas import RelevanceScores
from rag_chatbot.retrieval.base import LLMGetter, excerpts

Reranker = Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]]
RerankerFactory = Callable[[RerankerConfig, LLMGetter], Reranker]
RERANKERS: Registry[RerankerFactory] = Registry("reranker")

DEFAULT_COHERE_MODEL = "rerank-v3.5"


def build_reranker(cfg: RerankerConfig, get_llm: LLMGetter) -> Reranker:
    return RERANKERS.get(cfg.type)(cfg, get_llm)


def _apply(chunks: list[RetrievedChunk], scores: list[float], top_n: int, label: str) -> list[RetrievedChunk]:
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top_n]
    return rerank([replace(chunks[i], score=float(scores[i])) for i in order], source=label)


@RERANKERS.register("none")
def _none(cfg: RerankerConfig, get_llm: LLMGetter) -> Reranker:
    return lambda question, chunks: chunks


@lru_cache(maxsize=2)
def _cross_encoder(model: str) -> Any:
    return require("sentence_transformers", "local").CrossEncoder(model)


@RERANKERS.register("cross_encoder")
def _cross(cfg: RerankerConfig, get_llm: LLMGetter) -> Reranker:
    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        scores = _cross_encoder(cfg.model).predict([(question, c.text) for c in chunks])
        return _apply(chunks, list(scores), cfg.top_n, "rerank:cross_encoder")

    return run


@RERANKERS.register("cohere")
def _cohere(cfg: RerankerConfig, get_llm: LLMGetter) -> Reranker:
    if not os.environ.get("COHERE_API_KEY"):
        raise MissingCredentialsError("COHERE_API_KEY is not set but required by reranker.type=cohere")
    mod = require("langchain_cohere", "rerank")
    # the default reranker.model is a HuggingFace cross-encoder; use Cohere's default instead
    model = DEFAULT_COHERE_MODEL if "/" in cfg.model else cfg.model
    client = mod.CohereRerank(model=model, top_n=cfg.top_n)

    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        results = client.rerank(documents=[c.text for c in chunks], query=question, top_n=cfg.top_n)
        scores = [0.0] * len(chunks)
        for r in results:
            scores[r["index"]] = r["relevance_score"]
        return _apply(chunks, scores, cfg.top_n, "rerank:cohere")

    return run


@RERANKERS.register("llm")
def _llm(cfg: RerankerConfig, get_llm: LLMGetter) -> Reranker:
    def run(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        chain = prompts.RERANK | get_llm().with_structured_output(RelevanceScores)
        result = chain.invoke({"question": question, "context": excerpts(chunks)})
        scores = [0.0] * len(chunks)
        for item in result.scores:  # type: ignore[union-attr]
            if 1 <= item.id <= len(chunks):
                scores[item.id - 1] = item.score
        return _apply(chunks, scores, cfg.top_n, "rerank:llm")

    return run
