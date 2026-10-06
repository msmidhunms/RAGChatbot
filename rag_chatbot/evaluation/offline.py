"""Offline evaluation: deterministic stand-ins for the embedding model and the LLM.

They need no API key or network, so the retrieval pipeline and the answer
metrics can be regression-tested anywhere (``rag eval run --offline``,
``pytest evals``). Scores measure the retrieval stack and a simple extractive
baseline, not a real model; LLM-judge metrics and conversational follow-ups
need a real LLM and are skipped.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag_chatbot.config import RAGConfig, load_config
from rag_chatbot.generation.generator import NO_ANSWER

_WORD = re.compile(r"[a-z0-9]+")
_EXCERPT = re.compile(r"^\[(\d+)\] \([^\n]*\)\n", re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORD_TEXT = (
    "the a an and or of to in on for with by at from is are was were be been does do did what which who whom "
    "whose when where why how much many can could should would will it its this that these those there their "
    "about as into than then me my i you your we our us any all per"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

OFFLINE_OVERRIDES = [
    "vector_store.type=memory",
    "embeddings.cache.enabled=false",
    "llm.structured_output=false",
    "query_transform.type=none",
    "reranker.type=none",
    "compression.type=none",
    "memory.type=none",
    "generation.grade_documents=false",
    "generation.self_check=false",
]
# metrics and categories that need a real LLM
OFFLINE_UNSUPPORTED_CATEGORIES = ("conversational",)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashed into ``size`` buckets, L2-normalised: texts sharing words are similar."""

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in _WORD.findall(text.lower()):
            if len(word) > 2:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ExtractiveChatModel(BaseChatModel):
    """Answers with the context sentence that best overlaps the question, citing its excerpt.

    It refuses (``NO_ANSWER``) when no sentence covers at least ``min_overlap``
    of the question's content words. Prompts without a context block (e.g. a
    condense prompt) get the question echoed back.
    """

    min_overlap: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    @staticmethod
    def _split_prompt(text: str) -> tuple[str, str]:
        if "Context:\n" not in text or "Question:" not in text:
            return "", text.strip()
        context = text.split("Context:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
        question = text.rsplit("Question:", 1)[1].split("\n\nA previous draft", 1)[0].strip()
        return context, question

    @staticmethod
    def _excerpts(context: str) -> list[tuple[int, str]]:
        parts = _EXCERPT.split(context)
        # parts = [prefix, n1, text1, n2, text2, ...]
        return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]

    def answer(self, prompt: str) -> str:
        context, question = self._split_prompt(prompt)
        if not context:
            return question
        wanted = set(_tokens(question))
        if not wanted:
            return NO_ANSWER
        best: tuple[float, int, str] = (0.0, 0, "")
        for number, text in self._excerpts(context):
            for sentence in _SENTENCE.split(text):
                sentence = sentence.strip().lstrip("#-* ").strip()
                if len(sentence) < 3:
                    continue
                overlap = len(wanted & set(_tokens(sentence))) / len(wanted)
                if overlap > best[0]:
                    best = (overlap, number, sentence)
        overlap, number, sentence = best
        if overlap < self.min_overlap:
            return NO_ANSWER
        return f"{sentence} [{number}]"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages[-1].text if messages else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer(prompt)))])


def offline_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> RAGConfig:
    """Load a config and force the offline-safe settings on top of it."""
    return load_config(path, [*overrides, *OFFLINE_OVERRIDES])


def build_offline_pipeline(path: str | Path | None = None, overrides: Sequence[str] = ()) -> Any:
    from rag_chatbot.pipeline import RAGPipeline

    return RAGPipeline(
        offline_config(path, overrides), llm=ExtractiveChatModel(), embeddings=HashingEmbeddings()
    )
