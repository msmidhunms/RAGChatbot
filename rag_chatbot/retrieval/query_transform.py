"""Query transformation before retrieval.

Each transform returns the list of queries to search; results of multiple
queries are fused with reciprocal rank fusion by the retrieval pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.output_parsers import StrOutputParser

from rag_chatbot.config.schema import QueryTransformConfig
from rag_chatbot.core.registry import Registry
from rag_chatbot.generation import prompts
from rag_chatbot.generation.schemas import QueryList
from rag_chatbot.retrieval.base import LLMGetter, history_block

Transform = Callable[[str, str], list[str]]  # (question, history) -> queries
TransformFactory = Callable[[QueryTransformConfig, LLMGetter], Transform]
QUERY_TRANSFORMS: Registry[TransformFactory] = Registry("query transform")


def build_transform(cfg: QueryTransformConfig, get_llm: LLMGetter, name: str | None = None) -> Transform:
    return QUERY_TRANSFORMS.get(name or cfg.type)(cfg, get_llm)


def _text(prompt, get_llm: LLMGetter, **values: str) -> str:  # type: ignore[no-untyped-def]
    return (prompt | get_llm() | StrOutputParser()).invoke(values).strip()


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for q in (q.strip() for q in queries):
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


@QUERY_TRANSFORMS.register("none")
def _none(cfg: QueryTransformConfig, get_llm: LLMGetter) -> Transform:
    return lambda question, history: [question]


@QUERY_TRANSFORMS.register("rewrite")
def _rewrite(cfg: QueryTransformConfig, get_llm: LLMGetter) -> Transform:
    def run(question: str, history: str) -> list[str]:
        rewritten = _text(prompts.REWRITE, get_llm, question=question, history=history_block(history))
        return _dedupe([rewritten or question])

    return run


@QUERY_TRANSFORMS.register("multi_query")
def _multi_query(cfg: QueryTransformConfig, get_llm: LLMGetter) -> Transform:
    def run(question: str, history: str) -> list[str]:
        chain = prompts.MULTI_QUERY | get_llm().with_structured_output(QueryList)
        result = chain.invoke({"question": question, "history": history_block(history), "n": cfg.num_queries})
        return _dedupe([question, *result.queries[: cfg.num_queries]])  # type: ignore[union-attr]

    return run


@QUERY_TRANSFORMS.register("hyde")
def _hyde(cfg: QueryTransformConfig, get_llm: LLMGetter) -> Transform:
    def run(question: str, history: str) -> list[str]:
        passage = _text(prompts.HYDE, get_llm, question=question, history=history_block(history))
        return _dedupe([passage or question])

    return run


@QUERY_TRANSFORMS.register("step_back")
def _step_back(cfg: QueryTransformConfig, get_llm: LLMGetter) -> Transform:
    def run(question: str, history: str) -> list[str]:
        broader = _text(prompts.STEP_BACK, get_llm, question=question, history=history_block(history))
        return _dedupe([question, broader])

    return run
