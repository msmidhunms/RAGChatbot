"""LangGraph node implementations.

Each node reads the state, does one stage, and returns a partial update.
``timed`` adds the node's latency to ``state["timings"]`` (accumulated
across retries).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from rag_chatbot.config.schema import RAGConfig
from rag_chatbot.core.logging import get_logger
from rag_chatbot.generation import prompts
from rag_chatbot.generation.generator import Generator
from rag_chatbot.generation.schemas import GradeResult, GroundednessResult, RAGAnswer
from rag_chatbot.graph.state import RAGState, dump_chunks, load_chunks
from rag_chatbot.memory.history import format_history, update_summary
from rag_chatbot.retrieval.base import LLMGetter, excerpts
from rag_chatbot.retrieval.factory import RetrievalPipeline

log = get_logger("graph")

Node = Callable[..., dict[str, Any]]


def timed(name: str, fn: Node) -> Node:
    def wrapper(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        start = time.perf_counter()
        update = fn(state, config)
        timings = dict(state.get("timings") or {})
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
        return {**update, "timings": timings}

    wrapper.__name__ = name
    return wrapper


@dataclass
class Components:
    cfg: RAGConfig
    retrieval: RetrievalPipeline
    generator: Generator
    get_llm: LLMGetter


class Nodes:
    def __init__(self, c: Components) -> None:
        self.c = c
        self.cfg = c.cfg

    # ------------------------------------------------------------- helpers
    def _history(self, state: RAGState) -> str:
        past = state.get("messages", [])[:-1]  # last message is the current question
        return format_history(past, self.cfg.memory, state.get("summary", ""))

    def _question(self, state: RAGState) -> str:
        return state.get("standalone_question") or state["question"]

    # --------------------------------------------------------------- nodes
    def condense(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        history = self._history(state)
        question = state["question"]
        if history and self.cfg.query_transform.condense_with_history:
            chain = prompts.CONDENSE | self.c.get_llm() | StrOutputParser()
            question = chain.invoke({"history": history, "question": question}).strip() or question
        return {"standalone_question": question}

    def transform(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        # after failed grading, rewrite the query whatever the configured transform is
        mode = "rewrite" if state.get("retries", 0) > 0 else None
        queries = self.c.retrieval.transform(self._question(state), self._history(state), mode=mode)
        return {"queries": queries}

    def retrieve(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        chunks = self.c.retrieval.search(state.get("queries") or [self._question(state)], state.get("filter"))
        return {"chunks": dump_chunks(chunks)}

    def postprocess(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        chunks = self.c.retrieval.postprocess(self._question(state), load_chunks(state.get("chunks")))
        return {"chunks": dump_chunks(chunks)}

    def grade(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        chunks = load_chunks(state.get("chunks"))
        kept = []
        if chunks:
            chain = prompts.GRADE | self.c.get_llm().with_structured_output(GradeResult)
            result = chain.invoke({"question": self._question(state), "context": excerpts(chunks)})
            relevant = set(result.relevant)  # type: ignore[union-attr]
            kept = [c for i, c in enumerate(chunks, start=1) if i in relevant]
        retry = not kept and state.get("retries", 0) < self.cfg.generation.max_retries
        update: dict[str, Any] = {"chunks": dump_chunks(kept), "rewrite": retry}
        if retry:
            log.info("no relevant chunks; rewriting the query")
            update["retries"] = state.get("retries", 0) + 1
        return update

    def route_after_grade(self, state: RAGState) -> str:
        return "transform" if state.get("rewrite") else "generate"

    def generate(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        stream = bool((config.get("configurable") or {}).get("stream"))
        g = self.c.generator
        fn = g.generate_text if stream else g.generate
        answer = fn(
            self._question(state),
            load_chunks(state.get("chunks")),
            self._history(state),
            state.get("feedback", ""),
        )
        return {"answer": answer.model_dump()}

    def self_check(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        answer = RAGAnswer.model_validate(state["answer"])
        chunks = load_chunks(state.get("chunks"))
        if not chunks:  # refusals / general-knowledge answers are not checked against documents
            return {}
        chain = prompts.SELF_CHECK | self.c.get_llm().with_structured_output(GroundednessResult)
        result: GroundednessResult = chain.invoke(  # type: ignore[assignment]
            {"question": self._question(state), "context": excerpts(chunks, 4000), "answer": answer.answer}
        )
        answer.grounded = result.grounded
        update: dict[str, Any] = {"answer": answer.model_dump()}
        if not result.grounded and state.get("check_retries", 0) < self.cfg.generation.max_retries:
            log.info("answer not grounded (%s); regenerating", result.issues)
            update["check_retries"] = state.get("check_retries", 0) + 1
            update["feedback"] = result.issues or "unsupported claims"
        else:
            update["feedback"] = ""
        return update

    def route_after_check(self, state: RAGState) -> str:
        return "generate" if state.get("feedback") else "finalize"

    def finalize(self, state: RAGState, config: RunnableConfig) -> dict[str, Any]:
        answer = state.get("answer") or {}
        message = AIMessage(content=answer.get("answer", ""))
        messages = [*state.get("messages", []), message]
        summary, upto = update_summary(
            messages,
            self.cfg.memory,
            state.get("summary", ""),
            state.get("summarized_upto", 0),
            self.c.get_llm,
        )
        return {"messages": [message], "summary": summary, "summarized_upto": upto}
