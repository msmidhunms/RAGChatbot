"""``RAGPipeline``: the single entry point used by the CLI, evaluation and tests.

Components are built lazily from the config, so commands that only need
part of the system (e.g. ``rag retrieve`` with no query transform) never
build an LLM or ask for its API key. Any component can be injected, which
is how the tests run fully offline.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from rag_chatbot.config import RAGConfig, load_config
from rag_chatbot.core.logging import configure_logging
from rag_chatbot.core.types import IngestReport, RetrievedChunk
from rag_chatbot.generation.generator import ANSWER_TAG, Generator
from rag_chatbot.generation.schemas import RAGAnswer
from rag_chatbot.graph.builder import build_graph
from rag_chatbot.graph.nodes import Components
from rag_chatbot.graph.state import load_chunks
from rag_chatbot.ingestion.manifest import Manifest
from rag_chatbot.ingestion.pipeline import IngestionPipeline
from rag_chatbot.memory.history import build_checkpointer
from rag_chatbot.observability.tracing import TokenUsageCallback, log_timings, setup_tracing
from rag_chatbot.providers import build_embeddings, build_llm, embedding_namespace
from rag_chatbot.retrieval.factory import RetrievalPipeline
from rag_chatbot.stores import SQLiteDocStore, VectorStore, build_store


@dataclass
class QueryResult:
    question: str
    answer: RAGAnswer
    chunks: list[RetrievedChunk]
    standalone_question: str = ""
    queries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "queries": self.queries,
            "answer": self.answer.model_dump(),
            "sources": [
                {"rank": c.rank, "score": round(c.score, 4), "retriever": c.source, **c.metadata}
                for c in self.chunks
            ],
            "timings_ms": {k: round(v * 1000, 1) for k, v in self.timings.items()},
            "usage": self.usage,
            "session_id": self.session_id,
        }


class RAGPipeline:
    def __init__(
        self,
        cfg: RAGConfig,
        *,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        store: VectorStore | None = None,
        docstore: SQLiteDocStore | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.cfg = cfg
        self._llm = llm
        self._embeddings = embeddings
        self._store = store
        self._docstore = docstore
        self._checkpointer = checkpointer
        setup_tracing(cfg.observability)

    @classmethod
    def from_config(
        cls, path: str | Path | None = None, overrides: Iterable[str] = (), **kwargs: Any
    ) -> RAGPipeline:
        cfg = load_config(path, overrides)
        configure_logging(cfg.app.log_level)
        return cls(cfg, **kwargs)

    # ------------------------------------------------------------ components
    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(self.cfg.llm)
        return self._llm

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.cfg.embeddings)
        return self._embeddings

    @property
    def persistent(self) -> bool:
        return self.cfg.vector_store.type != "memory"

    def _data_file(self, prefix: str, suffix: str) -> Path:
        vs = self.cfg.vector_store
        return Path(self.cfg.app.data_dir) / f"{prefix}__{vs.type}__{vs.collection}{suffix}"

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = build_store(
                self.cfg.vector_store,
                self.embeddings,
                namespace=embedding_namespace(self.cfg.embeddings),
                data_dir=Path(self.cfg.app.data_dir),
            )
        return self._store

    @property
    def docstore(self) -> SQLiteDocStore:
        if self._docstore is None:
            path = self._data_file("docstore", ".sqlite") if self.persistent else ":memory:"
            self._docstore = SQLiteDocStore(path)
        return self._docstore

    @cached_property
    def manifest(self) -> Manifest:
        return Manifest(self._data_file("manifest", ".json") if self.persistent else None)

    @cached_property
    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(self.cfg, self.store, self.docstore, self.manifest, self.embeddings)

    @cached_property
    def retrieval(self) -> RetrievalPipeline:
        return RetrievalPipeline(self.cfg, self.store, self.docstore, self.embeddings, lambda: self.llm)

    @cached_property
    def generator(self) -> Generator:
        return Generator(lambda: self.llm, self.cfg)

    @cached_property
    def components(self) -> Components:
        return Components(self.cfg, self.retrieval, self.generator, lambda: self.llm)

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        if self._checkpointer is None:
            self._checkpointer = build_checkpointer(self.cfg.memory)
        return self._checkpointer

    @cached_property
    def chat_graph(self) -> Any:
        return build_graph(self.components, self.checkpointer)

    @cached_property
    def query_graph(self) -> Any:
        return build_graph(self.components, None)

    # --------------------------------------------------------------- ingest
    def ingest(
        self,
        sources: Iterable[str] | None = None,
        *,
        tags: Mapping[str, Any] | None = None,
        reset: bool = False,
        incremental: bool | None = None,
    ) -> IngestReport:
        return self.ingestion.run(sources, tags=tags, reset=reset, incremental=incremental)

    def delete_source(self, source: str) -> int:
        return self.ingestion.delete_source(source)

    def reset_store(self) -> None:
        self.ingestion.reset()

    def stats(self) -> dict[str, Any]:
        sources = self.docstore.sources()
        return {
            "vector_store": self.cfg.vector_store.type,
            "collection": self.cfg.vector_store.collection,
            "embedding": embedding_namespace(self.cfg.embeddings),
            "built_with": self.store.meta().get("embedding"),
            "dimension": self.store.meta().get("dimension"),
            "vectors": self.store.count(),
            "chunks": self.docstore.count(),
            "sources": len(sources),
            "per_source": sources,
        }

    # ------------------------------------------------------------- retrieve
    def retrieve(self, question: str, flt: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        return self.retrieval.run(question, "", flt)

    # ---------------------------------------------------------------- query
    def _input(self, question: str, flt: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "standalone_question": "",
            "filter": dict(flt or {}),
            "queries": [],
            "chunks": [],
            "answer": None,
            "retries": 0,
            "rewrite": False,
            "check_retries": 0,
            "feedback": "",
            "timings": {},
        }

    def _result(
        self, state: Mapping[str, Any], usage: TokenUsageCallback, session_id: str | None
    ) -> QueryResult:
        timings = dict(state.get("timings") or {})
        log_timings(timings, self.cfg.observability.log_timings)
        return QueryResult(
            question=state["question"],
            answer=RAGAnswer.model_validate(state["answer"]),
            chunks=load_chunks(state.get("chunks")),
            standalone_question=state.get("standalone_question", ""),
            queries=list(state.get("queries") or []),
            timings=timings,
            usage=usage.as_dict(),
            session_id=session_id,
        )

    def query(self, question: str, flt: Mapping[str, Any] | None = None) -> QueryResult:
        """One-shot question without conversation memory."""
        usage = TokenUsageCallback()
        state = self.query_graph.invoke(self._input(question, flt), {"callbacks": [usage]})
        return self._result(state, usage, None)

    def chat(self, question: str, session_id: str, flt: Mapping[str, Any] | None = None) -> QueryResult:
        """One conversational turn; history is kept per ``session_id``."""
        usage = TokenUsageCallback()
        config = {"configurable": {"thread_id": session_id}, "callbacks": [usage]}
        state = self.chat_graph.invoke(self._input(question, flt), config)
        return self._result(state, usage, session_id)

    def stream_chat(
        self, question: str, session_id: str | None = None, flt: Mapping[str, Any] | None = None
    ) -> Iterator[str | QueryResult]:
        """Yield answer tokens as they are generated, then the final ``QueryResult``."""
        usage = TokenUsageCallback()
        graph = self.chat_graph if session_id else self.query_graph
        configurable: dict[str, Any] = {"stream": True}
        if session_id:
            configurable["thread_id"] = session_id
        config = {"configurable": configurable, "callbacks": [usage]}
        final: Mapping[str, Any] = {}
        streamed = False
        for mode, payload in graph.stream(
            self._input(question, flt), config, stream_mode=["messages", "values"]
        ):
            if mode == "values":
                final = payload
                continue
            message, meta = payload
            if (
                isinstance(message, AIMessageChunk)
                and meta.get("langgraph_node") == "generate"
                and ANSWER_TAG in (meta.get("tags") or [])
                and message.content
            ):
                streamed = True
                yield message.text
        result = self._result(final, usage, session_id)
        if not streamed:  # e.g. refusal without an LLM call
            yield result.answer.answer
        yield result

    # ------------------------------------------------------------- sessions
    def history(self, session_id: str) -> list[BaseMessage]:
        snapshot = self.chat_graph.get_state({"configurable": {"thread_id": session_id}})
        return list((snapshot.values or {}).get("messages", []))

    def clear_session(self, session_id: str) -> None:
        self.checkpointer.delete_thread(session_id)

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]
