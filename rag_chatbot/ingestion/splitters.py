"""Chunking strategies, selected by ``splitter.type``.

Every factory returns an object with ``split_documents(docs) -> docs`` that
keeps the input metadata on each chunk.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from rag_chatbot.config.schema import SplitterConfig
from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.core.registry import Registry, require


class Splitter(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]: ...


SplitterFactory = Callable[[SplitterConfig, "Embeddings | None"], Splitter]
SPLITTERS: Registry[SplitterFactory] = Registry("splitter")


def build_splitter(cfg: SplitterConfig, embeddings: Embeddings | None = None) -> Splitter:
    return SPLITTERS.get(cfg.type)(cfg, embeddings)


def recursive(chunk_size: int, chunk_overlap: int, separators: list[str] | None = None) -> Splitter:
    kwargs: dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "add_start_index": True,
    }
    if separators:
        kwargs["separators"] = separators
    return RecursiveCharacterTextSplitter(**kwargs)


@SPLITTERS.register("recursive")
def _recursive(cfg: SplitterConfig, embeddings: Embeddings | None) -> Splitter:
    return recursive(cfg.chunk_size, cfg.chunk_overlap, cfg.separators)


@SPLITTERS.register("token")
def _token(cfg: SplitterConfig, embeddings: Embeddings | None) -> Splitter:
    try:
        return TokenTextSplitter(
            encoding_name=cfg.encoding_name, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
    except Exception as exc:  # tiktoken downloads encodings on first use
        raise ConfigError(f"token splitter needs tiktoken encoding '{cfg.encoding_name}': {exc}") from exc


class HeaderSplitter:
    """Split on markdown headings, then size-limit each section.

    The heading path is stored as ``metadata.section`` ("Intro > Setup").
    HTML documents are loaded with markdown-style headings, so this also
    serves the ``html_header`` strategy.
    """

    def __init__(self, cfg: SplitterConfig) -> None:
        self.levels = [(h, f"h{len(h)}") for h in sorted(cfg.headers, key=len)]
        self.inner = recursive(cfg.chunk_size, cfg.chunk_overlap, cfg.separators)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        md = MarkdownHeaderTextSplitter(headers_to_split_on=self.levels, strip_headers=False)
        sections: list[Document] = []
        for doc in documents:
            for part in md.split_text(doc.page_content):
                headers = [part.metadata[name] for _, name in self.levels if name in part.metadata]
                meta = dict(doc.metadata)
                if headers:
                    meta["section"] = " > ".join(headers)
                sections.append(Document(page_content=part.page_content, metadata=meta))
        return self.inner.split_documents(sections)


@SPLITTERS.register("markdown_header")
@SPLITTERS.register("html_header")
def _header(cfg: SplitterConfig, embeddings: Embeddings | None) -> Splitter:
    return HeaderSplitter(cfg)


@SPLITTERS.register("semantic")
def _semantic(cfg: SplitterConfig, embeddings: Embeddings | None) -> Splitter:
    if embeddings is None:
        raise ConfigError("the semantic splitter needs an embeddings model")
    mod = require("langchain_experimental.text_splitter", "semantic")
    return mod.SemanticChunker(embeddings, breakpoint_threshold_type=cfg.semantic_breakpoint)
