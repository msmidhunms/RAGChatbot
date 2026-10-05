"""Offline test doubles: keyword embeddings and a scriptable chat model."""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from rag.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"
_WORD = re.compile(r"[a-z0-9]+")


class KeywordEmbeddings(Embeddings):
    """Bag-of-words hashed into ``size`` buckets: texts sharing words are similar."""

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


Responder = Callable[[list[BaseMessage]], str]


class FakeChatModel(BaseChatModel):
    """Chat model that answers from a script.

    ``responses`` are consumed in order (the last one repeats); a callable
    receives the prompt messages. ``structured`` maps a Pydantic schema
    name to either an instance, a list of instances consumed in order, or a
    callable(messages) -> instance; ``with_structured_output`` uses it.
    """

    responses: list[Any] = ["ok"]
    structured: dict[str, Any] = {}
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _next(self, messages: list[BaseMessage]) -> str:
        self.calls.append(messages)
        item = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        return item(messages) if callable(item) else str(item)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._next(messages)
        msg = AIMessage(
            content=text, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._next(messages)
        for i, word in enumerate(text.split(" ")):
            token = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        name = schema.__name__ if isinstance(schema, type) else str(schema)

        def run(prompt: Any) -> BaseModel:
            messages = prompt.to_messages() if hasattr(prompt, "to_messages") else prompt
            self.calls.append(messages)
            if name not in self.structured:
                raise KeyError(f"FakeChatModel has no structured response for {name}")
            item = self.structured[name]
            if isinstance(item, list):
                item = item[0] if len(item) == 1 else item.pop(0)
            return item(messages) if callable(item) else item

        return RunnableLambda(run)


@pytest.fixture
def keyword_embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def fake_llm() -> FakeChatModel:
    return FakeChatModel(responses=["ok"], structured={}, calls=[])


@pytest.fixture
def make_cfg(tmp_path: Path) -> Callable[..., Any]:
    """Config with every path under tmp_path; extra args are --set overrides."""

    def make(*overrides: str) -> Any:
        base = [
            f"app.data_dir={tmp_path / 'data'}",
            f"embeddings.cache.path={tmp_path / 'emb'}",
            f"vector_store.chroma.persist_dir={tmp_path / 'chroma'}",
            f"vector_store.faiss.index_dir={tmp_path / 'faiss'}",
            f"vector_store.qdrant.path={tmp_path / 'qdrant'}",
            f"memory.path={tmp_path / 'sessions.sqlite'}",
            f"evaluation.output_dir={tmp_path / 'eval'}",
            "vector_store.type=memory",
            "embeddings.cache.enabled=false",
        ]
        return load_config(overrides=[*base, *overrides], env={})

    return make


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Copy of the text fixtures in a writable directory."""
    dest = tmp_path / "corpus"
    dest.mkdir()
    for name in ("sample.md", "sample.txt", "sample.csv", "sample.json", "sample.html"):
        shutil.copy(FIXTURES / name, dest / name)
    return dest


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a simple text PDF without binary fixtures (needs only pypdf-compatible syntax)."""
    objects: list[bytes] = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_obj = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        content_num = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_num} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode()
        )
        lines = text.replace("(", "").replace(")", "").split("\n")
        ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
        stream = ops.encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path
