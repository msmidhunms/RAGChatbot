import math

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from rag_chatbot.cache import CachedEmbeddings, NormalizedEmbeddings, SQLiteEmbeddingStore
from rag_chatbot.config import load_config
from rag_chatbot.core.exceptions import MissingCredentialsError
from rag_chatbot.providers import build_embeddings, build_llm, embedding_namespace
from rag_chatbot.providers.embeddings import EMBEDDING_PROVIDERS

KEYS = {"GOOGLE_API_KEY": "test", "OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test"}


@pytest.fixture
def api_keys(monkeypatch):
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)


def cfg(*overrides, tmp_path=None):
    extra = [f"embeddings.cache.path={tmp_path}"] if tmp_path else []
    return load_config(overrides=[*overrides, *extra], env={})


# ------------------------------------------------------------------------ llm
@pytest.mark.parametrize(
    "provider, model, cls, checks",
    [
        (
            "google_genai",
            "gemini-3.5-flash",
            "ChatGoogleGenerativeAI",
            {"max_output_tokens": 256, "timeout": 30},
        ),
        ("openai", "gpt-4o-mini", "ChatOpenAI", {"max_tokens": 256, "request_timeout": 30}),
        (
            "anthropic",
            "claude-sonnet-5-5",
            "ChatAnthropic",
            {"max_tokens": 256, "default_request_timeout": 30},
        ),
        ("ollama", "llama3.1", "ChatOllama", {"num_predict": 256}),
    ],
)
def test_build_llm_providers(api_keys, provider, model, cls, checks):
    llm = build_llm(
        cfg(f"llm.provider={provider}", f"llm.model={model}", "llm.max_tokens=256", "llm.timeout=30").llm
    )
    assert type(llm).__name__ == cls
    assert llm.temperature == pytest.approx(0.1)
    for attr, value in checks.items():
        assert getattr(llm, attr) == value


def test_build_llm_ollama_base_url():
    llm = build_llm(cfg("llm.provider=ollama", "llm.model=llama3.1", "llm.base_url=http://gpu:11434").llm)
    assert llm.base_url == "http://gpu:11434"


def test_build_llm_missing_key():
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY.*llm.provider=anthropic"):
        build_llm(cfg("llm.provider=anthropic").llm, env={})


def test_build_llm_ollama_needs_no_key():
    assert build_llm(cfg("llm.provider=ollama", "llm.model=llama3.1").llm, env={}) is not None


# ----------------------------------------------------------------- embeddings
@pytest.mark.parametrize(
    "provider, model, cls",
    [
        ("google", "models/gemini-embedding-001", "GoogleGenerativeAIEmbeddings"),
        ("openai", "text-embedding-3-small", "OpenAIEmbeddings"),
        ("ollama", "nomic-embed-text", "OllamaEmbeddings"),
    ],
)
def test_build_embeddings_providers(api_keys, provider, model, cls):
    emb = build_embeddings(
        cfg(
            f"embeddings.provider={provider}",
            f"embeddings.model={model}",
            "embeddings.cache.enabled=false",
            "embeddings.normalize=false",
        ).embeddings
    )
    assert type(emb).__name__ == cls


def test_build_embeddings_wrapping_order(api_keys, tmp_path):
    emb = build_embeddings(cfg(tmp_path=tmp_path).embeddings)
    assert isinstance(emb, NormalizedEmbeddings)
    assert isinstance(emb.base, CachedEmbeddings)
    assert emb.base.namespace == "google:models/gemini-embedding-001"
    assert (tmp_path / "embeddings.sqlite").exists()


def test_build_embeddings_missing_key():
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        build_embeddings(cfg("embeddings.provider=openai").embeddings, env={})


def test_embedding_namespace_changes_with_model():
    a = cfg("embeddings.model=a").embeddings
    b = cfg("embeddings.model=b").embeddings
    assert embedding_namespace(a) != embedding_namespace(b)


def test_registry_lists_all_schema_providers():
    assert EMBEDDING_PROVIDERS.available() == ["google", "huggingface", "ollama", "openai"]


# ---------------------------------------------------------------------- cache
class CountingEmbeddings(Embeddings):
    def __init__(self, size: int = 8) -> None:
        self.inner = DeterministicFakeEmbedding(size=size)
        self.doc_calls: list[list[str]] = []
        self.query_calls = 0

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return self.inner.embed_query(text)


def test_cache_hits_skip_provider(tmp_path):
    base = CountingEmbeddings()
    emb = CachedEmbeddings(base, SQLiteEmbeddingStore(tmp_path / "c.sqlite"), "ns")
    first = emb.embed_documents(["a", "b"])
    second = emb.embed_documents(["b", "a", "c"])
    assert base.doc_calls == [["a", "b"], ["c"]]
    assert second[0] == first[1] and second[1] == first[0]
    assert second == base.inner.embed_documents(["b", "a", "c"])


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedEmbeddings(CountingEmbeddings(), SQLiteEmbeddingStore(path), "ns").embed_documents(["x"])
    base = CountingEmbeddings()
    CachedEmbeddings(base, SQLiteEmbeddingStore(path), "ns").embed_documents(["x"])
    assert base.doc_calls == []


def test_cache_namespaces_isolated(tmp_path):
    store = SQLiteEmbeddingStore(tmp_path / "c.sqlite")
    CachedEmbeddings(CountingEmbeddings(), store, "model-a").embed_documents(["x"])
    base = CountingEmbeddings()
    CachedEmbeddings(base, store, "model-b").embed_documents(["x"])
    assert base.doc_calls == [["x"]]
    assert len(store) == 2


def test_cache_batches_and_dedupes_misses(tmp_path):
    base = CountingEmbeddings()
    emb = CachedEmbeddings(base, SQLiteEmbeddingStore(tmp_path / "c.sqlite"), "ns", batch_size=2)
    out = emb.embed_documents(["a", "b", "a", "c", "d"])
    assert base.doc_calls == [["a", "b"], ["c", "d"]]
    assert len(out) == 5 and out[0] == out[2]


def test_cache_large_lookup(tmp_path):
    # more keys than SQLite's bound-parameter limit in one call
    emb = CachedEmbeddings(CountingEmbeddings(size=2), SQLiteEmbeddingStore(tmp_path / "c.sqlite"), "ns")
    texts = [f"t{i}" for i in range(1200)]
    assert emb.embed_documents(texts) == emb.embed_documents(texts)


def test_query_cache_optional(tmp_path):
    base = CountingEmbeddings()
    store = SQLiteEmbeddingStore(tmp_path / "c.sqlite")
    CachedEmbeddings(base, store, "ns").embed_query("q")
    CachedEmbeddings(base, store, "ns").embed_query("q")
    assert base.query_calls == 2
    cached = CachedEmbeddings(base, store, "ns", cache_queries=True)
    cached.embed_query("q")
    cached.embed_query("q")
    assert base.query_calls == 3


def test_normalized_embeddings_unit_length():
    emb = NormalizedEmbeddings(DeterministicFakeEmbedding(size=16))
    for vec in [*emb.embed_documents(["a", "b"]), emb.embed_query("c")]:
        assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-9)


def test_normalized_zero_vector():
    class Zero(Embeddings):
        def embed_documents(self, texts):
            return [[0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 0.0]

    assert NormalizedEmbeddings(Zero()).embed_query("x") == [0.0, 0.0]
