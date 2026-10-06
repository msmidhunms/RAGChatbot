"""Embeddings factory: one config section -> a LangChain ``Embeddings``.

The provider model is wrapped as ``Normalized(Cached(provider))`` depending on
``embeddings.normalize`` and ``embeddings.cache.enabled``. The cache stores raw
provider vectors, so toggling normalization never invalidates it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from langchain_core.embeddings import Embeddings

from rag_chatbot.cache import CachedEmbeddings, NormalizedEmbeddings, SQLiteEmbeddingStore
from rag_chatbot.config.loader import EMBEDDING_KEY_ENV, ensure_env
from rag_chatbot.config.schema import EmbeddingsConfig
from rag_chatbot.core.registry import Registry, require

EmbeddingsFactory = Callable[[EmbeddingsConfig], Embeddings]
EMBEDDING_PROVIDERS: Registry[EmbeddingsFactory] = Registry("embeddings provider")

CACHE_FILE = "embeddings.sqlite"


def build_embeddings(cfg: EmbeddingsConfig, *, env: Mapping[str, str] | None = None) -> Embeddings:
    ensure_env(EMBEDDING_KEY_ENV.get(cfg.provider), f"embeddings.provider={cfg.provider}", env)
    embeddings = EMBEDDING_PROVIDERS.get(cfg.provider)(cfg)
    if cfg.cache.enabled:
        embeddings = CachedEmbeddings(
            embeddings,
            SQLiteEmbeddingStore(cfg.cache.path / CACHE_FILE),
            namespace=embedding_namespace(cfg),
            batch_size=cfg.batch_size,
        )
    if cfg.normalize:
        embeddings = NormalizedEmbeddings(embeddings)
    return embeddings


class LazyEmbeddings(Embeddings):
    """Builds the real embeddings on first use.

    Lets commands that never embed (store stats/reset/delete, chat history) run
    without provider credentials; ``ensure_ready`` forces the build early so
    missing keys are reported before work starts.
    """

    def __init__(self, factory: Callable[[], Embeddings]) -> None:
        self._factory = factory
        self._inner: Embeddings | None = None

    def ensure_ready(self) -> Embeddings:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.ensure_ready().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.ensure_ready().embed_query(text)


def embedding_namespace(cfg: EmbeddingsConfig) -> str:
    """Identifies the vector space; also used later to guard vector store collections."""
    return f"{cfg.provider}:{cfg.model}"


@EMBEDDING_PROVIDERS.register("google")
def _google(cfg: EmbeddingsConfig) -> Embeddings:
    mod = require("langchain_google_genai")
    kwargs = {"base_url": cfg.base_url} if cfg.base_url else {}
    return mod.GoogleGenerativeAIEmbeddings(model=cfg.model, **kwargs)


@EMBEDDING_PROVIDERS.register("openai")
def _openai(cfg: EmbeddingsConfig) -> Embeddings:
    mod = require("langchain_openai", "openai")
    kwargs = {"base_url": cfg.base_url} if cfg.base_url else {}
    return mod.OpenAIEmbeddings(model=cfg.model, chunk_size=cfg.batch_size, **kwargs)


@EMBEDDING_PROVIDERS.register("ollama")
def _ollama(cfg: EmbeddingsConfig) -> Embeddings:
    mod = require("langchain_ollama", "local")
    kwargs = {"base_url": cfg.base_url} if cfg.base_url else {}
    return mod.OllamaEmbeddings(model=cfg.model, **kwargs)


@EMBEDDING_PROVIDERS.register("huggingface")
def _huggingface(cfg: EmbeddingsConfig) -> Embeddings:
    mod = require("langchain_huggingface", "local")
    return mod.HuggingFaceEmbeddings(model_name=cfg.model, encode_kwargs={"batch_size": cfg.batch_size})
