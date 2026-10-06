"""Pydantic models describing the full RAG configuration.

Every section forbids unknown keys so a typo in YAML, an env var or a
``--set`` override fails loudly instead of being silently ignored.
Secrets never live here: sections reference the *name* of the env var
that holds a credential, and providers read it at build time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- app
class AppConfig(StrictModel):
    name: str = "ragchatbot"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    data_dir: Path = Path("./data")

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.upper() if isinstance(v, str) else v


# ------------------------------------------------------------------- providers
LLMProvider = Literal["google_genai", "openai", "anthropic", "ollama"]
EmbeddingProvider = Literal["google", "openai", "ollama", "huggingface"]


class LLMConfig(StrictModel):
    provider: LLMProvider = "google_genai"
    model: str = "gemini-3.5-flash"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int | None = Field(1024, gt=0)
    timeout: float | None = Field(60, gt=0)
    max_retries: int = Field(2, ge=0)
    structured_output: bool = True
    base_url: str | None = None  # e.g. a remote Ollama host


class EmbeddingCacheConfig(StrictModel):
    enabled: bool = True
    path: Path = Path("./data/emb_cache")


class EmbeddingsConfig(StrictModel):
    provider: EmbeddingProvider = "google"
    model: str = "models/gemini-embedding-001"
    batch_size: int = Field(64, gt=0)
    normalize: bool = True
    base_url: str | None = None
    cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)

    @field_validator("provider", mode="before")
    @classmethod
    def _no_anthropic(cls, v: Any) -> Any:
        if v == "anthropic":
            raise ValueError(
                "Anthropic does not offer an embeddings model; use google, openai, "
                "ollama or huggingface for embeddings (the LLM can still be anthropic)"
            )
        return v


# ------------------------------------------------------------------- ingestion
class CleaningConfig(StrictModel):
    normalize_whitespace: bool = True
    strip_headers_footers: bool = False
    min_chars: int = Field(30, ge=0)


class DedupeConfig(StrictModel):
    enabled: bool = True
    strategy: Literal["hash", "none"] = "hash"


class CSVLoaderConfig(StrictModel):
    content_columns: list[str] | None = None
    metadata_columns: list[str] = Field(default_factory=list)


class JSONLoaderConfig(StrictModel):
    jq_schema: str = "."
    content_key: str | None = None
    json_lines: bool = False


class URLLoaderConfig(StrictModel):
    recursive: bool = False
    max_depth: int = Field(2, ge=1)


class IngestionConfig(StrictModel):
    sources: list[str] = Field(default_factory=lambda: ["./docs"])
    glob: str = "**/*"
    exclude: list[str] = Field(default_factory=list)
    # file extension (without dot) -> loader name, e.g. {"pdf": "pymupdf"}
    loaders: dict[str, str] = Field(default_factory=lambda: {"pdf": "pypdf", "html": "bs4"})
    csv: CSVLoaderConfig = Field(default_factory=CSVLoaderConfig)
    json_: JSONLoaderConfig = Field(default_factory=JSONLoaderConfig, alias="json")
    url: URLLoaderConfig = Field(default_factory=URLLoaderConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    incremental: bool = True

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    @field_validator("loaders")
    @classmethod
    def _normalize_ext(cls, v: dict[str, str]) -> dict[str, str]:
        return {k.lower().lstrip("."): name for k, name in v.items()}


class SplitterConfig(StrictModel):
    type: Literal["recursive", "token", "markdown_header", "semantic", "html_header"] = "recursive"
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(150, ge=0)
    separators: list[str] | None = None
    encoding_name: str = "cl100k_base"  # token splitter
    headers: list[str] = Field(default_factory=lambda: ["#", "##", "###"])  # md/html header splitters
    semantic_breakpoint: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = (
        "percentile"
    )

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> SplitterConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        return self


# ---------------------------------------------------------------- vector store
class ChromaConfig(StrictModel):
    persist_dir: Path = Path("./data/chroma")


class FAISSConfig(StrictModel):
    index_dir: Path = Path("./data/faiss")


class QdrantConfig(StrictModel):
    url: str | None = None
    path: Path | None = Path("./data/qdrant")
    api_key_env: str | None = "QDRANT_API_KEY"


class PGVectorConfig(StrictModel):
    connection_env: str = "PG_CONN"


class VectorStoreConfig(StrictModel):
    type: Literal["chroma", "faiss", "qdrant", "pgvector", "memory"] = "chroma"
    collection: str = Field("ragchatbot", min_length=1)
    distance: Literal["cosine", "l2", "ip"] = "cosine"
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    pgvector: PGVectorConfig = Field(default_factory=PGVectorConfig)

    @model_validator(mode="after")
    def _qdrant_location(self) -> VectorStoreConfig:
        if self.type == "qdrant" and not (self.qdrant.url or self.qdrant.path):
            raise ValueError("qdrant needs either vector_store.qdrant.url or vector_store.qdrant.path")
        return self


# ------------------------------------------------------------------- retrieval
class HybridConfig(StrictModel):
    weights: list[float] = Field(default_factory=lambda: [0.6, 0.4])
    fusion: Literal["weighted", "rrf"] = "rrf"
    rrf_k: int = Field(60, gt=0)

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("hybrid.weights needs exactly two values: [dense, sparse]")
        if any(w < 0 for w in v):
            raise ValueError("hybrid.weights must be non-negative")
        if not math.isclose(sum(v), 1.0, abs_tol=1e-6):
            raise ValueError(f"hybrid.weights must sum to 1, got {sum(v)}")
        return v


class ParentConfig(StrictModel):
    parent_chunk_size: int = Field(2000, gt=0)
    child_chunk_size: int = Field(400, gt=0)

    @model_validator(mode="after")
    def _child_lt_parent(self) -> ParentConfig:
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError("parent.child_chunk_size must be smaller than parent.parent_chunk_size")
        return self


class RetrievalConfig(StrictModel):
    strategy: Literal["dense", "sparse", "hybrid", "parent"] = "hybrid"
    search_type: Literal["similarity", "mmr", "threshold"] = "mmr"
    k: int = Field(6, gt=0)
    fetch_k: int = Field(24, gt=0)
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)
    score_threshold: float | None = Field(None, ge=0.0, le=1.0)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    parent: ParentConfig = Field(default_factory=ParentConfig)
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> RetrievalConfig:
        if self.search_type == "mmr" and self.fetch_k < self.k:
            raise ValueError(f"fetch_k ({self.fetch_k}) must be >= k ({self.k}) for mmr search")
        if self.search_type == "threshold" and self.score_threshold is None:
            raise ValueError("search_type 'threshold' requires retrieval.score_threshold")
        return self


class QueryTransformConfig(StrictModel):
    type: Literal["none", "rewrite", "multi_query", "hyde", "step_back"] = "none"
    num_queries: int = Field(3, ge=1, le=10)
    condense_with_history: bool = True


class RerankerConfig(StrictModel):
    type: Literal["none", "cross_encoder", "cohere", "llm"] = "none"
    model: str = "BAAI/bge-reranker-base"
    top_n: int = Field(4, gt=0)


class CompressionConfig(StrictModel):
    type: Literal["none", "llm_extract", "embeddings_filter", "redundant_filter"] = "none"
    similarity_threshold: float = Field(0.75, ge=0.0, le=1.0)


# ------------------------------------------------------------------ generation
class GenerationConfig(StrictModel):
    prompt_template: str = "default"  # registry name or path to a template file
    max_context_tokens: int = Field(6000, gt=0)
    citation_style: Literal["numeric", "inline_source", "none"] = "numeric"
    answer_when_no_context: Literal["refuse", "general_knowledge"] = "refuse"
    grade_documents: bool = False
    self_check: bool = False
    max_retries: int = Field(1, ge=0, le=3)  # loops for grade/self-check retries


class MemoryConfig(StrictModel):
    type: Literal["none", "buffer", "window", "summary"] = "window"
    window_size: int = Field(6, ge=1)
    checkpointer: Literal["memory", "sqlite"] = "sqlite"
    path: Path = Path("./data/sessions.sqlite")


Metric = Literal["hit_rate", "mrr", "recall", "faithfulness", "answer_relevance", "correctness"]


def _default_metrics() -> list[Metric]:
    return ["hit_rate", "mrr", "faithfulness", "answer_relevance"]


class EvaluationConfig(StrictModel):
    dataset: Path = Path("./tests/fixtures/qa.jsonl")
    metrics: list[Metric] = Field(default_factory=_default_metrics)
    judge_llm: LLMConfig | None = None  # None -> reuse the main llm
    output_dir: Path = Path("./data/eval")


class ObservabilityConfig(StrictModel):
    langsmith: bool = False
    project: str = "ragchatbot"
    log_timings: bool = True


# ----------------------------------------------------------------------- root
class RAGConfig(StrictModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="after")
    def _cross_section(self) -> RAGConfig:
        if self.reranker.type != "none" and self.reranker.top_n > self.retrieval.k:
            raise ValueError(
                f"reranker.top_n ({self.reranker.top_n}) cannot exceed retrieval.k ({self.retrieval.k})"
            )
        return self
