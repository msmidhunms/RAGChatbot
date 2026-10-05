# Plan: Configurable RAG Chatbot

## Context

The repo has no RAG yet. `main.py` (24 lines) loads `GOOGLE_API_KEY` from `.env`, builds `gemini-1.5-flash` with `langchain.chat_models.init_chat_model`, wraps it in `with_structured_output(ChatMessages)` (a Pydantic model with `messages` and `title`), and prints one hardcoded answer. `requirements.txt` lists `langgraph`, `langsmith`, `langchain` and `pydantic`. There are no tests and no package structure.

Goal: build a modular RAG system on the LangChain/LangGraph stack. Every stage is chosen and tuned from one YAML config, with env-var overrides and no code changes: loading, chunking, embedding, vector store, retrieval, reranking, query transformation, generation, memory and evaluation. The CLI is the only interface for now. Agreed scope:
- **Vector stores:** Chroma (default), FAISS, Qdrant, PGVector
- **Document types:** PDF, TXT, MD, DOCX, HTML/URL, CSV, JSON
- **Providers:** Google Gemini (default, matching main.py), OpenAI, Anthropic, Ollama/HuggingFace

What to reuse from `main.py`:
- `init_chat_model(model=..., model_provider=...)`: this becomes the single LLM factory.
- The `with_structured_output(PydanticModel)` pattern: this becomes the answer schema (answer, citations, confidence).
- `load_dotenv()`: moves into the settings loader.

---

## Target layout

```
RAGChatbot/
├── pyproject.toml              # replaces bare requirements.txt; extras per provider/store
├── requirements.txt            # pinned core deps (kept for simple installs)
├── .env.example                # GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, QDRANT_URL, PG_CONN, LANGSMITH_*
├── configs/
│   ├── default.yaml            # Gemini + Chroma + recursive chunking + hybrid retrieval
│   ├── local.yaml              # Ollama + HF embeddings + FAISS (offline)
│   └── advanced.yaml           # multi-query + rerank + compression + qdrant
├── main.py                     # thin shim → rag.cli:app
├── rag/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py           # Pydantic config models (single source of truth)
│   │   └── loader.py           # YAML + env overrides + CLI --set overrides → RAGConfig
│   ├── core/
│   │   ├── registry.py         # generic Registry[T] with @register("name") decorator
│   │   ├── types.py            # RetrievedChunk, RAGAnswer, Citation, IngestReport
│   │   ├── exceptions.py
│   │   └── logging.py          # structlog/stdlib setup, level from config
│   ├── providers/
│   │   ├── llm.py              # build_llm(cfg) via init_chat_model
│   │   └── embeddings.py       # build_embeddings(cfg): google/openai/ollama/hf + cache wrapper
│   ├── ingestion/
│   │   ├── loaders.py          # registry: pdf, txt, md, docx, html, url, csv, json, directory
│   │   ├── cleaners.py         # whitespace/unicode normalisation, header/footer strip, dedupe
│   │   ├── splitters.py        # registry: recursive, token, markdown_header, semantic, html_header
│   │   ├── metadata.py         # source, page, section, doc_id(hash), ingested_at, custom tags
│   │   └── pipeline.py         # IngestionPipeline: load→clean→split→enrich→dedupe→embed→upsert
│   ├── stores/
│   │   ├── base.py             # VectorStoreAdapter protocol (add, delete_by_doc, as_retriever, stats, reset)
│   │   ├── chroma.py  faiss.py  qdrant.py  pgvector.py
│   │   └── docstore.py         # persistent parent-doc / BM25 corpus store (LocalFileStore/SQLite)
│   ├── retrieval/
│   │   ├── dense.py            # similarity / mmr / similarity_score_threshold
│   │   ├── sparse.py           # BM25Retriever built from docstore
│   │   ├── hybrid.py           # EnsembleRetriever with weights or RRF
│   │   ├── parent.py           # ParentDocumentRetriever (small-to-big)
│   │   ├── query_transform.py  # none | multi_query | hyde | step_back | rewrite (uses chat history)
│   │   ├── rerankers.py        # none | cross_encoder (HF) | cohere | llm_listwise
│   │   ├── compression.py      # LLM extractor / embeddings-filter / redundant-filter
│   │   ├── filters.py          # metadata filter translation per store
│   │   └── factory.py          # build_retriever(cfg, store, llm) composes above
│   ├── generation/
│   │   ├── prompts.py          # prompt templates loaded from config/files; system/qa/condense/no-answer
│   │   ├── schemas.py          # RAGAnswer pydantic (answer, citations[], confidence, title) — extends ChatMessages idea
│   │   ├── context.py          # context formatting, token budgeting, citation ids [1]..[n]
│   │   └── generator.py        # structured or streaming generation, fallback to plain text
│   ├── memory/
│   │   └── history.py          # none | buffer | window | summary; LangGraph checkpointer (memory/sqlite)
│   ├── graph/
│   │   ├── state.py            # RAGState TypedDict
│   │   ├── nodes.py            # condense, transform, retrieve, rerank, grade, generate, check
│   │   └── builder.py          # build_graph(cfg) → compiled LangGraph; conditional edges per config
│   ├── evaluation/
│   │   ├── dataset.py          # load QA jsonl (question, ground_truth, expected_sources)
│   │   ├── metrics.py          # hit@k, MRR, recall@k; LLM-judge faithfulness/relevance/correctness
│   │   └── runner.py           # run config(s) over dataset → table/json report; optional LangSmith
│   ├── observability/
│   │   └── tracing.py          # LangSmith toggle, per-node timings, token usage callbacks
│   ├── cache.py                # embedding cache (CacheBackedEmbeddings) + optional LLM response cache
│   └── cli.py                  # Typer app
└── tests/
    ├── conftest.py             # FakeEmbeddings, FakeListChatModel, tmp Chroma/FAISS
    ├── test_config.py  test_loaders.py  test_splitters.py  test_stores.py
    ├── test_retrieval.py  test_graph.py  test_cli.py
    └── fixtures/  (sample.pdf, sample.md, sample.csv, qa.jsonl)
```

---

## Module design

### 1. Configuration (`rag/config/`)
Nested Pydantic v2 models, each with `extra="forbid"` so typos fail fast. Use discriminated unions where each backend needs different params.

```yaml
# configs/default.yaml
app: { name: ragchatbot, log_level: INFO, data_dir: ./data }
llm:
  provider: google_genai        # google_genai | openai | anthropic | ollama
  model: gemini-3.5-flash
  temperature: 0.1
  max_tokens: 1024
  timeout: 60
  max_retries: 2
  structured_output: true
embeddings:
  provider: google              # google | openai | ollama | huggingface
  model: models/gemini-embedding-001
  batch_size: 64
  normalize: true
  cache: { enabled: true, path: ./data/emb_cache }
ingestion:
  sources: [./docs]
  glob: "**/*"
  loaders: { pdf: pypdf, html: bs4 }      # per-extension loader override
  cleaning: { normalize_whitespace: true, strip_headers_footers: false, min_chars: 30 }
  dedupe: { enabled: true, strategy: hash }   # hash | none
  incremental: true                           # skip unchanged files by content hash
splitter:
  type: recursive               # recursive | token | markdown_header | semantic | html_header
  chunk_size: 1000
  chunk_overlap: 150
  separators: null
vector_store:
  type: chroma                  # chroma | faiss | qdrant | pgvector
  collection: ragchatbot
  distance: cosine
  chroma: { persist_dir: ./data/chroma }
  faiss: { index_dir: ./data/faiss }
  qdrant: { url: null, path: ./data/qdrant, api_key_env: QDRANT_API_KEY }
  pgvector: { connection_env: PG_CONN }
retrieval:
  strategy: hybrid              # dense | sparse | hybrid | parent
  search_type: mmr              # similarity | mmr | threshold
  k: 6
  fetch_k: 24
  mmr_lambda: 0.5
  score_threshold: null
  hybrid: { weights: [0.6, 0.4], fusion: rrf, rrf_k: 60 }
  parent: { parent_chunk_size: 2000, child_chunk_size: 400 }
  filters: {}                   # default metadata filter
query_transform:
  type: none                    # none | rewrite | multi_query | hyde | step_back
  num_queries: 3
  condense_with_history: true
reranker:
  type: none                    # none | cross_encoder | cohere | llm
  model: BAAI/bge-reranker-base
  top_n: 4
compression:
  type: none                    # none | llm_extract | embeddings_filter | redundant_filter
  similarity_threshold: 0.75
generation:
  prompt_template: default      # name in prompts registry or path to file
  max_context_tokens: 6000
  citation_style: numeric       # numeric | inline_source | none
  answer_when_no_context: refuse  # refuse | general_knowledge
  grade_documents: false        # CRAG-style relevance grading node
  self_check: false             # hallucination/groundedness check + 1 retry
memory:
  type: window                  # none | buffer | window | summary
  window_size: 6
  checkpointer: sqlite          # memory | sqlite
  path: ./data/sessions.sqlite
evaluation:
  dataset: ./tests/fixtures/qa.jsonl
  metrics: [hit_rate, mrr, faithfulness, answer_relevance]
  judge_llm: null               # defaults to llm
observability: { langsmith: false, project: ragchatbot, log_timings: true }
```

`loader.py` (`load_config(path, overrides: list[str]) -> RAGConfig`) applies config in this order:
1. Built-in defaults
2. The YAML file
3. Env vars named `RAG__SECTION__KEY` (via `pydantic-settings`, `env_nested_delimiter="__"`)
4. CLI `--set retrieval.k=10`

It also calls `load_dotenv()`. Validators check that:
- `chunk_overlap` is less than `chunk_size`
- the hybrid weights sum to 1
- the API-key env var for the chosen provider is present (clear error message if not)
- the `anthropic` provider is never used for embeddings (Anthropic has no embedding model)

### 2. Registry / plugin system (`rag/core/registry.py`)
`Registry[T]` holds a `name -> factory(cfg, **deps)` mapping, with `@LOADERS.register("pdf")` decorators, `get(name)` and `available()`. There is one registry each for loaders, splitters, embeddings, stores, query transforms, rerankers, compressors and prompts. Imports of heavy or optional dependencies happen inside the factory functions. A missing extra raises `MissingDependencyError("pip install ragchatbot[qdrant]")`. Adding a backend means writing one decorated function, with no edits to core code.

### 3. Providers (`rag/providers/`)
- `build_llm(llm_cfg)`: `init_chat_model(model, model_provider, temperature, max_tokens, timeout, max_retries)`. This covers google_genai, openai, anthropic and ollama, and generalises `main.py`.
- `build_embeddings(emb_cfg)`: maps each provider to its class:
  - `GoogleGenerativeAIEmbeddings`
  - `OpenAIEmbeddings`
  - `OllamaEmbeddings`
  - `HuggingFaceEmbeddings` (sentence-transformers, with `normalize_embeddings`)

  It is wrapped in `CacheBackedEmbeddings` (`LocalFileStore`) when the embedding cache is enabled, namespaced by model so a model switch never reuses stale vectors.
- The embedding model and dimension are written into the collection metadata. A query against a collection built with a different embedding model raises a clear error. Mixing models in one collection is the most common RAG configuration bug.

### 4. Ingestion (`rag/ingestion/`)
- **Loaders**, by extension, each overridable in config:
  - PDF: `PyPDFLoader`, or `PyMuPDFLoader` when `pymupdf` is chosen
  - TXT: `TextLoader` (with encoding autodetect)
  - MD: `UnstructuredMarkdownLoader`, or a plain-text loader plus the markdown header splitter
  - DOCX: `Docx2txtLoader`
  - HTML: `BSHTMLLoader`
  - URL: `WebBaseLoader`, or `RecursiveUrlLoader` with a depth limit
  - CSV: `CSVLoader` (one row per document; `content_columns` and `metadata_columns` configurable)
  - JSON: `JSONLoader` (`jq_schema` and `content_key` from config)
  - `DirectoryLoader`-style walking honours `glob` and the exclude patterns
- **Cleaners:** composable functions `Document -> Document | None` (drop documents shorter than `min_chars`).
- **Splitters:**
  - `RecursiveCharacterTextSplitter`
  - `TokenTextSplitter` (tiktoken)
  - `MarkdownHeaderTextSplitter` followed by recursive splitting, keeping the header path in metadata
  - `HTMLHeaderTextSplitter`
  - `SemanticChunker` (langchain_experimental, uses the configured embeddings)
- **Metadata:**
  - `doc_id = sha256(source)`
  - `chunk_id = sha256(doc_id + index + content)`
  - `content_hash`, `source`, `page`, `section`, `ingested_at`, `splitter`, `embedding_model`, plus user `--tag k=v`
- **Pipeline:** `IngestionPipeline(cfg).run(sources) -> IngestReport` (files, chunks, skipped, errors, duration).
  - Incremental mode keeps a manifest (`data/manifest.json`: source → content_hash). It skips unchanged files and calls `store.delete_by_doc(doc_id)` before re-upserting changed ones.
  - Embedding and upserting run in `batch_size` batches.
  - Chunks are also written to the docstore so BM25 and parent retrieval can use them.

### 5. Vector stores (`rag/stores/`)
The `VectorStoreAdapter` protocol has these methods:
- `add(chunks, ids)`
- `delete_by_doc(doc_id)`
- `as_langchain()`
- `similarity_search_with_score(query, k, filter)`
- `stats()`
- `reset()`

| Store | Implementation | Persistence |
|---|---|---|
| chroma | `langchain_chroma.Chroma` with `collection_metadata={"hnsw:space": distance}` | `persist_dir` |
| faiss | `FAISS`, `save_local` / `load_local`; delete via the docstore id map | `index_dir` |
| qdrant | `QdrantVectorStore`; URL or local path; payload filters | server or local |
| pgvector | `langchain_postgres.PGVector` with `use_jsonb=True` | Postgres |

`filters.py` turns a store-agnostic filter dict such as `{"source": "a.pdf", "page": {"$gte": 3}}` into each backend's filter format.

### 6. Retrieval (`rag/retrieval/`)
`build_retriever(cfg, store, docstore, llm, embeddings)` builds the retriever in this order:
1. **Base retriever** (`retrieval.strategy`):
   - dense: `store.as_retriever(search_type, k, fetch_k, lambda_mult, score_threshold, filter)`
   - sparse: `BM25Retriever.from_documents(docstore chunks)`, cached and rebuilt when the manifest changes
   - hybrid: `EnsembleRetriever([dense, bm25], weights)`, or a custom RRF fusion when `fusion: rrf`
   - parent: `ParentDocumentRetriever` (child chunks in the vector store, parents in the docstore)
2. **Query transform** (`query_transform.type`):
   - rewrite: one LLM rewrite of the question
   - multi_query: `MultiQueryRetriever` with a custom prompt and `num_queries`, results deduped
   - hyde: generate a hypothetical answer, then embed that answer and search with it
   - step_back: retrieve for both the original and a more general question, then merge
3. **Compression** (`compression.type`):
   - `ContextualCompressionRetriever` with `LLMChainExtractor`, `EmbeddingsFilter` or `EmbeddingsRedundantFilter` (`DocumentCompressorPipeline`)
4. **Reranker** (`reranker.type`):
   - cross_encoder: `CrossEncoderReranker` with `HuggingFaceCrossEncoder`
   - cohere: `CohereRerank`
   - llm: an LLM scores relevance from 0 to 10 using structured output
   - Keeps `top_n`.

Output is `list[RetrievedChunk]` (doc, score, rank, retriever_source) so the CLI can show why each chunk was picked.

### 7. Generation (`rag/generation/`)
- **Prompts:**
  - Named templates in a registry: `default`, `concise`, `detailed`, `strict_citations`
  - A `path/to/template.txt` override is also accepted
  - Separate `condense_question`, `hyde`, `multi_query`, `grader` and `self_check` prompts, all overridable in config
- **Context builder:**
  - Numbers chunks `[1]..[n]` with source and page headers
  - Trims to `max_context_tokens` (tiktoken estimate), dropping the lowest-ranked chunks first
- **Answer schema:** `RAGAnswer(BaseModel)` with `answer: str`, `citations: list[Citation(id, source, page, quote)]`, `confidence: Literal[low, medium, high]` and `title: str`. The `title` field carries over the `ChatMessages` idea from `main.py`.
- **Generator modes:**
  - `structured_output: true` uses `llm.with_structured_output(RAGAnswer)`
  - otherwise it uses a plain `ChatPromptTemplate | llm | StrOutputParser` with streaming (`.stream`) for chat
  - If no chunks pass the threshold and `answer_when_no_context: refuse`, it returns a fixed "not found in the documents" answer without calling the LLM.

### 8. Conversation memory (`rag/memory/`)
- LangGraph checkpointer: `MemorySaver`, or `SqliteSaver` at `memory.path`, keyed by `thread_id` (the CLI `--session` flag).
- Strategies:
  - buffer: full history
  - window: the last N turns, via `trim_messages`
  - summary: an LLM summarises older turns into a running summary
- With `condense_with_history`, follow-up questions are rewritten into standalone questions before retrieval.

### 9. Orchestration with LangGraph (`rag/graph/`)
`RAGState` holds `messages`, `question`, `standalone_question`, `queries`, `documents`, `answer`, `retries` and `timings`.

Nodes are added to the graph only when enabled in config:
```
START → condense (if memory & history) → transform_query (if != none)
      → retrieve → rerank/compress (if enabled)
      → grade_documents (if generation.grade_documents) ──irrelevant & retries<1──→ transform_query (rewrite)
      → generate → self_check (if enabled) ──not grounded & retries<1──→ generate
      → END
```
`build_graph(cfg)` compiles the graph with the checkpointer. `RAGPipeline` (facade) exposes:
- `ingest()`
- `query(q, filters)`
- `chat(q, session_id)`
- `stream(...)`

The CLI and tests depend only on the facade.

### 10. Evaluation (`rag/evaluation/`)
- Dataset: JSONL records with `question`, `ground_truth`, `expected_sources[]`.
- Retrieval metrics (no LLM needed): hit_rate@k, MRR, recall@k, computed by matching on `source` or `doc_id`.
- Generation metrics: LLM-as-judge with structured output for faithfulness (claims supported by context), answer relevance, and correctness against the ground truth. An optional `ragas` integration ships as an extra.
- `rag eval --config a.yaml --config b.yaml` prints a comparison table (rich) and writes `data/eval/<timestamp>.json`, which is how to tune chunk size, k and the reranker.

### 11. Observability & caching
- With `observability.langsmith: true`, the app sets `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT`; langsmith is already a dependency.
- A callback handler records per-node latency and token usage, shown with the `--verbose` flag.
- Embedding cache (above). Optional LLM cache via `set_llm_cache(SQLiteCache)`.

### 12. CLI (`rag/cli.py`, Typer + rich)
All commands accept `--config/-c` and repeatable `--set key=value`.
- `rag ingest PATH... [--tag k=v] [--reset] [--no-incremental]`: prints an `IngestReport` table.
- `rag query "question" [--filter source=x.pdf] [--show-sources] [--json]`: one-shot answer with citations.
- `rag chat [--session ID]`: interactive REPL with streaming. Slash commands: `/sources`, `/reset`, `/config`, `/exit`.
- `rag retrieve "question"`: retrieval only, prints ranked chunks and scores for debugging.
- `rag eval [--config ...]*`: runs the evaluation.
- `rag store stats|reset|delete --source X`
- `rag config show|validate`: prints the merged config with secrets masked.

`main.py` becomes `from rag.cli import app; app()`. A `[project.scripts] rag = "rag.cli:app"` entry is also added.

### 13. Dependencies (`pyproject.toml`)
- **Core:**
  - langchain, langchain-core, langchain-community, langchain-text-splitters, langgraph, langgraph-checkpoint-sqlite, langsmith
  - pydantic, pydantic-settings, python-dotenv, pyyaml, typer, rich, tiktoken, rank_bm25
  - langchain-google-genai, langchain-chroma, pypdf
- **Extras:**
  - `openai` (langchain-openai)
  - `anthropic` (langchain-anthropic)
  - `local` (langchain-ollama, langchain-huggingface, sentence-transformers)
  - `faiss` (faiss-cpu)
  - `qdrant` (langchain-qdrant, qdrant-client)
  - `pgvector` (langchain-postgres, psycopg[binary])
  - `docs` (docx2txt, beautifulsoup4, jq, unstructured[md])
  - `semantic` (langchain-experimental)
  - `rerank` (langchain-cohere)
  - `eval` (ragas)
  - `dev` (pytest, ruff, mypy)

`requirements.txt` keeps the core set for a plain `pip install -r`.

---

## Implementation order (each step is a separate commit on `claude/configurable-rag-plan-voik2a`)
1. Scaffolding: pyproject, package skeleton, `.env.example`, config schema and loader, registry, logging, tests for config.
2. Providers (LLM and embeddings, with cache) and the `main.py` shim.
3. Ingestion: loaders, cleaners, splitters, metadata, incremental pipeline, docstore.
4. Vector stores: Chroma and FAISS first, then Qdrant and PGVector, with filter translation.
5. Retrieval: dense, BM25, hybrid/RRF, parent, query transforms, compression, rerankers.
6. Generation: prompts, context builder, `RAGAnswer` structured output, streaming.
7. Memory and the LangGraph graph, plus the `RAGPipeline` facade.
8. CLI commands.
9. Evaluation runner and metrics, observability.
10. README section, sample configs, and fixtures.

## Verification
- `pytest` runs offline using `FakeEmbeddings(size=...)` and `FakeListChatModel` / `GenericFakeChatModel`:
  - config precedence and validation errors
  - each loader against the fixtures
  - each splitter's chunk sizes and overlap
  - Chroma and FAISS round-trips (add, search, delete_by_doc, re-ingest skips unchanged files)
  - hybrid RRF ordering
  - graph paths with grading and self-check toggled on and off
  - CLI via `typer.testing.CliRunner`
- Qdrant tests use local mode (`path=`); PGVector tests are skipped unless `PG_CONN` is set.
- `ruff check` and `mypy rag`.
- Manual end to end, with `GOOGLE_API_KEY`:
  1. `rag ingest tests/fixtures`
  2. `rag query "..." --show-sources`
  3. `rag chat --session t1` (ask a follow-up to confirm the question is condensed)
  4. `rag eval -c configs/default.yaml -c configs/advanced.yaml` (compare metrics)
  5. `rag ingest` again (confirm incremental mode skips all files)

---

## Addendum: execution plan for steps 3–10 (after steps 1–2 landed)

### What steps 1–2 changed
- **Already in place:**
  - `rag/config` (schema and `load_config`, with `ensure_env` and `missing_env_vars`)
  - `rag/core` (`Registry`, `require`, exceptions, logging)
  - `rag/providers` (`build_llm`, `build_embeddings`, `embedding_namespace`)
  - `rag/cache.py` (`CachedEmbeddings`, `NormalizedEmbeddings`)
  - `rag/cli.py` (`config show|validate`, `llm`)
- **Library constraints:** the installed stack is LangChain 1.x. `MultiQueryRetriever`, `EnsembleRetriever`, `ContextualCompressionRetriever` and `ParentDocumentRetriever` now exist only in the legacy `langchain_classic` package. `BM25Retriever`, the `FAISS` store and most loaders exist only in `langchain_community`, which is being retired.
- **Decision:** build those pieces in-repo on top of `langchain_core` (`Document`, `Embeddings`, prompts, runnables) and maintained libraries: pypdf, docx2txt, bs4, rank_bm25, faiss, langchain-chroma, langchain-qdrant, langchain-postgres, langchain-text-splitters and langgraph. Each piece is small, and our own code gives full control over scores, filters and metadata.
- **Dropped:** the `ragas` integration (it is heavy; our own metrics cover the plan). The `langchain-community` dependency is removed from `pyproject.toml`.

### Shared types (`rag/core/types.py`)
- `RetrievedChunk(document, score, rank, source)`, where `source` names the retriever that produced the chunk (e.g. "dense", "bm25", "rrf").
- `IngestReport(files_seen, files_ingested, files_skipped, chunks, errors, seconds)`.

### Step 3: ingestion (`rag/ingestion/`)
- **`loaders.py`**: `LOADERS: Registry[(path_or_url, IngestionConfig) -> list[Document]]`. Each source is resolved to a loader by extension, with `ingestion.loaders` overrides taking priority.
  - `pypdf`: one Document per page, `metadata.page` 1-based
  - `pymupdf`: optional, extra `docs`
  - `text`: txt/md/rst/log; utf-8, with a latin-1 fallback
  - `docx`: docx2txt, extra `docs`
  - `bs4`: html/htm; strips script and style tags and keeps the `<title>`
  - `url`: `urllib` fetch plus bs4; follows same-host links up to `max_depth` when `recursive` is set
  - `csv`: stdlib csv, one Document per row, honouring `content_columns` and `metadata_columns`
  - `json` / `jsonl`: stdlib; `jq_schema` supports a simple dotted path with `[]` (e.g. `.items[]`) plus `content_key`, so the `jq` library is not needed
  - `discover_sources(sources, glob, exclude)` expands directories and passes URLs through unchanged.
- **`cleaners.py`**:
  - `normalize_whitespace` (NFKC, collapse runs of spaces and blank lines)
  - `strip_headers_footers`: removes lines repeated on 50% or more of a file's pages
  - `min_chars` filter
  - `build_cleaners(cfg) -> list[Callable]`
- **`splitters.py`**: `SPLITTERS` registry over `langchain_text_splitters`:
  - recursive: `RecursiveCharacterTextSplitter`
  - token: `TokenTextSplitter`, using `encoding_name`
  - markdown_header: `MarkdownHeaderTextSplitter`, then recursive splitting; the header path goes into `metadata.section`
  - html_header: `HTMLHeaderTextSplitter`, then recursive splitting
  - semantic: `langchain_experimental.SemanticChunker`, extra `semantic`, needs embeddings
  - The resulting interface is `split(docs) -> list[Document]`.
- **`metadata.py`**:
  - `doc_id = sha256(source)[:16]`
  - `chunk_id = sha256(doc_id|index|content)[:32]`
  - `content_hash` (of the whole file), plus `source`, `page`, `section`, `chunk_index`, `ingested_at`, `splitter`, `embedding_model`, and user tags
- **`manifest.py`**: JSON file at `data_dir/manifest.json`, `{source: {content_hash, doc_id, chunk_ids, ingested_at}}`.
- **`pipeline.py`**: `IngestionPipeline(cfg, store, docstore).run(sources=None, tags=None, reset=False) -> IngestReport`. For each file:
  1. Compute its hash; skip it if unchanged and incremental mode is on.
  2. Load, clean, split and enrich it.
  3. Dedupe chunks by chunk content hash.
  4. Delete the old chunks for this doc from the store and docstore.
  5. `store.add(chunks)` in batches, then `docstore.put`, then update the manifest.
  - A file that fails is recorded in `errors`; the run continues.
  - Sources removed from disk are not deleted automatically; `rag store delete` handles that.

### Step 4: vector stores (`rag/stores/`)
- **`base.py`**: `VectorStore` ABC with:
  - `add(chunks: list[Document])` (ids taken from `metadata.chunk_id`)
  - `delete(ids)`
  - `search(query_vec_or_text, k, filter) -> list[(Document, score)]`, where higher scores are always better (normalised per backend)
  - `mmr_search(query, k, fetch_k, lambda_mult, filter)` (MMR implemented once in base over fetched candidates plus their vectors)
  - `count()`, `reset()`, `get_meta()/set_meta()`
- **Embedding guard:** `set_meta()` stores `embedding_namespace` and the vector dimension. `check_compatible(namespace)` raises `ConfigError` on a mismatch ("collection built with google:X, config uses openai:Y; run `rag store reset` or change collection").
- **`STORES` registry**, built with `(cfg, embeddings)`:
  - `chroma.py`: `langchain_chroma.Chroma` (persist_dir, `collection_metadata={"hnsw:space": distance}`)
  - `faiss.py`: own adapter. `faiss.IndexIDMap2(IndexFlatIP|IndexFlatL2)`, int64 ids mapped to chunk ids; documents kept in a JSON sidecar; `remove_ids` handles deletes; `save()` after each write. Extra `faiss`.
  - `qdrant.py`: `langchain_qdrant.QdrantVectorStore` with `qdrant_client.QdrantClient(url|path)`; the collection is created with the embedding dimension. Extra `qdrant`.
  - `pgvector.py`: `langchain_postgres.PGVector(connection=env[connection_env], use_jsonb=True)`. Extra `pgvector`.
- **`filters.py`**: a store-agnostic filter dict `{field: value | {"$eq","$ne","$in","$gt","$gte","$lt","$lte": v}}` is translated per backend:
  - Chroma: native `$and` syntax
  - Qdrant: `models.Filter`
  - PGVector: native jsonb operators
  - FAISS and the docstore: a Python predicate (`matches(metadata, filter)`)
- **`docstore.py`**: `SQLiteDocStore` (`data_dir/docstore.sqlite`) holds chunk_id → (content, metadata JSON, doc_id), plus parents for parent retrieval. It supports `put`, `get_many`, `delete_doc`, `all_chunks(filter)` and `count`. It is the BM25 corpus and the source of truth for `rag store stats`.

### Step 5: retrieval (`rag/retrieval/`)
- **`base.py`**: the `Retriever` protocol is `retrieve(query: str, k: int, filter: dict|None) -> list[RetrievedChunk]`.
- **`dense.py`**: similarity, mmr, or threshold (drop results below `score_threshold`).
- **`sparse.py`**: `rank_bm25.BM25Okapi` over the docstore chunks, using simple lowercase word tokenisation. The index is rebuilt lazily when the docstore's change counter moves. Filters are applied via `matches`.
- **`hybrid.py`**: `fuse(lists, weights, method, rrf_k)`.
  - `rrf`: score = Σ wᵢ / (rrf_k + rankᵢ)
  - `weighted`: min-max normalise each list's scores, then take the weighted sum
  - Results are deduplicated by chunk_id.
- **`parent.py`**: ingestion stores child chunks (child_chunk_size) in the vector store and parent chunks (parent_chunk_size) in the docstore, linked by `metadata.parent_id`. Retrieval searches children, then returns unique parents ranked by their best child.
- **`query_transform.py`**: `QUERY_TRANSFORMS` registry, `(llm, cfg) -> (question, history) -> list[str]`:
  - `none`: [q]
  - `rewrite`: [rewritten]
  - `multi_query`: [q, *n variants] (structured `list[str]` output)
  - `hyde`: [hypothetical answer]; it is embedded and searched like any other query
  - `step_back`: [q, broader question]
  - Results from multiple queries are fused with RRF.
- **`rerankers.py`**: `RERANKERS` registry:
  - `none`: truncate to top_n
  - `cross_encoder`: `sentence_transformers.CrossEncoder`, extra `local`
  - `cohere`: `langchain_cohere.CohereRerank`, extra `rerank`
  - `llm`: structured relevance scores 0–10 for every chunk in one call
- **`compression.py`**: `COMPRESSORS` registry:
  - `embeddings_filter`: drop chunks whose cosine similarity to the query is below the threshold
  - `redundant_filter`: drop near-duplicate chunks (pairwise cosine above the threshold)
  - `llm_extract`: the LLM keeps only the relevant sentences; chunks that end up empty are dropped
- **`factory.py`**: `build_retriever(cfg, store, docstore, embeddings, llm) -> RetrievalPipeline`, where `RetrievalPipeline.run(question, history, filter) -> list[RetrievedChunk]` runs transform → per-query base retrieval → fuse → compress → rerank. The pipeline's stages are also exposed individually for the graph nodes.

### Step 6: generation (`rag/generation/`)
- **`prompts.py`**: `PROMPTS` registry of `ChatPromptTemplate`s:
  - answer templates: default, concise, detailed, strict_citations
  - helper prompts: condense, rewrite, multi_query, hyde, step_back, grade, self_check, extract, rerank, summary
  - `load_prompt(name_or_path)` also accepts a .txt file containing `{context}` and `{question}`.
- **`schemas.py`**: `Citation(id:int, source, page|None, quote)`, `RAGAnswer(title, answer, citations, confidence)`, plus `GradeResult` and `GroundednessResult`.
- **`context.py`**: `format_context(chunks, style, max_tokens)` numbers chunks [1..n] with source and page headers and drops the lowest-ranked chunks to fit the tiktoken (cl100k) budget. It returns `(text, used_chunks)`.
- **`generator.py`**: `Generator(llm, cfg).generate(question, chunks, history) -> RAGAnswer`. Behaviour:
  - Structured output when enabled. Otherwise a plain-text answer is wrapped in `RAGAnswer` with citations parsed from `[n]` markers.
  - `stream(...)` yields text tokens.
  - No chunks and `answer_when_no_context=refuse` gives a fixed `NO_ANSWER` reply without calling the LLM. With `general_knowledge`, the LLM answers with a "not from documents" note.
  - Citation ids are validated against the chunks actually used, and source and page are filled in from chunk metadata.

### Step 7: memory, graph and pipeline facade
- **`memory/history.py`**: `select_history(messages, cfg, llm)`:
  - none: []
  - buffer: all messages
  - window: the last `window_size` turns, via `trim_messages`
  - summary: a running summary in state, updated when history exceeds the window
  
  `build_checkpointer(cfg)` returns a `MemorySaver`, or `SqliteSaver` (from `langgraph-checkpoint-sqlite`, core dependency) on a `sqlite3` connection.
- **`graph/state.py`**: `RAGState` TypedDict with `messages` (`add_messages`), `question`, `standalone_question`, `queries`, `chunks`, `answer`, `summary`, `retries`, `timings` and `filter`.
- **`graph/nodes.py`**: node factories that close over the components:
  - `condense` (only with history and `condense_with_history`)
  - `transform`, `retrieve`, `postprocess` (compress and rerank)
  - `grade` (structured yes/no per chunk; drops irrelevant chunks; if none survive and retries remain, a rewrite loop goes back to `transform`)
  - `generate`
  - `self_check` (not grounded and retries remain → back to `generate` with feedback)
  - `finalize` (appends the AI message and updates the summary)
  
  Every node records its latency in `timings`.
- **`graph/builder.py`**: `build_graph(components, cfg, checkpointer)` wires in only the enabled nodes, using conditional edges.
- **`rag/pipeline.py`**: `RAGPipeline.from_config(cfg)` lazily builds the embeddings, store, docstore, LLM, retriever, generator and graph. Its methods:
  - `ingest(sources, tags, reset)`
  - `retrieve(q, filter)`
  - `query(q, filter) -> QueryResult(answer, chunks, timings)`, run without memory under a fresh thread id
  - `chat(q, session_id, filter)`
  - `stream_chat(...)`
  - `stats()`, `reset_store()`, `delete_source(src)`
  
  The components can also be injected directly (`RAGPipeline(cfg, llm=..., embeddings=...)`) so tests can use fake models.

### Step 8: CLI (`rag/cli.py`, extend)
- **Commands:**
  - `ingest [PATHS]... --tag k=v --reset --no-incremental`: Rich table of the `IngestReport`
  - `query Q --filter k=v --show-sources --json`
  - `retrieve Q --filter --k`: table of rank, score, retriever, source, page and snippet
  - `chat --session ID`: REPL with streaming answers; supports `/sources`, `/reset`, `/config`, `/exit`
  - `store stats|reset|delete --source X`
  - `eval -c A -c B`
  - existing: `config show|validate` and `llm`
- **Shared behaviour:** `--verbose` prints per-node timings. Errors are reported as `RAGError` (exit code 1) or as a config error (exit code 2).

### Step 9: evaluation and observability
- **`evaluation/dataset.py`**: JSONL loader with a `QAItem` model.
- **`evaluation/metrics.py`**:
  - retrieval: `hit_rate@k`, `mrr`, `recall@k` (expected source matched as a path suffix against `metadata.source`)
  - LLM judge: `faithfulness`, `answer_relevance` and `correctness` via structured 0–1 scores, using `judge_llm` or falling back to `llm`
- **`evaluation/runner.py`**: `run_eval(configs, dataset) -> list[EvalReport]`. Each config gets a pipeline; the run ingests the dataset's corpus when `--ingest DIR` is given, then writes JSON to `output_dir/<ts>.json`. The CLI prints a comparison table.
- **`observability/tracing.py`**: `setup_tracing(cfg)` sets the `LANGSMITH_TRACING` and `LANGSMITH_PROJECT` env vars. Per-node timings are logged at INFO when `log_timings` is on, and a `TokenUsageCallback` sums `usage_metadata`.

### Step 10: docs and fixtures
- **`README.md`**: install, quickstart, config reference (pointing to `configs/default.yaml`), how to add a backend via the registry, and the CLI reference.
- **Fixtures:** `tests/fixtures/` gets `sample.md`, `sample.txt`, `sample.csv`, `sample.json`, `sample.html` and `qa.jsonl`, plus a `sample.pdf` generated by a test helper using pypdf (so no binary is committed).
- **Packaging:** `pyproject.toml` gets the README and the updated dependency set (core gains langgraph-checkpoint-sqlite, chromadb via langchain-chroma, rank-bm25 and pypdf; `langchain-community` is removed).

### Testing approach (all offline)
- `tests/conftest.py` provides:
  - `DeterministicFakeEmbedding`, and a keyword-hash embedding so that similar texts actually rank higher
  - `FakeChatModel`: scripted responses that also support `with_structured_output` by returning pre-set Pydantic objects in sequence
  - a `tmp_cfg` fixture that points `data_dir`, the store paths and memory at `tmp_path`
- **New test files**:
  - `test_loaders.py`, `test_cleaners_splitters.py`, `test_ingestion.py` (incremental skip, re-ingest on change, error isolation)
  - `test_stores.py`, parametrised over Chroma, FAISS and Qdrant local (each skipped if its library is missing; PGVector only with `PG_CONN`), covering add, search, filter, delete and the embedding guard
  - `test_retrieval.py` (BM25, RRF ordering, weighted fusion, parent retrieval, transforms, rerank and compression with fakes)
  - `test_generation.py`, `test_graph.py` (grade-and-rewrite loop, self-check loop, memory condensing across turns with sqlite)
  - `test_pipeline_cli.py` (ingest, query, retrieve and store commands end to end with fakes)
  - `test_eval.py`
- **Install for development:** `chromadb`/`langchain-chroma`, `faiss-cpu`, `qdrant-client`/`langchain-qdrant`, `pypdf`, `docx2txt`, `beautifulsoup4`, `rank-bm25`, `langgraph-checkpoint-sqlite`.
- **Checks:** `ruff check`, `ruff format`, `mypy rag` and `pytest` before each commit. There is one commit per step, each pushed to `claude/configurable-rag-plan-voik2a`.
- **Live checks:** model calls against real providers cannot run here (no API keys). The final report says so and gives the user the manual smoke-test commands.
