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
  model: gemini-1.5-flash
  temperature: 0.1
  max_tokens: 1024
  timeout: 60
  max_retries: 2
  structured_output: true
embeddings:
  provider: google              # google | openai | ollama | huggingface
  model: models/text-embedding-004
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
