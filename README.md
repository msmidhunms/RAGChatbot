# RAGChatbot

A retrieval-augmented generation (RAG) chatbot where every stage is chosen and tuned from one YAML file:
loaders, chunking, embeddings, vector store, retrieval strategy, query rewriting, reranking, compression,
prompting, conversation memory and evaluation. Built on LangChain and LangGraph.

```
documents ─► load ─► clean ─► split ─► embed ─► vector store + docstore
                                                      │
question ─► condense (memory) ─► transform ─► retrieve (dense | BM25 | hybrid | parent)
        ─► compress / rerank ─► [grade → rewrite] ─► generate ─► [self-check → regenerate] ─► answer + citations
```

## Install

```bash
pip install -r requirements.txt     # locked core dependencies (compiled by uv from requirements.in)
pip install -e .                     # core: Gemini, Chroma, BM25, PDF/TXT/MD/HTML/CSV/JSON
pip install -e '.[openai,anthropic]' # more LLM / embedding providers
pip install -e '.[local]'            # Ollama, HuggingFace embeddings, cross-encoder reranker
pip install -e '.[faiss,qdrant,pgvector]'
pip install -e '.[docs]'             # DOCX and PyMuPDF
pip install -e '.[all,dev]'
cp .env.example .env                 # then add the keys your config needs
```

Python 3.10+. Secrets live only in environment variables / `.env`; configs reference them by name.

## Quickstart

```bash
rag config validate                      # checks settings and that required API keys are set
rag llm "hello"                          # smoke-test the configured chat model
rag ingest docs/ https://example.com/faq # files, directories or URLs
rag query "How do I reset my password?" --show-sources -v
rag chat                                 # interactive, streaming, with memory (/help for commands)
```

`python main.py ...` is equivalent to `rag ...`.

## Configuration

Settings are layered, later wins:

1. built-in defaults (documented in full in [`configs/default.yaml`](configs/default.yaml))
2. a YAML file: `--config configs/local.yaml` or `RAG_CONFIG=...`
3. environment variables: `RAG__RETRIEVAL__K=10`
4. command-line overrides: `--set retrieval.k=10 --set reranker.type=llm`

Unknown keys and inconsistent values (e.g. `chunk_overlap >= chunk_size`, hybrid weights not summing to 1)
are rejected with a precise message. Sample configs:

| File | Setup |
|---|---|
| `configs/default.yaml` | Gemini + Google embeddings, Chroma, recursive chunks, hybrid (dense + BM25, RRF) with MMR |
| `configs/local.yaml` | fully offline: Ollama + sentence-transformers + FAISS + cross-encoder reranking |
| `configs/advanced.yaml` | Qdrant, header-aware chunking, multi-query, reranking, redundancy filter, grading and self-check |

### What can be configured

| Section | Options |
|---|---|
| `llm` | `google_genai`, `openai`, `anthropic`, `ollama`; model, temperature, max tokens, timeout, retries, structured output |
| `embeddings` | `google`, `openai`, `ollama`, `huggingface`; batch size, normalization, persistent cache |
| `ingestion` | sources, glob/exclude, per-extension loader, CSV/JSON field selection, URL crawling (same host, HTML/PDF/text, 20 MB per response), cleaning, dedupe, incremental |
| `splitter` | `recursive`, `token`, `markdown_header`, `html_header`, `semantic`; size, overlap, separators |
| `vector_store` | `chroma`, `faiss`, `qdrant` (local or server), `pgvector`, `memory`; collection, distance |
| `retrieval` | `dense`, `sparse` (BM25), `hybrid` (RRF or weighted), `parent` (small-to-big); `similarity`, `mmr`, `threshold`; k, fetch_k, default filters |
| `query_transform` | `none`, `rewrite`, `multi_query`, `hyde`, `step_back`; follow-up condensing |
| `reranker` | `none`, `cross_encoder`, `flashrank` (fast, local, no API key), `cohere`, `llm`; top_n |
| `compression` | `none`, `embeddings_filter`, `redundant_filter`, `llm_extract` |
| `generation` | prompt template (built-in or file), context token budget, citation style, refuse vs. general knowledge, document grading, self-check, retries |
| `memory` | `none`, `buffer`, `window`, `summary`; in-memory or SQLite sessions |
| `evaluation` | dataset, metrics, judge model |
| `observability` | LangSmith tracing, timing logs |

Switching the embedding model on an existing collection is detected and refused (mixing vector spaces silently
breaks retrieval): run `rag store reset` or use a new `vector_store.collection`.

Similarity scores mean the same thing on every store, so `retrieval.score_threshold` is portable:
`cosine` → cosine similarity (-1..1), `ip` → dot product, `l2` → `1 / (1 + euclidean distance)`.
PGVector keeps all collections in shared tables, so one Postgres database can only hold one embedding size.

## CLI

| Command | Purpose |
|---|---|
| `rag ingest [PATHS] [--tag k=v] [--reset] [--no-incremental]` | index documents; unchanged files are skipped |
| `rag query "Q" [-f key=value] [--show-sources] [--json] [-v]` | one-shot answer with citations, timings and token usage |
| `rag retrieve "Q" [-f key=value] [--json]` | inspect retrieval only (ranks, scores, retriever) |
| `rag chat [--session ID] [--no-stream]` | conversation with memory; `/sources /history /reset /config /exit` |
| `rag store stats \| reset \| delete --source S` | inspect or maintain the index (`S` may be a unique path suffix) |
| `rag eval -c a.yaml -c b.yaml [--dataset qa.jsonl] [--ingest DIR]` | compare configurations on a QA set |
| `rag config show \| validate` | print merged config / check it and the credentials |

Filters use `key=value` (equality) or operators: `-f 'page={$gte: 3}'`, `-f 'source={$in: [a.md, b.md]}'`
(`$eq $ne $in $nin $gt $gte $lt $lte`).

## Evaluation

A dataset is JSONL with `question`, optional `ground_truth` and `expected_sources` (path suffixes):

```json
{"question": "Which planet is the hottest?", "ground_truth": "Venus", "expected_sources": ["planets.md"]}
```

Retrieval metrics (`hit_rate`, `mrr`, `recall`) need no LLM. `faithfulness`, `answer_relevance` and
`correctness` use an LLM judge (`evaluation.judge_llm`, defaulting to `llm`). Reports are written to
`evaluation.output_dir`.

## Python API

```python
from rag_chatbot.pipeline import RAGPipeline

rag = RAGPipeline.from_config("configs/default.yaml", ["retrieval.k=8"])
rag.ingest(["docs/"])
result = rag.query("What is the refund policy?")
print(result.answer.answer, result.answer.citations)

session = rag.new_session_id()
rag.chat("Who wrote the report?", session)
for item in rag.stream_chat("When was it published?", session):
    print(item if isinstance(item, str) else "", end="")
```

## Extending

Each component family has a registry; adding a backend is one decorated function, no core changes:

```python
from rag_chatbot.retrieval.rerankers import RERANKERS, _apply

@RERANKERS.register("by_length")
def by_length(cfg, get_llm):
    return lambda question, chunks: _apply(chunks, [len(c.text) for c in chunks], cfg.top_n, "rerank:length")
```

Registries: `LOADERS`, `SPLITTERS`, `STORES`, `EMBEDDING_PROVIDERS`, `LLM_PROVIDERS`, `QUERY_TRANSFORMS`,
`COMPRESSORS`, `RERANKERS`, prompt `STYLE`s. Selecting a new name in YAML also requires adding it to the
corresponding `Literal` in `rag_chatbot/config/schema.py`, which keeps typos caught at load time.

## Project layout

```
rag_chatbot/config        schema (pydantic) and layered loader
rag_chatbot/core          registry, exceptions, logging, shared types
rag_chatbot/providers     chat model and embeddings factories
rag_chatbot/cache.py      persistent embedding cache
rag_chatbot/ingestion     loaders, cleaners, splitters, metadata, manifest, pipeline
rag_chatbot/stores        vector store backends, filters, SQLite docstore
rag_chatbot/retrieval     dense, BM25, hybrid, parent, transforms, compression, rerankers
rag_chatbot/generation    prompts, schemas, context budgeting, answer generator
rag_chatbot/memory        history selection, summaries, checkpointers
rag_chatbot/graph         LangGraph state, nodes and builder
rag_chatbot/evaluation    datasets, metrics, runner
rag_chatbot/pipeline.py   RAGPipeline facade used by the CLI and API
```

## Development

```bash
pip install -e '.[dev,faiss,qdrant]'
pytest            # fully offline: fake LLM and keyword embeddings
ruff check . && ruff format --check . && mypy rag_chatbot
```

PGVector tests run when `PG_CONN` points at a Postgres with the `vector` extension.
