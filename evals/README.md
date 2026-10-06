# Evaluations

## Bundled benchmark

`benchmark/corpus/` holds 10 short documents about **Nimbus Robotics**, a fictional company. Because the
company is made up, a model cannot answer from what it already knows, so the scores really measure retrieval
and grounding. The documents cover several formats: Markdown, plain text, HTML, CSV and JSON.

`benchmark/qa.jsonl` has 40 hand-written questions:

| category | count | what it tests |
|---|---|---|
| factual | 18 | single-fact lookup |
| multi_hop | 8 | combining facts from two documents (sometimes with arithmetic) |
| unanswerable | 6 | plausible questions the corpus cannot answer: the bot must refuse |
| conversational | 5 | follow-ups ("and what is its payload?") that need the chat history |
| filtered | 3 | answers restricted by a metadata filter (`file_type`) |

## Dataset format

One JSON object per line. Only `question` is required:

```json
{"id": "mh-x1-care", "category": "multi_hop",
 "question": "What is the total cost of a Drone X1 bought together with the Nimbus Care extended warranty?",
 "ground_truth": "5,290 EUR (4,900 EUR for the drone plus 390 EUR for Nimbus Care).",
 "expected_sources": ["pricing.csv", "warranty.txt"],
 "evidence": ["4900", "costs 390 EUR"],
 "answer_must_contain": ["5,290"]}
```

| field | meaning |
|---|---|
| `category` | `factual`, `multi_hop`, `unanswerable`, `conversational` or `filtered` |
| `ground_truth` | reference answer, used for `exact_match`, `token_f1`, `correctness` and `context_recall` |
| `expected_sources` | file names (path suffixes) that contain the answer |
| `evidence` | short verbatim snippets that must be in the retrieved context; this makes a chunk count as relevant |
| `answer_must_contain` | keywords the answer must include (`keyword_coverage`) |
| `history` | earlier user turns for conversational questions |
| `filter` | metadata filter applied to the query, e.g. `{"file_type": "csv"}` |

`tests/test_eval_dataset.py` checks that every evidence snippet still appears in its source after loading,
so editing the corpus cannot silently break the benchmark.

## Your own documents

```bash
rag ingest docs/
rag eval generate -o data/eval/mine.jsonl --n 40
```

`rag eval generate` samples chunks from the index (round-robin across documents, reproducible with
`--seed`) and asks the configured LLM for factual, multi-hop, unanswerable and conversational questions.
Items whose evidence is not quoted verbatim from the source, and near-duplicate questions, are rejected.
**Review the generated questions and answers** before trusting them, then run
`rag eval run --dataset data/eval/mine.jsonl`.

## Offline mode and the regression suite

`rag eval run --offline` and `pytest evals` use deterministic stand-ins: hashing embeddings, an extractive
"model" that quotes the best-matching sentence (or refuses), and an in-memory store. No API key or network
is needed. The scores measure the retrieval stack and a simple extractive baseline, not a real LLM, so
LLM-judge metrics and conversational questions are skipped.

`evals/baseline.yaml` holds the gates the suite enforces. When a deliberate change moves the scores,
re-measure them with `pytest evals -s` and update the file.
