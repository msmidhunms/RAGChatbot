"""Offline benchmark fixtures: no API key, deterministic models, in-memory store."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_chatbot.evaluation.dataset import QAItem, load_dataset
from rag_chatbot.evaluation.offline import OFFLINE_UNSUPPORTED_CATEGORIES, build_offline_pipeline
from rag_chatbot.evaluation.runner import EvalReport, Evaluator

ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark"


@pytest.fixture(scope="session")
def baseline() -> dict:
    return yaml.safe_load((ROOT / "baseline.yaml").read_text())


@pytest.fixture(scope="session")
def benchmark_items() -> list[QAItem]:
    items = load_dataset(BENCHMARK / "qa.jsonl")
    return [i for i in items if i.category not in OFFLINE_UNSUPPORTED_CATEGORIES]


@pytest.fixture(scope="session")
def run_benchmark(tmp_path_factory, benchmark_items):
    """Evaluate one retrieval setup on the benchmark; cached per setup for the session."""
    cache: dict[tuple[str, str], EvalReport] = {}

    def run(strategy: str, search_type: str) -> EvalReport:
        key = (strategy, search_type)
        if key not in cache:
            data_dir = tmp_path_factory.mktemp(f"{strategy}-{search_type}")
            pipeline = build_offline_pipeline(
                None,
                [
                    f"app.data_dir={data_dir}",
                    "ingestion.json.jq_schema=.faq[]",
                    f"retrieval.strategy={strategy}",
                    f"retrieval.search_type={search_type}",
                ],
            )
            pipeline.ingest([str(BENCHMARK / "corpus")])
            evaluator = Evaluator(pipeline, pipeline.cfg.evaluation.metrics)
            cache[key] = evaluator.run(benchmark_items, name=f"{strategy}-{search_type}")
        return cache[key]

    return run
