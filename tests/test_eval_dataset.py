import json
import re
from pathlib import Path

import pytest

from rag_chatbot.config.schema import CleaningConfig, IngestionConfig
from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.evaluation.dataset import QAItem, filter_items, load_dataset, save_dataset
from rag_chatbot.ingestion.cleaners import clean_documents
from rag_chatbot.ingestion.loaders import discover_sources, load_source

BENCHMARK = Path(__file__).resolve().parent.parent / "evals" / "benchmark"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def test_minimal_item_defaults():
    item = QAItem(question="What is X?")
    assert item.category == "factual" and item.answerable
    assert item.id.startswith("q-") and item.id == QAItem(question="What is X?").id


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"question": "q", "category": "unanswerable", "expected_sources": ["a.md"]}, "unanswerable"),
        ({"question": "q", "category": "conversational"}, "history"),
        ({"question": "q", "category": "filtered"}, "filter"),
        ({"question": "q", "category": "opinion"}, "category"),
        ({"question": ""}, "question"),
    ],
)
def test_item_validation(data, fragment):
    with pytest.raises(ValueError, match=fragment):
        QAItem.model_validate(data)


def test_save_and_load_roundtrip(tmp_path):
    items = [
        QAItem(question="a?", ground_truth="x", expected_sources=["a.md"], evidence=["x"]),
        QAItem(question="b?", category="conversational", history=["a?"]),
    ]
    path = save_dataset(items, tmp_path / "out" / "qa.jsonl")
    assert load_dataset(path) == items
    first = json.loads(path.read_text().splitlines()[0])
    assert "history" not in first and "generated" not in first  # defaults are omitted


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text('{"id": "a", "question": "x"}\n{"id": "a", "question": "y"}\n')
    with pytest.raises(ConfigError, match="duplicate item ids: a"):
        load_dataset(path)


def test_filter_items():
    items = [QAItem(question=f"q{i}", category=c) for i, c in enumerate(["factual", "multi_hop", "factual"])]
    assert len(filter_items(items, ["factual"])) == 2
    assert len(filter_items(items, limit=1)) == 1
    with pytest.raises(ConfigError, match="unknown categories"):
        filter_items(items, ["nope"])


def test_legacy_fixture_still_loads():
    items = load_dataset(Path(__file__).parent / "fixtures" / "qa.jsonl")
    assert all(i.category == "factual" for i in items)


# --------------------------------------------------------- bundled benchmark
def test_benchmark_composition():
    items = load_dataset(BENCHMARK / "qa.jsonl")
    counts = {c: sum(i.category == c for i in items) for c in {i.category for i in items}}
    assert len(items) >= 40
    assert counts["unanswerable"] >= 5 and counts["multi_hop"] >= 5 and counts["conversational"] >= 4
    assert all(i.expected_sources and i.evidence and i.ground_truth for i in items if i.answerable)


def test_benchmark_evidence_survives_ingestion():
    """Every evidence snippet must be present in its expected source after loading and cleaning."""
    cfg = IngestionConfig()
    texts = {}
    for src in discover_sources([str(BENCHMARK / "corpus")], cfg=cfg):
        docs = clean_documents(load_source(src, cfg), CleaningConfig())
        texts[Path(src).name] = _norm(" ".join(d.page_content for d in docs))
    for item in load_dataset(BENCHMARK / "qa.jsonl"):
        for source in item.expected_sources:
            assert Path(source).name in texts, f"{item.id}: unknown source {source}"
        for snippet in item.evidence:
            assert any(_norm(snippet) in texts[Path(s).name] for s in item.expected_sources), (
                f"{item.id}: evidence {snippet!r} not found in {item.expected_sources}"
            )
