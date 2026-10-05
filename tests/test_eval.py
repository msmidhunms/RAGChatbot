import json

import pytest
from langchain_core.documents import Document

from rag.core.exceptions import ConfigError
from rag.core.types import RetrievedChunk
from rag.evaluation.dataset import load_dataset
from rag.evaluation.metrics import retrieval_scores, source_matches
from rag.evaluation.runner import evaluate, run_eval, save_reports
from rag.generation.schemas import JudgeScores, LLMAnswer
from rag.pipeline import RAGPipeline
from tests.conftest import FIXTURES


def chunk(source):
    return RetrievedChunk(Document(page_content="x", metadata={"source": source}), 1.0)


def test_source_matches():
    assert source_matches("docs/sample.md", "sample.md")
    assert source_matches("sample.md", "./sample.md")
    assert not source_matches("docs/other-sample.md", "sample.md")


def test_retrieval_scores():
    chunks = [chunk("a/x.md"), chunk("a/y.md"), chunk("a/z.md")]
    assert retrieval_scores(chunks, ["y.md"]) == {"hit_rate": 1.0, "mrr": 0.5, "recall": 1.0}
    assert retrieval_scores(chunks, ["y.md", "q.md"])["recall"] == 0.5
    assert retrieval_scores(chunks, ["q.md"]) == {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}


def test_load_dataset(tmp_path):
    items = load_dataset(FIXTURES / "qa.jsonl")
    assert len(items) == 4 and items[0].expected_sources == ["sample.md"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "ok"}\n{"nope": 1}\n')
    with pytest.raises(ConfigError, match="bad.jsonl:2"):
        load_dataset(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# only a comment\n")
    with pytest.raises(ConfigError, match="empty"):
        load_dataset(empty)
    with pytest.raises(ConfigError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


@pytest.fixture
def pipeline(make_cfg, keyword_embeddings, fake_llm, corpus):
    def build(*overrides):
        cfg = make_cfg(
            "ingestion.cleaning.min_chars=10",
            "splitter.chunk_size=300",
            "splitter.chunk_overlap=30",
            *overrides,
        )
        pipe = RAGPipeline(cfg, llm=fake_llm, embeddings=keyword_embeddings)
        pipe.ingest([str(corpus)])
        return pipe

    return build


def test_retrieval_only_eval_makes_no_llm_calls(pipeline, fake_llm):
    report = evaluate(pipeline(), load_dataset(FIXTURES / "qa.jsonl"), ["hit_rate", "mrr", "recall"])
    assert fake_llm.calls == []
    assert report.items == 4 and report.errors == 0
    assert report.metrics["hit_rate"] == 1.0
    assert 0 < report.metrics["mrr"] <= 1.0
    assert report.details[0]["sources"]


def test_judge_metrics(pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = LLMAnswer(title="t", answer="a [1]", citations=[1], confidence="high")
    fake_llm.structured["JudgeScores"] = JudgeScores(faithfulness=0.8, answer_relevance=0.6, correctness=0.4)
    items = load_dataset(FIXTURES / "qa.jsonl")
    items[1].ground_truth = None  # correctness only counted with a reference
    report = evaluate(pipeline(), items, ["hit_rate", "faithfulness", "answer_relevance", "correctness"])
    assert report.metrics == {
        "hit_rate": 1.0,
        "faithfulness": 0.8,
        "answer_relevance": 0.6,
        "correctness": 0.4,
    }
    assert "correctness" not in report.details[1] and report.details[0]["answer"] == "a [1]"


def test_errors_are_counted(pipeline, fake_llm):
    # no LLMAnswer/JudgeScores configured -> judge raises; evaluation continues
    fake_llm.responses = ["text answer [1]"]
    report = evaluate(pipeline(), load_dataset(FIXTURES / "qa.jsonl"), ["faithfulness"])
    assert report.errors == 4 and all("error" in d for d in report.details)


def test_run_eval_compares_configs(pipeline, tmp_path, corpus):
    pipes = {"a": pipeline("retrieval.strategy=dense"), "b": pipeline("retrieval.strategy=sparse")}
    reports = run_eval(
        ["a.yaml", "b.yaml"],
        load_dataset(FIXTURES / "qa.jsonl"),
        pipeline_factory=lambda path, ov: pipes[str(path)[0]],
    )
    assert [r.name for r in reports] == ["a", "b"]
    out = save_reports(reports, tmp_path / "out")
    saved = json.loads(out.read_text())
    assert [r["name"] for r in saved] == ["a", "b"] and "hit_rate" in saved[0]["metrics"]
