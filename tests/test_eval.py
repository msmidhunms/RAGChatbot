import json
import math

import pytest
from langchain_core.documents import Document

from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.evaluation.dataset import QAItem, load_dataset
from rag_chatbot.evaluation.gates import check_gates, parse_gate, parse_gates
from rag_chatbot.evaluation.metrics import (
    METRICS,
    answer_scores,
    exact_match,
    expand,
    is_refusal,
    judge_scores,
    keyword_coverage,
    latency_summary,
    percentile,
    retrieval_scores,
    source_matches,
    token_f1,
    usage_totals,
)
from rag_chatbot.evaluation.report import load_reports, render_markdown, worst_items, write_reports
from rag_chatbot.evaluation.runner import EvalReport, Evaluator, resolve_metrics, run_eval
from rag_chatbot.generation.generator import NO_ANSWER
from rag_chatbot.generation.schemas import Citation, GradeResult, JudgeScores, LLMAnswer, RAGAnswer
from rag_chatbot.pipeline import RAGPipeline
from tests.conftest import FIXTURES


def chunk(source, text="x", **meta):
    return RetrievedChunk(Document(page_content=text, metadata={"source": source, **meta}), 1.0)


def answer(text, ids=(), confidence="high"):
    return RAGAnswer(
        title="t", answer=text, confidence=confidence, citations=[Citation(id=i, source="s") for i in ids]
    )


# -------------------------------------------------------------- retrieval
def test_source_matches():
    assert source_matches("/abs/docs/sample.md", "sample.md")
    assert source_matches("sample.md", "./sample.md")
    assert not source_matches("docs/other-sample.md", "sample.md")


def test_retrieval_scores_by_source():
    chunks = [chunk("a/x.md"), chunk("a/y.md"), chunk("a/z.md")]
    item = QAItem(question="q", expected_sources=["y.md"])
    s = retrieval_scores(chunks, item, k=3)
    assert (s["hit_rate"], s["mrr"], s["recall"]) == (1.0, 0.5, 1.0)
    assert s["precision"] == pytest.approx(1 / 3)
    assert s["ndcg"] == pytest.approx(1 / math.log2(3))
    assert "evidence_recall" not in s and "filter_compliance" not in s


def test_retrieval_scores_evidence_and_k():
    chunks = [
        chunk("a.md", "Unrelated text."),
        chunk("b.md", "The   Drone X1 flies for 38 MINUTES per battery."),
    ]
    item = QAItem(question="q", evidence=["38 minutes per battery", "IP54"])
    s = retrieval_scores(chunks, item, k=2)
    assert s["hit_rate"] == 1.0 and s["mrr"] == 0.5 and s["evidence_recall"] == 0.5
    assert retrieval_scores(chunks, item, k=1)["hit_rate"] == 0.0  # relevant chunk beyond k


def test_retrieval_scores_filter_and_unanswerable():
    chunks = [chunk("a.csv", file_type="csv"), chunk("b.md", file_type="md")]
    filtered = QAItem(
        question="q", category="filtered", filter={"file_type": "csv"}, expected_sources=["a.csv"]
    )
    assert retrieval_scores(chunks, filtered, k=5)["filter_compliance"] == 0.5
    assert retrieval_scores([], filtered, k=5)["filter_compliance"] == 1.0
    assert retrieval_scores(chunks, QAItem(question="q", category="unanswerable"), k=5) == {}


def test_ndcg_perfect_ranking():
    chunks = [chunk("x.md"), chunk("y.md"), chunk("other.md")]
    item = QAItem(question="q", expected_sources=["x.md", "y.md"])
    assert retrieval_scores(chunks, item, k=3)["ndcg"] == pytest.approx(1.0)


# ----------------------------------------------------------------- answer
def test_text_overlap_metrics():
    assert exact_match("The answer is 5,290 EUR.", "the answer is 5290 eur") == 1.0
    # articles and citation markers are ignored; "planet" is the one extra token
    assert token_f1("Venus is the hottest planet [1]", "Venus is hottest") == pytest.approx(6 / 7)
    assert token_f1("Mars", "Venus") == 0.0
    assert token_f1("", "") == 1.0
    assert keyword_coverage("Costs 5290 EUR in total", ["5,290", "EUR", "Nimbus"]) == pytest.approx(2 / 3)


def test_is_refusal():
    assert is_refusal(answer(NO_ANSWER, confidence="low"))
    assert is_refusal(answer("Sorry, the documents do not contain that.", confidence="low"))
    assert not is_refusal(answer("The documents do not contain errors; X is 5 [1].", confidence="high"))


def test_answer_scores_answerable():
    item = QAItem(
        question="q", ground_truth="38 minutes", expected_sources=["x1.md"], answer_must_contain=["38"]
    )
    chunks = [chunk("/c/x1.md"), chunk("/c/other.md")]
    s = answer_scores(answer("38 minutes [1] and more [2] and [7]", ids=[1, 2]), chunks, item)
    assert s["refusal_accuracy"] == 1.0 and s["false_refusal_rate"] == 0.0
    assert s["keyword_coverage"] == 1.0 and 0 < s["token_f1"] < 1
    assert s["citation_validity"] == pytest.approx(2 / 3)
    assert s["citation_precision"] == 0.5


def test_answer_scores_unanswerable():
    item = QAItem(question="q", category="unanswerable")
    refused = answer_scores(answer(NO_ANSWER, confidence="low"), [], item)
    assert refused == {"refusal_accuracy": 1.0, "missed_refusal_rate": 0.0}
    answered = answer_scores(answer("It is 42 [1].", ids=[1]), [chunk("a.md")], item)
    assert answered["refusal_accuracy"] == 0.0 and answered["missed_refusal_rate"] == 1.0
    assert "citation_precision" not in answered


# ------------------------------------------------------------------ judge
def test_judge_scores(fake_llm):
    fake_llm.structured["JudgeScores"] = JudgeScores(
        faithfulness=0.9, answer_relevance=0.8, correctness=0.7, context_recall=0.6
    )
    fake_llm.structured["GradeResult"] = GradeResult(relevant=[1, 5])
    item = QAItem(question="q", ground_truth="ref")
    chunks = [chunk("a.md", "alpha"), chunk("b.md", "beta")]
    wanted = {"faithfulness", "correctness", "context_recall", "context_precision"}
    s = judge_scores(fake_llm, item, "q", "a", chunks, wanted)
    assert s == {"faithfulness": 0.9, "correctness": 0.7, "context_recall": 0.6, "context_precision": 0.5}
    no_ref = judge_scores(fake_llm, QAItem(question="q"), "q", "a", chunks, {"correctness", "faithfulness"})
    assert no_ref == {"faithfulness": 0.9}


# ------------------------------------------------------------ performance
def test_performance_helpers():
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([1, 2, 3, 4], 95) == 4
    assert math.isnan(percentile([], 50))
    assert latency_summary([1.0, 3.0])["latency_mean"] == 2.0 and latency_summary([]) == {}
    assert usage_totals([{"llm_calls": 1, "total_tokens": 10}, {"llm_calls": 2}]) == {
        "llm_calls": 3,
        "total_tokens": 10,
    }


def test_metric_registry():
    assert expand(["refusal_accuracy"]) == ["refusal_accuracy", "false_refusal_rate", "missed_refusal_rate"]
    assert METRICS["faithfulness"].needs_judge and METRICS["token_f1"].needs_answer
    assert not METRICS["hit_rate"].needs_answer and not METRICS["missed_refusal_rate"].higher_is_better
    with pytest.raises(ValueError, match="unknown metrics"):
        expand(["bogus"])
    assert "faithfulness" in resolve_metrics(["hit_rate"], None, judge=True)
    assert resolve_metrics(["hit_rate"], ["mrr"], judge=False) == ["mrr"]


# -------------------------------------------------------------- evaluator
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
    report = Evaluator(pipeline(), ["hit_rate", "mrr", "recall"]).run(load_dataset(FIXTURES / "qa.jsonl"))
    assert fake_llm.calls == []
    assert report.items == 4 and report.errors == 0
    assert report.metrics["hit_rate"] == 1.0 and 0 < report.metrics["mrr"] <= 1.0
    assert report.details[0]["sources"] and report.details[0]["answer"] is None
    assert report.latency["latency_p95"] >= 0 and report.counts == {"factual": 4}


def test_answer_and_judge_eval(pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = LLMAnswer(title="t", answer="a [1]", citations=[1], confidence="high")
    fake_llm.structured["JudgeScores"] = JudgeScores(
        faithfulness=0.8, answer_relevance=0.6, correctness=0.4, context_recall=1.0
    )
    items = load_dataset(FIXTURES / "qa.jsonl")
    items[1].ground_truth = None  # correctness only counted with a reference
    report = Evaluator(pipeline(), ["hit_rate", "faithfulness", "correctness", "refusal_accuracy"]).run(items)
    assert report.metrics["faithfulness"] == 0.8 and report.metrics["correctness"] == 0.4
    assert report.metrics["refusal_accuracy"] == 1.0 and report.metrics["false_refusal_rate"] == 0.0
    assert "correctness" not in report.details[1]["scores"] and report.details[0]["answer"] == "a [1]"
    assert "llm_calls" in report.usage  # the fake model's structured calls report no tokens
    assert report.value("factual.faithfulness") == 0.8 and report.value("latency_p95") is not None


def test_unanswerable_and_conversational_items(pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = LLMAnswer(
        title="t", answer="Venus [1]", citations=[1], confidence="high"
    )
    fake_llm.responses = ["Which planet is the hottest?"]  # condensed follow-up
    items = [
        QAItem(id="u", question="What is the CEO's shoe size?", category="unanswerable"),
        QAItem(
            id="c",
            question="And which one is the hottest?",
            category="conversational",
            history=["Tell me about the planets."],
            expected_sources=["sample.md"],
        ),
    ]
    pipe = pipeline()
    report = Evaluator(pipe, ["hit_rate", "refusal_accuracy"]).run(items)
    by_id = {d["id"]: d for d in report.details}
    assert by_id["u"]["scores"]["missed_refusal_rate"] == 1.0  # the fake model answered anyway
    assert by_id["c"]["standalone_question"] == "Which planet is the hottest?"
    assert by_id["c"]["scores"]["hit_rate"] == 1.0
    assert report.by_category["unanswerable"]["refusal_accuracy"] == 0.0
    assert pipe.history("eval-c") == []  # evaluation sessions are cleaned up


def test_errors_are_counted(pipeline, fake_llm):
    fake_llm.responses = ["text answer [1]"]  # no JudgeScores scripted -> judge fails per item
    report = Evaluator(pipeline(), ["faithfulness"]).run(load_dataset(FIXTURES / "qa.jsonl"))
    assert report.errors == 4 and all(d["error"] for d in report.details)
    assert report.metrics == {}


def test_run_eval_compares_configs_and_saves(pipeline, tmp_path):
    pipes = {"a": pipeline("retrieval.strategy=dense"), "b": pipeline("retrieval.strategy=sparse")}
    reports = run_eval(
        ["a.yaml", "b.yaml"],
        load_dataset(FIXTURES / "qa.jsonl"),
        metrics=["hit_rate", "mrr"],
        pipeline_factory=lambda path, ov: pipes[str(path)[0]],
    )
    assert [r.name for r in reports] == ["a", "b"]
    assert reports[0].config["retrieval"]["strategy"] == "dense"
    json_path, md_path = write_reports(reports, tmp_path / "out")
    saved = json.loads(json_path.read_text())["reports"]
    assert [r["name"] for r in saved] == ["a", "b"] and "hit_rate" in saved[0]["metrics"]
    assert "| hit_rate | " in md_path.read_text()


# ------------------------------------------------------------------ gates
def _report(name="cfg", **metrics):
    return EvalReport(
        name=name,
        items=3,
        metrics=metrics,
        by_category={"unanswerable": {"refusal_accuracy": 0.5}},
        latency={"latency_p95": 2.5},
        usage={"total_tokens": 900},
        selected_metrics=list(metrics),
    )


def test_parse_gate():
    assert str(parse_gate("hit_rate >= 0.8")) == "hit_rate>=0.8"
    assert parse_gate("mrr=0.5").op == ">="  # bare '=' means at least
    assert parse_gate("unanswerable.refusal_accuracy>0.9").target == "unanswerable.refusal_accuracy"
    assert len(parse_gates(["a>=1", "a >= 1", "b<=2"])) == 2
    for bad in ("hit_rate", "hit_rate>=x", ">=0.5", "hit rate>=1"):
        with pytest.raises(ConfigError, match="invalid gate"):
            parse_gate(bad)


def test_check_gates():
    results = check_gates(
        [_report(hit_rate=0.9)],
        parse_gates(
            [
                "hit_rate>=0.8",
                "latency_p95<=2",
                "unanswerable.refusal_accuracy>=0.9",
                "mrr>=0.1",
                "total_tokens<1000",
            ]
        ),
    )
    assert [r.passed for r in results] == [True, False, False, False, True]
    assert "not measured" in results[3].message and "actual 2.5" in results[1].message


# ---------------------------------------------------------------- reports
def test_markdown_report_and_worst_items(tmp_path):
    report = _report(hit_rate=0.5, token_f1=0.4)
    report.details = [
        {"id": "good", "category": "factual", "question": "q1", "scores": {"hit_rate": 1.0, "token_f1": 1.0}},
        {
            "id": "bad",
            "category": "factual",
            "question": "q2",
            "ground_truth": "x",
            "answer": "y",
            "sources": ["/a/b/doc.md"],
            "scores": {"hit_rate": 0.0, "token_f1": 0.0},
        },
        {"id": "boom", "category": "factual", "question": "q3", "error": "RuntimeError: x", "scores": {}},
    ]
    assert [d["id"] for d in worst_items(report)] == ["boom", "bad"]
    gates = check_gates([report], parse_gates(["hit_rate>=0.8"]))
    md = render_markdown([report], gates)
    assert "| hit_rate | 0.500 |" in md and "1 failed" in md and "### bad (factual)" in md
    assert "retrieved: doc.md" in md and "### good" not in md
    json_path, md_path = write_reports([report], tmp_path, gates)
    loaded, loaded_gates = load_reports(json_path)
    assert loaded[0].metrics == report.metrics and loaded_gates[0].passed is False
    with pytest.raises(ConfigError, match="not an evaluation report"):
        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        load_reports(bad)
