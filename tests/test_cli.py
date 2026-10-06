import json
import pathlib

import pytest
from langchain_core.language_models import FakeListChatModel
from typer.testing import CliRunner

from rag_chatbot import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env / RAG_CONFIG from the repo
    for var in ("RAG_CONFIG", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_config_show_with_override():
    result = runner.invoke(cli.app, ["config", "show", "--set", "retrieval.k=9"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["retrieval"]["k"] == 9


def test_config_show_invalid_exits_2():
    result = runner.invoke(cli.app, ["config", "show", "--set", "splitter.chunk_overlap=5000"])
    assert result.exit_code == 2
    assert "chunk_overlap" in result.output


def test_config_validate_reports_missing_key():
    result = runner.invoke(cli.app, ["config", "validate"])
    assert result.exit_code == 1
    assert "GOOGLE_API_KEY" in result.output


def test_config_validate_ok(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    result = runner.invoke(cli.app, ["config", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_llm_command_raw(monkeypatch):
    monkeypatch.setattr("rag_chatbot.providers.build_llm", lambda cfg: FakeListChatModel(responses=["pong"]))
    result = runner.invoke(cli.app, ["llm", "ping", "--raw"])
    assert result.exit_code == 0, result.output
    assert "pong" in result.output


def test_llm_command_missing_key():
    result = runner.invoke(cli.app, ["llm", "ping"])
    assert result.exit_code == 1
    assert "GOOGLE_API_KEY" in result.output


# ------------------------------------------------------- end-to-end commands
@pytest.fixture
def e2e(monkeypatch, tmp_path, keyword_embeddings, fake_llm):
    """Run CLI commands against a FAISS-backed store with fake models."""
    from rag_chatbot.generation.schemas import LLMAnswer
    from rag_chatbot.pipeline import RAGPipeline

    pytest.importorskip("faiss")
    monkeypatch.setattr(
        cli, "_pipeline", lambda cfg: RAGPipeline(cfg, llm=fake_llm, embeddings=keyword_embeddings)
    )
    fake_llm.structured["LLMAnswer"] = LLMAnswer(
        title="Espresso pressure", answer="About nine bars [1].", citations=[1], confidence="high"
    )
    base = [
        "--set",
        f"app.data_dir={tmp_path / 'data'}",
        "--set",
        "vector_store.type=faiss",
        "--set",
        f"vector_store.faiss.index_dir={tmp_path / 'faiss'}",
        "--set",
        f"memory.path={tmp_path / 'sessions.sqlite'}",
        "--set",
        "ingestion.cleaning.min_chars=10",
    ]

    def run(*args, input=None):
        return runner.invoke(cli.app, [*args, *base], input=input)

    return run


def test_ingest_query_retrieve_store(e2e, corpus):
    result = e2e("ingest", str(corpus), "--tag", "team=docs")
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output

    again = e2e("ingest", str(corpus))
    assert "unchanged (skipped)" in again.output

    q = e2e("query", "espresso pressure", "--show-sources", "-v")
    assert q.exit_code == 0, q.output
    assert "About nine bars [1]." in q.output and "sample.html" in q.output and "timings" in q.output

    js = e2e("query", "espresso pressure", "--json")
    data = json.loads(js.stdout)
    assert data["answer"]["citations"][0]["source"].endswith("sample.html")
    assert data["sources"][0]["team"] == "docs"

    r = e2e("retrieve", "espresso pressure", "--filter", "file_type=html", "--json")
    rows = json.loads(r.stdout)
    assert rows and all(row["metadata"]["file_type"] == "html" for row in rows)

    stats = e2e("store", "stats")
    assert "sample.md" in stats.output and "faiss" in stats.output

    deleted = e2e("store", "delete", "--source", "sample.md")  # unique suffix of the full path
    assert deleted.exit_code == 0 and "deleted" in deleted.output
    assert e2e("store", "delete", "--source", "sample.md").exit_code == 1

    assert e2e("store", "reset", "--yes").exit_code == 0
    assert "no results" in e2e("retrieve", "espresso").output


def test_ingest_reports_errors(e2e, tmp_path):
    result = e2e("ingest", str(tmp_path / "missing.md"))
    assert result.exit_code == 1 and "missing.md" in result.output


def test_chat_repl(e2e, corpus):
    e2e("ingest", str(corpus))
    script = "\n".join(
        ["/help", "espresso pressure?", "/sources", "/history", "/config", "/bogus", "/reset", "/exit"]
    )
    result = e2e("chat", "--session", "t1", "--no-stream", input=script + "\n")
    assert result.exit_code == 0, result.output
    assert "About nine bars [1]." in result.output
    assert "Last retrieved context" in result.output
    assert "human: espresso pressure?" in result.output
    assert "session cleared" in result.output and "unknown command" in result.output


def test_chat_streaming(e2e, fake_llm, corpus):
    e2e("ingest", str(corpus))
    fake_llm.responses = ["Nine bars of pressure [1]."]
    result = e2e("chat", input="espresso pressure?\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Nine bars of pressure [1]." in result.output and "Sources:" in result.output


def test_bad_filter_and_tag(e2e):
    assert e2e("query", "x", "--filter", "nonsense").exit_code == 2
    assert e2e("ingest", "--tag", "x").exit_code == 2


def test_eval_command(e2e, monkeypatch, keyword_embeddings, fake_llm, tmp_path, corpus):
    from rag_chatbot.generation.schemas import JudgeScores
    from rag_chatbot.pipeline import RAGPipeline
    from tests.conftest import FIXTURES

    fake_llm.structured["JudgeScores"] = JudgeScores(faithfulness=1, answer_relevance=0.5, correctness=1)
    original = RAGPipeline.from_config.__func__

    def from_config(cls, path=None, overrides=(), **kw):
        return original(cls, path, overrides, llm=fake_llm, embeddings=keyword_embeddings)

    monkeypatch.setattr(RAGPipeline, "from_config", classmethod(from_config))
    result = e2e(
        "eval",
        "run",
        "--dataset",
        str(FIXTURES / "qa.jsonl"),
        "--ingest",
        str(corpus),
        "--set",
        f"evaluation.output_dir={tmp_path / 'eval'}",
        "--set",
        "evaluation.metrics=[hit_rate, mrr, faithfulness]",  # judge metrics are opt-in
    )
    assert result.exit_code == 0, result.output
    assert "hit_rate" in result.output and "faithfulness" in result.output
    assert list((tmp_path / "eval").glob("eval-*.json")) and list((tmp_path / "eval").glob("eval-*.md"))
    failing = e2e(
        "eval", "run", "--dataset", str(FIXTURES / "qa.jsonl"), "--ingest", str(corpus),
        "--set", f"evaluation.output_dir={tmp_path / 'eval'}", "--set", "evaluation.metrics=[hit_rate]",
        "--fail-under", "hit_rate>=1.5",
    )  # fmt: skip
    assert failing.exit_code == 3 and "FAIL" in failing.output


# --------------------------------------------------------------- error handling
class _Boom:
    def query(self, *a, **k):
        raise RuntimeError("upstream 503")


def test_unexpected_errors_print_one_line(monkeypatch):
    monkeypatch.setattr(cli, "_pipeline", lambda cfg: _Boom())
    result = runner.invoke(cli.app, ["query", "x"])
    assert result.exit_code == 1
    assert "error: RuntimeError: upstream 503" in result.output
    assert "Traceback" not in result.output


def test_debug_env_shows_traceback(monkeypatch):
    monkeypatch.setattr(cli, "_pipeline", lambda cfg: _Boom())
    monkeypatch.setenv("RAG_DEBUG", "1")
    result = runner.invoke(cli.app, ["query", "x"])
    assert isinstance(result.exception, RuntimeError)


def test_store_stats_needs_no_api_key(tmp_path, keyword_embeddings, corpus):
    """Maintenance commands never embed, so they must work without provider credentials."""
    pytest.importorskip("faiss")
    from rag_chatbot.config import load_config
    from rag_chatbot.pipeline import RAGPipeline

    sets = [
        f"app.data_dir={tmp_path / 'data'}",
        "vector_store.type=faiss",
        f"vector_store.faiss.index_dir={tmp_path / 'faiss'}",
        "ingestion.cleaning.min_chars=10",
    ]
    cfg = load_config(overrides=sets, env={})
    RAGPipeline(cfg, embeddings=keyword_embeddings).ingest([str(corpus)])

    args = [x for s in sets for x in ("--set", s)]
    stats = runner.invoke(cli.app, ["store", "stats", *args])  # real pipeline, GOOGLE_API_KEY unset
    assert stats.exit_code == 0, stats.output
    assert "sample.md" in stats.output
    deleted = runner.invoke(cli.app, ["store", "delete", "--source", "sample.md", *args])
    assert deleted.exit_code == 0, deleted.output
    query = runner.invoke(cli.app, ["retrieve", "espresso", *args])
    assert query.exit_code == 1 and "GOOGLE_API_KEY" in query.output


# ------------------------------------------------------------- eval group
BENCHMARK = pathlib.Path(__file__).resolve().parent.parent / "evals" / "benchmark"
BENCH_SETS = [
    "--set", f"evaluation.dataset={BENCHMARK / 'qa.jsonl'}",
    "--set", f"evaluation.corpus={BENCHMARK / 'corpus'}",
]  # fmt: skip


def test_eval_run_offline_and_report(tmp_path):
    out = tmp_path / "reports"
    result = runner.invoke(
        cli.app,
        ["eval", "run", "--offline", "--report-dir", str(out), "--set", f"app.data_dir={tmp_path / 'data'}",
         "--category", "factual", "--category", "unanswerable", "--fail-under", "hit_rate>=0.5", *BENCH_SETS],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "hit_rate" in result.output and "factual" in result.output and "pass" in result.output
    md = next(out.glob("eval-*.md")).read_text()
    assert "## By category - defaults" in md and "unanswerable" in md
    json_report = next(out.glob("eval-*.json"))
    shown = runner.invoke(cli.app, ["eval", "report", str(json_report)])
    assert shown.exit_code == 0 and "hit_rate" in shown.output
    as_md = runner.invoke(cli.app, ["eval", "report", str(json_report), "--markdown"])
    assert as_md.output.startswith("# Evaluation report")


def test_eval_run_offline_skips_llm_only_work(tmp_path):
    result = runner.invoke(
        cli.app,
        ["eval", "run", "--offline", "--judge", "--limit", "40", "--report-dir", str(tmp_path),
         "--set", f"app.data_dir={tmp_path / 'data'}", *BENCH_SETS],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "skipped 5 conversational questions" in result.output
    assert "faithfulness" not in result.output


def test_eval_generate_command(e2e, corpus, fake_llm, tmp_path):
    from rag_chatbot.generation.schemas import GeneratedQA

    counter = iter(range(100))

    def factual(messages):
        text = messages[-1].content.split("Excerpt:\n", 1)[1]
        n = next(counter)
        return GeneratedQA(
            question=f"Topic{n} item{n} question?", answer="a", evidence=[" ".join(text.split()[:5])]
        )

    fake_llm.structured["GeneratedQA"] = factual
    out = tmp_path / "gen.jsonl"
    result = e2e(
        "eval", "generate", "-o", str(out), "--n", "3", "--category", "factual", "--ingest", str(corpus)
    )
    assert result.exit_code == 0, result.output
    assert "wrote 3 questions" in result.output and len(out.read_text().splitlines()) == 3
