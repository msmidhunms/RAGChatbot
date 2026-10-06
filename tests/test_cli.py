import json

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
        "--dataset",
        str(FIXTURES / "qa.jsonl"),
        "--ingest",
        str(corpus),
        "--set",
        f"evaluation.output_dir={tmp_path / 'eval'}",
    )
    assert result.exit_code == 0, result.output
    assert "hit_rate" in result.output and "faithfulness" in result.output
    assert list((tmp_path / "eval").glob("eval-*.json"))
