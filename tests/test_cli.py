import json

import pytest
from langchain_core.language_models import FakeListChatModel
from typer.testing import CliRunner

from rag import cli

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
    monkeypatch.setattr("rag.providers.build_llm", lambda cfg: FakeListChatModel(responses=["pong"]))
    result = runner.invoke(cli.app, ["llm", "ping", "--raw"])
    assert result.exit_code == 0, result.output
    assert "pong" in result.output


def test_llm_command_missing_key():
    result = runner.invoke(cli.app, ["llm", "ping"])
    assert result.exit_code == 1
    assert "GOOGLE_API_KEY" in result.output
