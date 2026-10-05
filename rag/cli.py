"""Command line interface.

Only configuration and provider smoke-test commands exist so far; ingest,
query, chat, retrieve, eval and store commands arrive with their modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import BaseModel, Field
from rich.console import Console

from rag.config import RAGConfig, dump_config, load_config, missing_env_vars
from rag.core.exceptions import RAGError
from rag.core.logging import configure_logging

app = typer.Typer(help="Configurable RAG chatbot.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()
err_console = Console(stderr=True)

ConfigOpt = Annotated[
    Optional[Path],  # noqa: UP045 - typer needs Optional on py3.10
    typer.Option("--config", "-c", help="YAML config file (default: $RAG_CONFIG or built-in defaults)."),
]
SetOpt = Annotated[
    Optional[list[str]],  # noqa: UP045
    typer.Option("--set", "-s", help="Override a setting, e.g. --set retrieval.k=10. Repeatable."),
]


def _load(config: Path | None, overrides: list[str] | None) -> RAGConfig:
    try:
        cfg = load_config(config, overrides or [])
    except RAGError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    configure_logging(cfg.app.log_level)
    return cfg


@config_app.command("show")
def config_show(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Print the fully merged configuration as JSON."""
    cfg = _load(config, set_)
    console.print_json(json.dumps(dump_config(cfg)))


@config_app.command("validate")
def config_validate(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Validate the configuration and check required credentials are set."""
    cfg = _load(config, set_)
    missing = missing_env_vars(cfg)
    if missing:
        for var, why in missing.items():
            err_console.print(f"[red]missing env var {var}[/red] (needed by {why})")
        raise typer.Exit(1)
    console.print("[green]configuration is valid[/green]")


class LLMReply(BaseModel):
    """Structured reply used by the provider smoke test (formerly main.py)."""

    title: str = Field(description="Short title for the answer")
    answer: str = Field(description="The answer to the prompt")


@app.command("llm")
def llm_command(
    prompt: Annotated[str, typer.Argument(help="Prompt to send to the configured chat model.")],
    config: ConfigOpt = None,
    set_: SetOpt = None,
    raw: Annotated[bool, typer.Option(help="Plain text instead of structured output.")] = False,
) -> None:
    """Send a prompt straight to the configured LLM (no retrieval) to check the provider works."""
    from rag.providers import build_llm

    cfg = _load(config, set_)
    try:
        llm = build_llm(cfg.llm)
        if raw or not cfg.llm.structured_output:
            console.print(llm.invoke(prompt).text)
        else:
            reply = llm.with_structured_output(LLMReply).invoke(prompt)
            console.print(f"[bold]{reply.title}[/bold]\n{reply.answer}")  # type: ignore[union-attr]
    except RAGError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
