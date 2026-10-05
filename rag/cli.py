"""Command line interface (``rag`` or ``python main.py``).

Every command takes ``--config/-c`` and repeatable ``--set section.key=value``.
Exit codes: 0 ok, 1 runtime error, 2 invalid configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from rag.config import RAGConfig, dump_config, load_config, missing_env_vars
from rag.config.loader import parse_value
from rag.core.exceptions import ConfigError, RAGError
from rag.core.logging import configure_logging

app = typer.Typer(help="Configurable RAG chatbot.", no_args_is_help=True, pretty_exceptions_enable=False)
config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
store_app = typer.Typer(help="Inspect and maintain the vector store.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(store_app, name="store")

console = Console()
err_console = Console(stderr=True)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (default: $RAG_CONFIG or built-in defaults)."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a setting, e.g. --set retrieval.k=10. Repeatable."),
]
FilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--filter", "-f", help="Metadata filter key=value, e.g. -f source=docs/a.pdf -f 'page={$gte: 3}'."
    ),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show timings and token usage.")]


# ------------------------------------------------------------------ helpers
def _load(config: Path | None, overrides: list[str] | None, verbose: bool = False) -> RAGConfig:
    try:
        cfg = load_config(config, overrides or [])
    except ConfigError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    configure_logging("INFO" if verbose and cfg.app.log_level not in ("DEBUG", "INFO") else cfg.app.log_level)
    return cfg


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except ConfigError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    except RAGError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


def _pipeline(cfg: RAGConfig) -> Any:
    from rag.pipeline import RAGPipeline

    return RAGPipeline(cfg)


def _pairs(items: list[str] | None, what: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            err_console.print(f"[red]{what} must look like key=value, got {item!r}[/red]")
            raise typer.Exit(2)
        out[key.strip()] = parse_value(value.strip())
    return out


def _print_answer(result: Any, show_sources: bool, verbose: bool, body: bool = True) -> None:
    answer = result.answer
    if body:
        console.print(f"[bold]{answer.title}[/bold]")
        console.print(answer.answer, markup=False, highlight=False)
    if answer.citations:
        console.print("\n[dim]Sources:[/dim]")
        for c in answer.citations:
            where = f", page {c.page}" if c.page is not None else ""
            section = f" ({c.section})" if c.section else ""
            console.print(f"[dim]  [{c.id}] {c.source}{where}{section}[/dim]", markup=True, highlight=False)
    if show_sources:
        _print_chunks(result.chunks, "Retrieved context")
    if verbose:
        _print_diagnostics(result)


def _print_diagnostics(result: Any) -> None:
    timings = ", ".join(f"{k} {v * 1000:.0f}ms" for k, v in result.timings.items())
    console.print(
        f"[dim]confidence: {result.answer.confidence}"
        + (f", grounded: {result.answer.grounded}" if result.answer.grounded is not None else "")
        + "[/dim]"
    )
    if result.standalone_question and result.standalone_question != result.question:
        console.print(f"[dim]standalone question: {result.standalone_question}[/dim]", markup=True)
    if len(result.queries) > 1 or (result.queries and result.queries[0] != result.standalone_question):
        console.print(f"[dim]queries: {result.queries}[/dim]", markup=False)
    console.print(f"[dim]timings: {timings}[/dim]")
    u = result.usage
    console.print(
        f"[dim]llm calls: {u.get('llm_calls', 0)}, tokens: {u.get('total_tokens', 0)} "
        f"(in {u.get('input_tokens', 0)}, out {u.get('output_tokens', 0)})[/dim]"
    )


def _print_chunks(chunks: list[Any], title: str) -> None:
    table = Table(title=title, show_lines=False)
    for col in ("#", "score", "retriever", "source", "page", "text"):
        table.add_column(col, overflow="fold")
    for c in chunks:
        page = c.metadata.get("page")
        snippet = " ".join(c.text.split())[:160]
        table.add_row(
            str(c.rank),
            f"{c.score:.3f}",
            c.source,
            str(c.metadata.get("source", "")),
            "" if page is None else str(page),
            snippet,
        )
    console.print(table)


# ------------------------------------------------------------------- config
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


# ------------------------------------------------------------------- ingest
@app.command("ingest")
def ingest(
    paths: Annotated[
        list[str] | None, typer.Argument(help="Files, directories or URLs (default: ingestion.sources).")
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", "-t", help="Metadata tag key=value added to every chunk.")
    ] = None,
    reset: Annotated[bool, typer.Option(help="Delete everything in the collection first.")] = False,
    incremental: Annotated[bool, typer.Option(help="Skip files whose content has not changed.")] = True,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Load, chunk, embed and index documents."""
    cfg = _load(config, set_)
    with _errors():
        report = _pipeline(cfg).ingest(
            paths or None, tags=_pairs(tag, "--tag"), reset=reset, incremental=incremental
        )
    table = Table(title="Ingestion report", show_header=False)
    for key, value in [
        ("sources found", report.files_seen),
        ("ingested", report.files_ingested),
        ("unchanged (skipped)", report.files_skipped),
        ("failed", len(report.errors)),
        ("chunks added", report.chunks_added),
        ("chunks replaced", report.chunks_deleted),
        ("seconds", report.seconds),
    ]:
        table.add_row(key, str(value))
    console.print(table)
    for source, error in report.errors.items():
        err_console.print(f"[red]x {source}[/red]: {error}", markup=True, highlight=False)
    if report.errors and not report.files_ingested and not report.files_skipped:
        raise typer.Exit(1)


# -------------------------------------------------------------------- query
@app.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question to answer from the indexed documents.")],
    filter_: FilterOpt = None,
    show_sources: Annotated[bool, typer.Option("--show-sources", help="Print the retrieved chunks.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Answer one question (no conversation memory)."""
    cfg = _load(config, set_, verbose)
    with _errors():
        result = _pipeline(cfg).query(question, _pairs(filter_, "--filter") or None)
    if as_json:
        console.print_json(json.dumps(result.to_dict(), default=str))
    else:
        _print_answer(result, show_sources, verbose)


@app.command("retrieve")
def retrieve(
    question: Annotated[str, typer.Argument(help="Search query.")],
    filter_: FilterOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Show what retrieval returns for a query (no answer generation)."""
    cfg = _load(config, set_)
    with _errors():
        chunks = _pipeline(cfg).retrieve(question, _pairs(filter_, "--filter") or None)
    if as_json:
        rows = [
            {"rank": c.rank, "score": c.score, "retriever": c.source, "text": c.text, "metadata": c.metadata}
            for c in chunks
        ]
        console.print_json(json.dumps(rows, default=str))
    elif chunks:
        _print_chunks(chunks, f"Top {len(chunks)} for: {question}")
    else:
        console.print("[yellow]no results[/yellow]")


# --------------------------------------------------------------------- chat
CHAT_HELP = (
    "/sources  last retrieved chunks\n/history  conversation so far\n/reset    forget this session\n"
    "/config   active settings\n/exit     quit"
)


@app.command("chat")
def chat(
    session: Annotated[
        str | None, typer.Option("--session", help="Session id to resume (default: new).")
    ] = None,
    filter_: FilterOpt = None,
    stream: Annotated[bool, typer.Option(help="Stream answer tokens as they are generated.")] = True,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Interactive chat with conversation memory."""
    cfg = _load(config, set_, verbose)
    pipe = _pipeline(cfg)
    session_id = session or pipe.new_session_id()
    flt = _pairs(filter_, "--filter") or None
    console.print(f"[dim]session {session_id} - type /help for commands, /exit to quit[/dim]")
    last = None
    while True:
        try:
            question = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question:
            continue
        if question.startswith("/"):
            cmd = question.lower()
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                console.print(CHAT_HELP, markup=False)
            elif cmd == "/sources":
                _print_chunks(last.chunks, "Last retrieved context") if last else console.print("nothing yet")
            elif cmd == "/history":
                with _errors():
                    for m in pipe.history(session_id):
                        console.print(f"[bold]{m.type}:[/bold] {m.text}", highlight=False)
            elif cmd == "/reset":
                with _errors():
                    pipe.clear_session(session_id)
                console.print("[dim]session cleared[/dim]")
            elif cmd == "/config":
                c = pipe.cfg
                settings = [
                    f"llm {c.llm.provider}:{c.llm.model}",
                    f"embeddings {c.embeddings.provider}:{c.embeddings.model}",
                    f"store {c.vector_store.type}/{c.vector_store.collection}",
                    f"retrieval {c.retrieval.strategy} k={c.retrieval.k}",
                    f"transform {c.query_transform.type}",
                    f"reranker {c.reranker.type}",
                    f"memory {c.memory.type}",
                ]
                console.print(
                    " | ".join(settings),
                    markup=False,
                )
            else:
                console.print(f"unknown command {question}; /help lists commands", markup=False)
            continue
        console.print("[bold green]bot>[/bold green] ", end="")
        try:
            with _errors():
                if stream:
                    for item in pipe.stream_chat(question, session_id, flt):
                        if isinstance(item, str):
                            console.print(item, end="", markup=False, highlight=False)
                        else:
                            last = item
                    console.print()
                    _print_answer(last, False, verbose, body=False)
                else:
                    last = pipe.chat(question, session_id, flt)
                    _print_answer(last, False, verbose)
        except typer.Exit:
            continue  # error already printed; keep the session alive


# -------------------------------------------------------------------- store
@store_app.command("stats")
def store_stats(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Collection size, embedding model and per-source chunk counts."""
    cfg = _load(config, set_)
    with _errors():
        stats = _pipeline(cfg).stats()
    table = Table(show_header=False, title="Vector store")
    for key in (
        "vector_store",
        "collection",
        "embedding",
        "built_with",
        "dimension",
        "vectors",
        "chunks",
        "sources",
    ):
        table.add_row(key, str(stats[key]))
    console.print(table)
    if stats["per_source"]:
        per = Table("source", "chunks", title="Sources")
        for source, n in sorted(stats["per_source"].items()):
            per.add_row(source, str(n))
        console.print(per)


@store_app.command("reset")
def store_reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Delete every vector, chunk and manifest entry in the collection."""
    cfg = _load(config, set_)
    target = f"{cfg.vector_store.type}/{cfg.vector_store.collection}"
    if not yes and not typer.confirm(f"Delete everything in {target}?"):
        raise typer.Exit(1)
    with _errors():
        _pipeline(cfg).reset_store()
    console.print(f"[green]reset {target}[/green]")


@store_app.command("delete")
def store_delete(
    source: Annotated[
        str, typer.Option("--source", help="Source path or URL as shown by `rag store stats`.")
    ],
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Remove one source's chunks from the index."""
    cfg = _load(config, set_)
    with _errors():
        deleted = _pipeline(cfg).delete_source(source)
    if not deleted:
        err_console.print(f"[yellow]no chunks found for {source}[/yellow]")
        raise typer.Exit(1)
    console.print(f"deleted {deleted} chunks from {source}")


# --------------------------------------------------------------------- eval
@app.command("eval")
def eval_command(
    configs: Annotated[
        list[Path] | None, typer.Option("--config", "-c", help="Config to evaluate; repeat to compare.")
    ] = None,
    dataset: Annotated[Path | None, typer.Option(help="QA JSONL (default: evaluation.dataset).")] = None,
    ingest_dir: Annotated[
        list[str] | None, typer.Option("--ingest", help="Ingest these sources before evaluating.")
    ] = None,
    set_: SetOpt = None,
) -> None:
    """Score retrieval and answers on a QA dataset; compare several configs side by side."""
    from rag.evaluation.dataset import load_dataset
    from rag.evaluation.runner import run_eval, save_reports

    paths: list[Path | None] = list(configs) if configs else [None]
    first = _load(paths[0], set_)
    with _errors():
        items = load_dataset(dataset or first.evaluation.dataset)
        reports = run_eval(paths, items, overrides=set_ or [], ingest=ingest_dir)
        out = save_reports(reports, first.evaluation.output_dir)
    metrics = sorted({m for r in reports for m in r.metrics})
    table = Table("config", *metrics, "errors", "seconds", title=f"Evaluation ({len(items)} questions)")
    for r in reports:
        table.add_row(
            r.name, *[f"{r.metrics.get(m, float('nan')):.3f}" for m in metrics], str(r.errors), str(r.seconds)
        )
    console.print(table)
    console.print(f"[dim]details: {out}[/dim]")


# ---------------------------------------------------------------------- llm
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
    with _errors():
        llm = build_llm(cfg.llm)
        if raw or not cfg.llm.structured_output:
            console.print(llm.invoke(prompt).text, markup=False)
        else:
            reply = llm.with_structured_output(LLMReply).invoke(prompt)
            console.print(f"[bold]{reply.title}[/bold]")  # type: ignore[union-attr]
            console.print(reply.answer, markup=False)  # type: ignore[union-attr]


if __name__ == "__main__":
    app()
