"""Write evaluation reports as JSON (complete) and Markdown (for people)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.evaluation.gates import GateResult
from rag_chatbot.evaluation.metrics import METRICS
from rag_chatbot.evaluation.runner import EvalReport

WORST_ITEMS = 10


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.3f}"


def item_badness(detail: dict[str, Any]) -> float:
    """Higher means worse: errors first, then the lowest average score."""
    if detail.get("error"):
        return 2.0
    scores = detail.get("scores") or {}
    values = []
    for name, value in scores.items():
        spec = METRICS.get(name)
        if spec is not None:
            values.append(value if spec.higher_is_better else 1.0 - value)
    return 1.0 - (sum(values) / len(values)) if values else 0.0


def worst_items(report: EvalReport, n: int = WORST_ITEMS) -> list[dict[str, Any]]:
    ranked = sorted(report.details, key=item_badness, reverse=True)
    return [d for d in ranked[:n] if item_badness(d) > 0]


def render_markdown(reports: Sequence[EvalReport], gates: Sequence[GateResult] = ()) -> str:
    if not reports:
        return "# Evaluation report\n\nNo reports.\n"
    lines = [f"# Evaluation report ({datetime.now():%Y-%m-%d %H:%M})", ""]
    first = reports[0]
    lines += [f"Questions: {first.items} - configs: {', '.join(r.name for r in reports)}", ""]

    # overall comparison
    metrics = [m for m in first.selected_metrics if any(m in r.metrics for r in reports)]
    lines += ["## Metrics", "", "| metric | " + " | ".join(r.name for r in reports) + " |"]
    lines.append("|---|" + "---|" * len(reports))
    for m in metrics:
        arrow = "" if METRICS[m].higher_is_better else " (lower is better)"
        lines.append(f"| {m}{arrow} | " + " | ".join(_fmt(r.metrics.get(m)) for r in reports) + " |")
    lines.append("")

    # per category, one table per report
    for r in reports:
        cats = list(r.by_category)
        if not cats:
            continue
        lines += [
            f"## By category - {r.name}",
            "",
            "| metric | " + " | ".join(f"{c} ({r.counts.get(c, 0)})" for c in cats) + " |",
        ]
        lines.append("|---|" + "---|" * len(cats))
        for m in metrics:
            if any(m in r.by_category[c] for c in cats):
                lines.append(f"| {m} | " + " | ".join(_fmt(r.by_category[c].get(m)) for c in cats) + " |")
        lines.append("")

    # latency and cost
    lines += ["## Latency and cost", "", "| | " + " | ".join(r.name for r in reports) + " |"]
    lines.append("|---|" + "---|" * len(reports))
    for key in ("latency_mean", "latency_p50", "latency_p95", "latency_max"):
        lines.append(f"| {key} (s) | " + " | ".join(_fmt(r.latency.get(key)) for r in reports) + " |")
    for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens"):
        lines.append(f"| {key} | " + " | ".join(str(r.usage.get(key, "-")) for r in reports) + " |")
    lines.append("| errors | " + " | ".join(str(r.errors) for r in reports) + " |")
    lines.append("| seconds | " + " | ".join(str(r.seconds) for r in reports) + " |")
    lines.append("")

    if gates:
        failed = [g for g in gates if not g.passed]
        lines += [f"## Quality gates - {'all passed' if not failed else f'{len(failed)} failed'}", ""]
        lines += [f"- {'✅' if g.passed else '❌'} {g.message}" for g in gates]
        lines.append("")

    for r in reports:
        worst = worst_items(r)
        if not worst:
            continue
        lines += [f"## Weakest questions - {r.name}", ""]
        for d in worst:
            lines.append(f"### {d['id']} ({d['category']})")
            if d.get("history"):
                lines.append(f"- history: {' / '.join(d['history'])}")
            lines.append(f"- question: {d['question']}")
            if d.get("ground_truth"):
                lines.append(f"- expected: {d['ground_truth']}")
            if d.get("answer") is not None:
                lines.append(f"- answer: {d['answer'].strip()}")
            if d.get("error"):
                lines.append(f"- error: `{d['error']}`")
            if d.get("sources"):
                names = [Path(s).name for s in d["sources"][:5]]
                lines.append(f"- retrieved: {', '.join(names)}")
            if d.get("scores"):
                lines.append("- scores: " + ", ".join(f"{k}={v:.2f}" for k, v in d["scores"].items()))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    reports: Sequence[EvalReport], out_dir: str | Path, gates: Sequence[GateResult] = ()
) -> tuple[Path, Path]:
    """Write ``eval-<timestamp>.json`` and ``.md``; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    payload = {"reports": [r.to_dict() for r in reports], "gates": [asdict(g) for g in gates]}
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = stem.with_suffix(".md")
    md_path.write_text(render_markdown(reports, gates), encoding="utf-8")
    return json_path, md_path


def load_reports(path: str | Path) -> tuple[list[EvalReport], list[GateResult]]:
    """Read a JSON report written by ``write_reports``."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"report not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        reports = [EvalReport(**r) for r in data["reports"]]
        gates = [GateResult(**g) for g in data.get("gates", [])]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigError(f"{p} is not an evaluation report: {exc}") from None
    return reports, gates
