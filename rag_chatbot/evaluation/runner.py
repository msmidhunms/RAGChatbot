"""Run one or more configurations over a QA dataset and compare them."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_chatbot.core.logging import get_logger
from rag_chatbot.evaluation.dataset import QAItem
from rag_chatbot.evaluation.metrics import JUDGE_METRICS, RETRIEVAL_METRICS, judge, retrieval_scores
from rag_chatbot.pipeline import RAGPipeline
from rag_chatbot.providers import build_llm

log = get_logger("evaluation")


@dataclass
class EvalReport:
    name: str
    items: int
    metrics: dict[str, float]
    details: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    errors: int = 0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else float("nan")


def evaluate(
    pipeline: RAGPipeline,
    items: Sequence[QAItem],
    metrics: Iterable[str],
    *,
    name: str = "config",
    judge_llm: BaseChatModel | None = None,
) -> EvalReport:
    wanted = list(metrics)
    needs_answer = any(m in JUDGE_METRICS for m in wanted)
    if needs_answer and judge_llm is None:
        judge_cfg = pipeline.cfg.evaluation.judge_llm
        judge_llm = build_llm(judge_cfg) if judge_cfg else pipeline.llm

    start = time.perf_counter()
    collected: dict[str, list[float]] = {m: [] for m in wanted}
    details = []
    errors = 0
    for item in items:
        row: dict[str, Any] = {"question": item.question}
        try:
            if needs_answer:
                result = pipeline.query(item.question)
                chunks, row["answer"] = result.chunks, result.answer.answer
            else:
                chunks = pipeline.retrieve(item.question)
            row["sources"] = [c.metadata.get("source") for c in chunks]
            scores: dict[str, float] = {}
            if item.expected_sources and any(m in RETRIEVAL_METRICS for m in wanted):
                scores.update(retrieval_scores(chunks, item.expected_sources))
            if needs_answer:
                assert judge_llm is not None
                scores.update(
                    judge(
                        judge_llm,
                        item.question,
                        row["answer"],
                        chunks,
                        item.ground_truth,
                        pipeline.cfg.generation.max_context_tokens,
                    )
                )
            for metric in wanted:
                if metric in scores:
                    collected[metric].append(scores[metric])
                    row[metric] = scores[metric]
        except Exception as exc:  # keep evaluating the rest
            errors += 1
            row["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("evaluation failed for %r: %s", item.question, exc)
        details.append(row)
    return EvalReport(
        name=name,
        items=len(items),
        metrics={m: _mean(v) for m, v in collected.items()},
        details=details,
        seconds=round(time.perf_counter() - start, 2),
        errors=errors,
    )


def run_eval(
    configs: Sequence[str | Path | None],
    items: Sequence[QAItem],
    *,
    overrides: Sequence[str] = (),
    ingest: Sequence[str] | None = None,
    pipeline_factory: Callable[[str | Path | None, Sequence[str]], RAGPipeline] | None = None,
) -> list[EvalReport]:
    factory = pipeline_factory or (lambda path, ov: RAGPipeline.from_config(path, ov))
    reports = []
    for path in configs or [None]:
        pipeline = factory(path, overrides)
        if ingest:
            report = pipeline.ingest(ingest)
            log.info("ingested %d chunks for %s", report.chunks_added, path or "defaults")
        name = Path(path).stem if path else "defaults"
        reports.append(evaluate(pipeline, items, pipeline.cfg.evaluation.metrics, name=name))
    return reports


def save_reports(reports: Sequence[EvalReport], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps([asdict(r) for r in reports], indent=2, default=str), encoding="utf-8")
    return path
