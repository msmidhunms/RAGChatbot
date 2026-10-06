"""Run a RAG configuration over a QA dataset and aggregate the scores.

``Evaluator`` does only the work the selected metrics need:

- retrieval metrics only: ``pipeline.retrieve`` (no LLM unless a query transform is configured)
- answer metrics: ``pipeline.query``, or a fresh chat session for conversational items
- judge metrics: one or two extra LLM calls per answerable question
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_chatbot.config import dump_config
from rag_chatbot.core.logging import get_logger
from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.evaluation.dataset import QAItem
from rag_chatbot.evaluation.metrics import (
    METRICS,
    answer_scores,
    expand,
    is_refusal,
    judge_scores,
    latency_summary,
    retrieval_scores,
    usage_totals,
)
from rag_chatbot.pipeline import RAGPipeline
from rag_chatbot.providers import build_llm

log = get_logger("evaluation")

JUDGE_METRIC_NAMES = [m for m, s in METRICS.items() if s.needs_judge]


@dataclass
class ItemResult:
    id: str
    category: str
    question: str
    history: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    answer: str | None = None
    standalone_question: str | None = None
    refused: bool | None = None
    sources: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    latency: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    name: str
    items: int
    metrics: dict[str, float]
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    seconds: float = 0.0
    selected_metrics: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)

    def value(self, name: str) -> float | None:
        """Look up ``metric``, ``category.metric`` or a latency/usage figure."""
        if "." in name:
            category, metric = name.split(".", 1)
            return self.by_category.get(category, {}).get(metric)
        if name in self.metrics:
            return self.metrics[name]
        if name in self.latency:
            return self.latency[name]
        return float(self.usage[name]) if name in self.usage else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4)


def _aggregate(results: Sequence[ItemResult], metrics: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        values = [r.scores[metric] for r in results if metric in r.scores]
        if values:
            out[metric] = _mean(values)
    return out


class Evaluator:
    def __init__(
        self,
        pipeline: RAGPipeline,
        metrics: Iterable[str],
        *,
        judge_llm: BaseChatModel | None = None,
        k: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = expand(metrics)
        self.wanted = set(self.metrics)
        specs = [METRICS[m] for m in self.metrics]
        self.needs_answer = any(s.needs_answer for s in specs)
        self.needs_judge = any(s.needs_judge for s in specs)
        cfg = pipeline.cfg
        self.k = k or cfg.evaluation.k or cfg.retrieval.k
        self._judge_llm = judge_llm

    @property
    def judge_llm(self) -> BaseChatModel:
        if self._judge_llm is None:
            judge_cfg = self.pipeline.cfg.evaluation.judge_llm
            self._judge_llm = build_llm(judge_cfg) if judge_cfg else self.pipeline.llm
        return self._judge_llm

    # ---------------------------------------------------------------- items
    def _run_item(self, item: QAItem) -> tuple[list[RetrievedChunk], ItemResult]:
        res = ItemResult(
            id=item.id,
            category=item.category,
            question=item.question,
            history=list(item.history),
            ground_truth=item.ground_truth,
        )
        flt = item.filter or None
        if not self.needs_answer:
            # retrieval only: a follow-up question is searched together with its history
            query = " ".join([*item.history, item.question])
            start = time.perf_counter()
            chunks = self.pipeline.retrieve(query, flt)
            res.latency = round(time.perf_counter() - start, 4)
            return chunks, res

        if item.history:
            session = f"eval-{item.id}-{self.pipeline.new_session_id()}"
            try:
                for turn in item.history:
                    self.pipeline.chat(turn, session, flt)
                result = self.pipeline.chat(item.question, session, flt)
            finally:
                self.pipeline.clear_session(session)
        else:
            result = self.pipeline.query(item.question, flt)
        res.answer = result.answer.answer
        res.standalone_question = result.standalone_question
        res.refused = is_refusal(result.answer)
        res.latency = round(sum(result.timings.values()), 4)
        res.usage = dict(result.usage)
        res.scores.update(answer_scores(result.answer, result.chunks, item))
        if self.needs_judge and item.answerable:
            res.scores.update(
                judge_scores(
                    self.judge_llm,
                    item,
                    result.standalone_question or item.question,
                    result.answer.answer,
                    result.chunks,
                    self.wanted,
                    self.pipeline.cfg.generation.max_context_tokens,
                )
            )
        return result.chunks, res

    def evaluate_item(self, item: QAItem) -> ItemResult:
        try:
            chunks, res = self._run_item(item)
            res.sources = [str(c.metadata.get("source", "")) for c in chunks]
            res.scores.update(retrieval_scores(chunks, item, self.k))
            res.scores = {m: round(v, 4) for m, v in res.scores.items() if m in self.wanted}
            return res
        except Exception as exc:  # keep evaluating the rest
            log.warning("evaluation failed for %s (%r): %s", item.id, item.question, exc)
            return ItemResult(
                id=item.id,
                category=item.category,
                question=item.question,
                history=list(item.history),
                ground_truth=item.ground_truth,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ run
    def run(self, items: Sequence[QAItem], name: str = "config") -> EvalReport:
        start = time.perf_counter()
        results = [self.evaluate_item(item) for item in items]
        categories = sorted({r.category for r in results})
        return EvalReport(
            name=name,
            items=len(results),
            metrics=_aggregate(results, self.metrics),
            by_category={
                c: _aggregate([r for r in results if r.category == c], self.metrics) for c in categories
            },
            counts={c: sum(r.category == c for r in results) for c in categories},
            latency=latency_summary([r.latency for r in results if r.latency is not None]),
            usage=usage_totals([r.usage for r in results if r.usage]),
            errors=sum(r.error is not None for r in results),
            seconds=round(time.perf_counter() - start, 2),
            selected_metrics=list(self.metrics),
            config=dump_config(self.pipeline.cfg),
            details=[asdict(r) for r in results],
        )


def resolve_metrics(configured: Sequence[str], override: Sequence[str] | None, judge: bool) -> list[str]:
    metrics = list(override) if override else list(configured)
    if judge:
        metrics += [m for m in JUDGE_METRIC_NAMES if m not in metrics]
    return expand(metrics)


def run_eval(
    configs: Sequence[str | Path | None],
    items: Sequence[QAItem],
    *,
    overrides: Sequence[str] = (),
    ingest: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    judge: bool = False,
    pipeline_factory: Callable[[str | Path | None, Sequence[str]], RAGPipeline] | None = None,
) -> list[EvalReport]:
    """Evaluate each config (optionally ingesting a corpus first) on the same items."""
    factory = pipeline_factory or (lambda path, ov: RAGPipeline.from_config(path, ov))
    reports = []
    for path in configs or [None]:
        pipeline = factory(path, overrides)
        if ingest:
            report = pipeline.ingest(ingest)
            log.info("ingested %d chunks for %s", report.chunks_added, path or "defaults")
        selected = resolve_metrics(pipeline.cfg.evaluation.metrics, metrics, judge)
        name = Path(path).stem if path else "defaults"
        reports.append(Evaluator(pipeline, selected).run(items, name=name))
    return reports
