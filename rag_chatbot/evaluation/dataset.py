"""Evaluation dataset: one JSON object per line (JSONL).

Only ``question`` is required, so minimal files still load. A full item::

    {"id": "pricing-03", "category": "multi_hop",
     "question": "What does a Drone X1 with extended warranty cost?",
     "ground_truth": "5,290 EUR (4,900 + 390).",
     "expected_sources": ["pricing.csv", "warranty.txt"],
     "evidence": ["4900", "390 EUR"],
     "answer_must_contain": ["5,290"]}

``evidence`` lists short verbatim snippets that the retrieved context must
contain; it gives chunk-level relevance without an LLM judge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from rag_chatbot.core.exceptions import ConfigError

Category = Literal["factual", "multi_hop", "unanswerable", "conversational", "filtered"]
CATEGORIES: tuple[str, ...] = ("factual", "multi_hop", "unanswerable", "conversational", "filtered")


class QAItem(BaseModel):
    question: str = Field(min_length=1)
    id: str = ""
    category: Category = "factual"
    ground_truth: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    answer_must_contain: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)  # earlier user turns (conversational)
    filter: dict[str, Any] = Field(default_factory=dict)  # metadata filter for the query
    generated: bool = False

    @model_validator(mode="after")
    def _check(self) -> QAItem:
        if not self.id:
            self.id = "q-" + hashlib.sha1(self.question.encode()).hexdigest()[:8]
        if self.category == "unanswerable" and (self.expected_sources or self.evidence):
            raise ValueError("unanswerable items must not have expected_sources or evidence")
        if self.category == "conversational" and not self.history:
            raise ValueError("conversational items need at least one earlier turn in 'history'")
        if self.category == "filtered" and not self.filter:
            raise ValueError("filtered items need a 'filter'")
        return self

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_dataset(path: str | Path) -> list[QAItem]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"evaluation dataset not found: {p}")
    items = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            items.append(QAItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigError(f"{p}:{n}: invalid dataset line: {exc}") from None
    if not items:
        raise ConfigError(f"evaluation dataset {p} is empty")
    ids = [i.id for i in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"{p}: duplicate item ids: {', '.join(duplicates)}")
    return items


def save_dataset(items: Iterable[QAItem], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(i.model_dump(exclude_defaults=True), ensure_ascii=False) for i in items]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def filter_items(
    items: Sequence[QAItem], categories: Iterable[str] | None = None, limit: int | None = None
) -> list[QAItem]:
    wanted = set(categories or [])
    unknown = wanted - set(CATEGORIES)
    if unknown:
        raise ConfigError(f"unknown categories {sorted(unknown)}; use {', '.join(CATEGORIES)}")
    selected = [i for i in items if not wanted or i.category in wanted]
    return selected[:limit] if limit else selected
