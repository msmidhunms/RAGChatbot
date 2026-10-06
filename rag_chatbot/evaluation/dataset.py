"""Evaluation dataset: JSONL with question, ground_truth and expected_sources."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from rag_chatbot.core.exceptions import ConfigError


class QAItem(BaseModel):
    question: str
    ground_truth: str | None = None
    expected_sources: list[str] = Field(default_factory=list)


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
    return items
