"""Tracing, timing and token accounting."""

from __future__ import annotations

import os
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from rag_chatbot.config.schema import ObservabilityConfig
from rag_chatbot.core.logging import get_logger

log = get_logger("observability")


def setup_tracing(cfg: ObservabilityConfig) -> None:
    """Enable LangSmith tracing for every LangChain/LangGraph call when configured."""
    if cfg.langsmith:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", cfg.project)
        log.info("LangSmith tracing enabled (project %s)", os.environ["LANGSMITH_PROJECT"])


class TokenUsageCallback(BaseCallbackHandler):
    """Sums token usage reported by chat models during one request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        with self._lock:
            self.calls += 1
            for generations in response.generations:
                for gen in generations:
                    usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += int(usage.get("input_tokens", 0))
                    self.output_tokens += int(usage.get("output_tokens", 0))

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


def log_timings(timings: dict[str, float], enabled: bool) -> None:
    if enabled and timings:
        parts = ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in timings.items())
        log.info("timings: %s (total %.0fms)", parts, sum(timings.values()) * 1000)
