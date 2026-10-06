"""Quality gates: fail an evaluation when a metric crosses a threshold.

A gate is ``metric OP value`` with OP one of ``>= <= > < ==`` (a bare ``=``
means ``>=``). ``metric`` can be any overall metric, ``category.metric`` for a
per-category score, or a latency/usage figure such as ``latency_p95``::

    hit_rate>=0.8   unanswerable.refusal_accuracy>=0.9   latency_p95<=5
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.evaluation.runner import EvalReport

_GATE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(>=|<=|==|>|<|=)\s*(-?\d+(?:\.\d+)?)\s*$")
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class Gate:
    target: str
    op: str
    threshold: float

    def __str__(self) -> str:
        return f"{self.target}{self.op}{self.threshold:g}"


@dataclass(frozen=True)
class GateResult:
    report: str
    gate: str
    actual: float | None
    passed: bool

    @property
    def message(self) -> str:
        if self.actual is None:
            return f"{self.report}: {self.gate} - metric was not measured"
        status = "ok" if self.passed else "FAILED"
        return f"{self.report}: {self.gate} - actual {self.actual:.4g} [{status}]"


def parse_gate(expression: str) -> Gate:
    match = _GATE.match(expression)
    if not match:
        raise ConfigError(f"invalid gate {expression!r}; expected e.g. 'hit_rate>=0.8' or 'latency_p95<=5'")
    target, op, value = match.groups()
    return Gate(target, ">=" if op == "=" else op, float(value))


def parse_gates(expressions: Iterable[str]) -> list[Gate]:
    gates: list[Gate] = []
    for expr in expressions:
        gate = parse_gate(expr)
        if all(str(g) != str(gate) for g in gates):
            gates.append(gate)
    return gates


def check_gates(reports: Sequence[EvalReport], gates: Sequence[Gate]) -> list[GateResult]:
    """Evaluate every gate against every report; unmeasured metrics fail."""
    results = []
    for report in reports:
        for gate in gates:
            actual = report.value(gate.target)
            passed = actual is not None and _OPS[gate.op](actual, gate.threshold)
            results.append(GateResult(report.name, str(gate), actual, passed))
    return results
