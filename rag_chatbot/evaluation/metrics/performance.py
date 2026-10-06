"""Latency and cost aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; NaN for an empty sequence."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {}
    return {
        "latency_mean": round(sum(latencies) / len(latencies), 4),
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p95": round(percentile(latencies, 95), 4),
        "latency_max": round(max(latencies), 4),
    }


def usage_totals(usages: Sequence[Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals
