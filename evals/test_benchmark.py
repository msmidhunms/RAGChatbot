"""Offline regression suite: `pytest evals`.

Runs the bundled Nimbus Robotics benchmark through the real ingestion, retrieval and
generation code with deterministic stand-in models, and fails when quality drops below
the gates in evals/baseline.yaml.
"""

from __future__ import annotations

import pytest

from rag_chatbot.evaluation.gates import check_gates, parse_gates

pytestmark = pytest.mark.evals

SETUPS = [(s, t) for s in ("dense", "sparse", "hybrid") for t in ("similarity", "mmr")]


@pytest.mark.parametrize(("strategy", "search_type"), SETUPS, ids=[f"{s}-{t}" for s, t in SETUPS])
def test_quality_gates(run_benchmark, baseline, strategy, search_type):
    report = run_benchmark(strategy, search_type)
    name = f"{strategy}-{search_type}"
    gates = parse_gates([*baseline["common"], *baseline["strategies"][name]])
    results = check_gates([report], gates)
    summary = ", ".join(f"{k}={v:.3f}" for k, v in report.metrics.items())
    print(f"\n{name}: {summary}")
    assert report.errors == 0, [d["error"] for d in report.details if d["error"]]
    failed = [r.message for r in results if not r.passed]
    assert not failed, "quality gates failed:\n" + "\n".join(failed)


def test_every_category_is_evaluated(run_benchmark, benchmark_items):
    report = run_benchmark("hybrid", "similarity")
    assert set(report.counts) == {i.category for i in benchmark_items}
    assert sum(report.counts.values()) == len(benchmark_items)


def test_filtered_questions_only_see_filtered_sources(run_benchmark):
    report = run_benchmark("hybrid", "similarity")
    for detail in report.details:
        if detail["category"] == "filtered":
            assert detail["scores"]["filter_compliance"] == 1.0, detail["id"]
            assert detail["scores"]["hit_rate"] == 1.0, detail["id"]


def test_unanswerable_questions_are_refused(run_benchmark):
    report = run_benchmark("hybrid", "similarity")
    refused = [d["refused"] for d in report.details if d["category"] == "unanswerable"]
    assert refused and all(refused)
