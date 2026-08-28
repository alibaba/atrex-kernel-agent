#!/usr/bin/env python3
"""Measure the incremental CPU cost of Wiki attribution normalization.

The benchmark intentionally excludes journal file I/O, timestamps, and live-memory
sync so the reported curve isolates the work added by the attribution contract.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from long_horizon import journal  # noqa: E402


QUERY_ID = "wiki-query-0123456789abcdef0123456789abcdef"


def experiment(status: str, rows: int) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": "benchmark",
        "wiki_usage_status": status,
        "evaluation": {
            "correctness": "pass",
            "performance": "improved",
            "latency_us": 12.5,
            "kernel_hash": "abc123",
        },
    }
    if status != "not_queried":
        value["wiki_query_ids"] = [QUERY_ID]
    if rows:
        value["wiki_usage"] = [{
            "query_id": QUERY_ID,
            "wiki_id": f"gpu_wiki::benchmark.record-{index}",
            "disposition": "applied",
            "use": "benchmark",
            "evidence": "benchmark",
        } for index in range(rows)]
    return value


def before_contract(template: dict[str, Any]) -> int:
    entry = dict(template)
    return len(entry)


def after_contract(template: dict[str, Any]) -> int:
    entry = dict(template)
    journal.normalize_wiki_attribution(entry)
    evaluation, errors = journal.normalize_experiment_evaluation(entry["evaluation"])
    entry["evaluation"] = evaluation
    if errors:
        entry["evaluation_errors"] = errors
    return len(entry) + len(entry.get("wiki_usage", []))


def sample(function: Callable[[dict[str, Any]], int], template: dict[str, Any],
           iterations: int) -> float:
    checksum = 0
    started = time.perf_counter_ns()
    for _ in range(iterations):
        checksum += function(template)
    elapsed = time.perf_counter_ns() - started
    if checksum <= 0:
        raise RuntimeError("invalid benchmark checksum")
    return elapsed / iterations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")

    scenarios = [
        ("not_queried", 0),
        ("no_material_use", 0),
        ("declared", 1),
        ("declared", 4),
        ("declared", 16),
    ]
    results = []
    for status, rows in scenarios:
        template = experiment(status, rows)
        for function in (before_contract, after_contract):
            sample(function, template, min(args.iterations, 1_000))
        before = []
        after = []
        for _ in range(args.repeats):
            before.append(sample(before_contract, template, args.iterations))
            after.append(sample(after_contract, template, args.iterations))
        overhead = [after[index] - before[index] for index in range(args.repeats)]
        results.append({
            "status": status,
            "wiki_usage_rows": rows,
            "raw_ns_per_op": {"before": before, "after": after, "overhead": overhead},
            "median_ns_per_op": {
                "before": statistics.median(before),
                "after": statistics.median(after),
                "overhead": statistics.median(overhead),
            },
        })

    print(json.dumps({
        "scope": "CPU normalization only; journal I/O and live-memory sync excluded",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
