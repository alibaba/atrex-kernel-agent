#!/usr/bin/env python3
"""Report latency, correctness, Episode results and per-Shape winners."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments,
    attempt_documents,
    discover_scopes,
    emit,
    evaluation_result,
    finite_positive,
    iter_records,
    main_error,
    observed_at,
    table,
)
from observability.light_latency_status import accepted_best, best_passing_full_eval
from aka.light.workspace.baseline import read_baseline_record


SHAPE_FAILURE = re.compile(r"(?:sid|shape(?:_id)?)=([^: ,]+).*?(?:correctness|compile|performance)", re.I)


def _attempt_result(attempt: dict | None) -> dict | None:
    if not isinstance(attempt, dict):
        return None
    verification = attempt.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    gate = verification.get("gate")
    if gate is None and attempt.get("episode") == 0 and attempt.get("accepted") is True:
        gate = "PASS"
    latency = finite_positive(verification.get("candidate_latency_us")) or finite_positive(
        attempt.get("latency_us_geomean")
    )
    if verification.get("error") or gate == "ERROR":
        state = "infrastructure_error"
    elif attempt.get("accepted") is True:
        state = "pass"
    elif gate in {"FAIL", "REJECT"}:
        state = "rejected"
    else:
        state = str(attempt.get("status") or "unmeasured")
    return {
        "state": state,
        "correctness": "PASS" if gate == "PASS" else None,
        "gate": gate,
        "complete": attempt.get("accepted") is True,
        "latency_us": latency,
        "case_latencies_us": {},
        "expected_case_ids": [],
        "score": verification.get("candidate_performance_score"),
        "metrics": {
            key: value
            for key, value in {
                "improvement_pct": verification.get("improvement_pct"),
                "incumbent_latency_us": verification.get("incumbent_latency_us"),
            }.items()
            if value is not None
        },
    }


def _v0_result(record: dict | None) -> dict | None:
    if not isinstance(record, dict):
        return None
    correctness = record.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    performance = record.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    status = str(correctness.get("status") or "").upper() or None
    latency = finite_positive(performance.get("latency_us_geomean"))
    return {
        "state": "pass" if status == "PASS" else "correctness_failure",
        "correctness": status,
        "gate": "PASS" if status == "PASS" else "FAIL",
        "complete": status == "PASS" and latency is not None,
        "latency_us": latency,
        "case_latencies_us": dict(performance.get("latency_us_by_shape") or {}),
        "expected_case_ids": sorted(str(key) for key in (performance.get("latency_us_by_shape") or {})),
        "score": performance.get("performance_score"),
        "metrics": {
            key: performance[key]
            for key in ("latency_us_arith_mean", "performance_objective")
            if key in performance
        },
    }


def _shape_rows(
    result: dict, record_id: str, kernel_commit_id: str | None, scope: str
) -> list[dict]:
    latencies = result.get("latency_us_by_shape")
    latencies = latencies if isinstance(latencies, dict) else {}
    failed = set()
    unscoped_failure = False
    for failure in result.get("failures") or []:
        if match := SHAPE_FAILURE.search(str(failure)):
            failed.add(match.group(1))
        else:
            unscoped_failure = True
    known = set(str(key) for key in latencies) | failed
    if not known and result.get("all_pass") is False:
        known.add("unknown")
    return [
        {
            "shape_id": shape,
            "latency_us": finite_positive(latencies.get(shape)),
            "correctness": (
                "fail" if shape in failed
                else "pass" if result.get("all_pass") is True
                else "pass" if shape in latencies and not unscoped_failure
                else "unknown"
            ),
            "gateway_record_id": record_id,
            "kernel_commit_id": kernel_commit_id,
            "scope": scope,
        }
        for shape in sorted(known)
    ]


def snapshot(workspace: Path) -> dict:
    scopes = discover_scopes(workspace)
    top = scopes[0]
    accepted = accepted_best(top.workspace, top.campaign_root)
    observed = best_passing_full_eval(top.workspace, top.campaign_root)
    baseline_rows = []
    episode_rows = []
    evaluations = []
    shapes = []
    winners: dict[str, dict] = {}

    for scope in scopes:
        attempts = attempt_documents(scope)
        framework_attempt = next(
            (row for row in attempts if row.get("episode") == 0), None
        )
        accepted_attempts = [row for row in attempts if row.get("accepted") is True]
        incumbent_attempt = accepted_attempts[-1] if accepted_attempts else None
        baseline_rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "v0": _v0_result(
                    read_baseline_record(scope.workspace, migrate_legacy=False)
                ),
                "framework_baseline": {
                    "episode": 0,
                    "result": _attempt_result(framework_attempt),
                    "attempt": framework_attempt,
                } if framework_attempt is not None else None,
                "incumbent": _attempt_result(incumbent_attempt),
            }
        )
        for attempt in attempts:
            episode_rows.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": attempt.get("episode"),
                    "version": attempt.get("version", attempt.get("memory_version")),
                    "kernel_commit_id": (
                        attempt.get("candidate_kernel_commit_id")
                        or attempt.get("candidate_commit")
                    ),
                    "promoted": (
                        attempt.get("accepted") is True
                        and isinstance(attempt.get("episode"), int)
                        and attempt["episode"] > 0
                    ),
                    "status": attempt.get("status"),
                    "result": _attempt_result(attempt),
                }
            )
        for row in iter_records(scope):
            record = row["record"]
            if record.get("operation") != "evaluate":
                continue
            result = evaluation_result(record)
            if result is None:
                continue
            identity = record.get("kernels", {}).get("candidate", {})
            kernel_commit_id = identity.get("kernel_id") if isinstance(identity, dict) else None
            result_row = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "gateway_record_id": row["gateway_record_id"],
                "kernel_commit_id": kernel_commit_id,
                "created_at": record.get("created_at"),
                "cacheable": record.get("cacheable") is True,
                "all_pass": result.get("all_pass"),
                "latency_us_geomean": finite_positive(result.get("latency_us_geomean")),
                "latency_us_arith_mean": finite_positive(result.get("latency_us_arith_mean")),
                "shape_count": len(result.get("latency_us_by_shape") or {}),
                "mode": result.get("mode", record.get("request", {}).get("options", {}).get("evaluation_mode", "full")),
                "failures": [str(value) for value in (result.get("failures") or [])],
            }
            evaluations.append(result_row)
            for shape in _shape_rows(
                result, row["gateway_record_id"], kernel_commit_id, scope.label
            ):
                shapes.append(shape)
                latency = shape["latency_us"]
                if shape["correctness"] != "pass" or latency is None:
                    continue
                previous = winners.get(shape["shape_id"])
                if previous is None or latency < previous["latency_us"]:
                    winners[shape["shape_id"]] = shape

    return {
        "schema_version": 2,
        "observed_at": observed_at(),
        "campaign_workspace": str(top.workspace),
        "accepted_best": accepted,
        "best_passing_full_eval": observed,
        "baselines": baseline_rows,
        "episodes": sorted(episode_rows, key=lambda row: (str(row["scope"]), row["version"])),
        "evaluations": sorted(evaluations, key=lambda row: (row.get("created_at") or "", row["gateway_record_id"])),
        "shape_results": shapes,
        "best_by_shape": {key: winners[key] for key in sorted(winners)},
    }


def _latency(value):
    return None if value is None else round(value, 6)


def render(value: dict) -> str:
    lines = ["AKA Light 性能与正确性", ""]
    headline = []
    for name in ("accepted_best", "best_passing_full_eval"):
        row = value[name]
        headline.append([name, _latency(row.get("latency_us_geomean")) if row else None,
                         (row or {}).get("kernel_commit_id"),
                         (row or {}).get("gateway_record_id"),
                         (row or {}).get("verification_evidence") or (
                             "full_eval_record" if (row or {}).get("gateway_record_id") else None
                         )])
    lines.append(table(["Metric", "GeoMean(us)", "Kernel/Commit", "Record", "Evidence"], headline))
    lines += ["", "Episodes"]
    lines.append(table(
        ["Scope", "Episode", "Version", "Promoted", "State", "Latency(us)"],
        [[row["scope"], row["episode"], row["version"], row["promoted"],
          (row["result"] or {}).get("state"), _latency((row["result"] or {}).get("latency_us"))]
         for row in value["episodes"]],
    ))
    lines += ["", "Per-Shape best"]
    lines.append(table(
        ["Shape", "Latency(us)", "Kernel", "Record"],
        [[shape, _latency(row["latency_us"]), row["kernel_commit_id"], row["gateway_record_id"]]
         for shape, row in value["best_by_shape"].items()],
    ))
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-performance", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
