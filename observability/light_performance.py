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
    discover_scopes,
    emit,
    evaluation_result,
    finite_positive,
    iter_records,
    load_snapshot,
    main_error,
    observed_at,
    read_json,
    table,
)
from observability.light_latency_status import accepted_best, best_passing_full_eval


SHAPE_FAILURE = re.compile(r"(?:sid|shape(?:_id)?)=([^: ,]+).*?(?:correctness|compile|performance)", re.I)


def _result(value) -> dict | None:
    if value is None:
        return None
    return {
        "state": value.state,
        "correctness": value.correctness,
        "gate": value.gate,
        "complete": value.complete,
        "latency_us": value.latency_us,
        "case_latencies_us": dict(value.case_latencies_us),
        "expected_case_ids": list(value.expected_case_ids),
        "score": value.score,
        "metrics": dict(value.metrics),
    }


def _shape_rows(result: dict, record_id: str, kernel_id: str | None, scope: str) -> list[dict]:
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
            "kernel_id": kernel_id,
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
        public = load_snapshot(scope)
        framework_marker = read_json(scope.workspace / "framework_baseline.json", default=None)
        framework_version = None
        if isinstance(framework_marker, dict):
            raw_version = framework_marker.get("version")
            if isinstance(raw_version, str) and raw_version.startswith("v") and raw_version[1:].isdigit():
                framework_version = int(raw_version[1:])
        framework_candidate = next(
            (candidate for candidate in public.candidates if candidate.version == framework_version),
            None,
        )
        baseline_rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "bootstrap": _result(public.baseline.result) if public.baseline else None,
                "framework_baseline": {
                    "marker": framework_marker,
                    "result": _result(framework_candidate.result) if framework_candidate else None,
                } if framework_marker is not None else None,
                "incumbent": _result(public.incumbent.result) if public.incumbent else None,
            }
        )
        by_episode = {episode.episode: episode for episode in public.episodes}
        for candidate in public.candidates:
            episode = by_episode.get(candidate.episode)
            episode_rows.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": candidate.episode,
                    "version": candidate.version,
                    "promoted": candidate.promoted,
                    "status": episode.status if episode else None,
                    "result": _result(candidate.result),
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
            kernel_id = identity.get("kernel_id") if isinstance(identity, dict) else None
            result_row = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "gateway_record_id": row["gateway_record_id"],
                "kernel_id": kernel_id,
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
            for shape in _shape_rows(result, row["gateway_record_id"], kernel_id, scope.label):
                shapes.append(shape)
                latency = shape["latency_us"]
                if shape["correctness"] != "pass" or latency is None:
                    continue
                previous = winners.get(shape["shape_id"])
                if previous is None or latency < previous["latency_us"]:
                    winners[shape["shape_id"]] = shape

    return {
        "schema_version": 1,
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
                         (row or {}).get("git_commit") or (row or {}).get("kernel_id"),
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
        [[shape, _latency(row["latency_us"]), row["kernel_id"], row["gateway_record_id"]]
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
