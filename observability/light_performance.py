#!/usr/bin/env python3
"""Report V0, Evaluate observations, accepted submissions and Shape winners."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments, attempt_documents, discover_scopes, emit, evaluation_result,
    finite_positive, iter_records, main_error, observed_at, table,
)
from observability.submissions import campaign_submission_snapshot
from aka.light.workspace.baseline import read_baseline_record

SHAPE_FAILURE = re.compile(r"(?:sid|shape(?:_id)?)=([^: ,]+).*?(?:correctness|compile|performance)", re.I)


def _v0_result(record: dict | None) -> dict | None:
    if not isinstance(record, dict):
        return None
    correctness = record.get("correctness") if isinstance(record.get("correctness"), dict) else {}
    performance = record.get("performance") if isinstance(record.get("performance"), dict) else {}
    status = str(correctness.get("status") or "").upper() or None
    return {"correctness": status, "complete": status == "PASS" and finite_positive(performance.get("latency_us_geomean")) is not None,
            "latency_us_geomean": finite_positive(performance.get("latency_us_geomean")),
            "latency_us_arith_mean": finite_positive(performance.get("latency_us_arith_mean")),
            "latency_us_by_shape": dict(performance.get("latency_us_by_shape") or {}),
            "performance_objective": performance.get("performance_objective")}


def _shape_rows(result: dict, record_id: str, commit_id: str | None, scope: str) -> list[dict]:
    latencies = result.get("latency_us_by_shape") if isinstance(result.get("latency_us_by_shape"), dict) else {}
    failed, unscoped = set(), False
    for failure in result.get("failures") or []:
        match = SHAPE_FAILURE.search(str(failure))
        failed.add(match.group(1)) if match else None
        unscoped = unscoped or match is None
    known = set(map(str, latencies)) | failed
    if not known and result.get("all_pass") is False:
        known.add("unknown")
    return [{"shape_id": shape, "latency_us": finite_positive(latencies.get(shape)),
             "correctness": "fail" if shape in failed else "pass" if result.get("all_pass") is True else "pass" if shape in latencies and not unscoped else "unknown",
             "gateway_record_id": record_id, "kernel_commit_id": commit_id, "scope": scope}
            for shape in sorted(known)]


def snapshot(workspace: Path) -> dict:
    scopes = discover_scopes(workspace)
    top = scopes[0]
    submission = campaign_submission_snapshot(top.workspace)
    episodes, evaluations, shapes = [], [], []
    observed_winners, accepted_winners = {}, {}
    accepted_commits = {row["kernel_commit_id"] for row in submission["accepted_submissions"]}
    baselines = []
    for scope in scopes:
        baselines.append({"scope": scope.label, "trajectory": scope.trajectory,
                          "v0": _v0_result(read_baseline_record(scope.workspace, migrate_legacy=False))})
        for attempt in attempt_documents(scope):
            episodes.append({"scope": scope.label, "trajectory": scope.trajectory,
                             "episode": attempt.get("episode"), "status": attempt.get("status"),
                             "session_id": attempt.get("session_id"),
                             "agent_head_commit_id": attempt.get("candidate_kernel_commit_id") or attempt.get("candidate_commit"),
                             "summary": attempt.get("summary"), "violation": attempt.get("violation")})
        for item in iter_records(scope):
            record = item["record"]
            if record.get("operation") != "evaluate" or (result := evaluation_result(record)) is None:
                continue
            identity = record.get("kernels", {}).get("candidate")
            commit_id = identity.get("kernel_id") if isinstance(identity, dict) else None
            evaluations.append({"scope": scope.label, "trajectory": scope.trajectory,
                                "gateway_record_id": item["gateway_record_id"], "kernel_commit_id": commit_id,
                                "created_at": record.get("created_at"), "cacheable": record.get("cacheable") is True,
                                "all_pass": result.get("all_pass"),
                                "latency_us_geomean": finite_positive(result.get("latency_us_geomean")),
                                "latency_us_arith_mean": finite_positive(result.get("latency_us_arith_mean")),
                                "shape_count": len(result.get("latency_us_by_shape") or {}),
                                "mode": result.get("mode", "full"), "failures": list(map(str, result.get("failures") or [])),
                                "accepted_submission": commit_id in accepted_commits})
            for shape in _shape_rows(result, item["gateway_record_id"], commit_id, scope.label):
                shapes.append(shape)
                latency = shape["latency_us"]
                if shape["correctness"] != "pass" or latency is None:
                    continue
                previous = observed_winners.get(shape["shape_id"])
                if previous is None or latency < previous["latency_us"]:
                    observed_winners[shape["shape_id"]] = shape
                if commit_id in accepted_commits:
                    previous = accepted_winners.get(shape["shape_id"])
                    if previous is None or latency < previous["latency_us"]:
                        accepted_winners[shape["shape_id"]] = shape
    accepted_best = min(submission["lineage_best"], key=lambda row: row["latency_us_geomean"], default=None)
    observed_best = min(submission["full_evaluations"], key=lambda row: row["latency_us_geomean"], default=None)
    return {"schema_version": 3, "observed_at": observed_at(), "campaign_workspace": str(top.workspace),
            "accepted_best": accepted_best, "best_passing_full_eval": observed_best,
            "lineage_best": submission["lineage_best"], "global_best": submission["global_best"],
            "submissions": submission["submissions"], "baselines": baselines,
            "episodes": sorted(episodes, key=lambda row: (str(row["scope"]), row["episode"] if isinstance(row["episode"], int) else -1)),
            "evaluations": sorted(evaluations, key=lambda row: (row.get("created_at") or "", row["gateway_record_id"])),
            "shape_results": shapes,
            "accepted_best_by_shape": {key: accepted_winners[key] for key in sorted(accepted_winners)},
            "observed_best_by_shape": {key: observed_winners[key] for key in sorted(observed_winners)}}


def _latency(value):
    return None if value is None else round(value, 6)


def render(value: dict) -> str:
    headline = []
    for name in ("accepted_best", "best_passing_full_eval"):
        row = value[name] or {}
        headline.append([name, _latency(row.get("latency_us_geomean")), row.get("kernel_commit_id"), row.get("gateway_record_id")])
    return "\n".join(["AKA Light 性能与正确性", "",
        table(["Metric", "GeoMean(us)", "Kernel commit", "Gateway Record"], headline),
        "", "Accepted-submission per-Shape frontier",
        table(["Shape", "Latency(us)", "Kernel", "Record"],
              [[shape, _latency(row["latency_us"]), row["kernel_commit_id"], row["gateway_record_id"]]
               for shape, row in value["accepted_best_by_shape"].items()])])


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
