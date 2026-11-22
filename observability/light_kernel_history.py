#!/usr/bin/env python3
"""List evaluated Kernels with their submit-kernel decisions by lineage."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import add_output_arguments, emit, main_error, observed_at, table
from observability._internal.token_usage import snapshot as token_snapshot
from observability._internal.submissions import campaign_submission_snapshot
from observability._internal.view_helpers import checkpoint_indexes, evidence_health, milestone_usage


def _checkpoint(
    *,
    scope: str,
    event: str,
    checkpoint_id: object = None,
    record_id: object,
    commit_id: object,
    by_id: dict,
    by_record: dict,
    by_commit: dict,
) -> dict | None:
    row = None
    if isinstance(checkpoint_id, str):
        row = by_id.get((scope, event, checkpoint_id))
    if isinstance(record_id, str):
        row = row or by_record.get((scope, event, record_id))
    if row is None and isinstance(commit_id, str):
        row = by_commit.get((scope, event, commit_id))
    return milestone_usage(row)


def _result_event_time(row: dict) -> str:
    return max(
        [
            str(row.get("evaluated_at") or ""),
            *[
                str(submission.get("submitted_at") or "")
                for submission in row.get("submissions", [])
            ],
        ]
    )


def snapshot(
    workspace: Path,
    *,
    lineages: tuple[str, ...] = (),
    commits: tuple[str, ...] = (),
    last: int | None = None,
) -> dict:
    if last is not None and (type(last) is not int or last <= 0):
        raise ValueError("--last must be a positive integer")
    submissions = campaign_submission_snapshot(workspace)
    tokens = token_snapshot(workspace)
    checkpoints = tokens.get("token_checkpoints", [])
    by_record, by_commit = checkpoint_indexes(checkpoints)
    by_id = {
        (row.get("scope"), row.get("event"), row.get("checkpoint_id")): row
        for row in checkpoints
        if isinstance(row.get("checkpoint_id"), str)
    }
    by_scope: dict[str, dict] = {}
    result_by_identity: dict[tuple[str, object, object], dict] = {}
    missing_checkpoints = []
    selected_evaluations = submissions["full_evaluations"]
    selected_submissions = submissions["submissions"]
    if lineages:
        wanted = set(lineages)
        selected_evaluations = [row for row in selected_evaluations if row.get("scope") in wanted]
        selected_submissions = [row for row in selected_submissions if row.get("scope") in wanted]
    if commits:
        wanted_commits = set(commits)
        selected_evaluations = [
            row for row in selected_evaluations if row.get("kernel_commit_id") in wanted_commits
        ]
        selected_submissions = [
            row for row in selected_submissions if row.get("kernel_commit_id") in wanted_commits
        ]
    for evaluation in selected_evaluations:
        scope = evaluation["scope"]
        item = by_scope.setdefault(
            scope,
            {
                "lineage": scope,
                "trajectory": evaluation.get("trajectory"),
                "kernel_results": [],
            },
        )
        record_id = evaluation.get("gateway_record_id")
        commit_id = evaluation.get("kernel_commit_id")
        token_usage = _checkpoint(
            scope=scope,
            event="passing-full-evaluate",
            checkpoint_id=record_id,
            record_id=record_id,
            commit_id=commit_id,
            by_id=by_id,
            by_record=by_record,
            by_commit=by_commit,
        )
        if token_usage is None:
            missing_checkpoints.append({
                "scope": scope,
                "event": "passing-full-evaluate",
                "gateway_record_id": record_id,
                "kernel_commit_id": commit_id,
            })
        result = {
            "kernel_commit_id": commit_id,
            "gateway_record_id": record_id,
            "evaluated_at": evaluation.get("created_at"),
            "performance": {
                "latency_us_geomean": evaluation.get("latency_us_geomean"),
                "latency_us_arith_mean": evaluation.get("latency_us_arith_mean"),
                "shape_count": evaluation.get("shape_count"),
                "latency_us_by_shape": evaluation.get("latency_us_by_shape") or {},
            },
            "evaluate_token_usage": token_usage,
            "evaluate_token_usage_availability": (
                "available" if token_usage else "missing_checkpoint"
            ),
            "submissions": [],
        }
        item["kernel_results"].append(result)
        result_by_identity[(scope, commit_id, record_id)] = result

    for submission in selected_submissions:
        scope = submission["scope"]
        item = by_scope.setdefault(
            scope,
            {
                "lineage": scope,
                "trajectory": submission.get("trajectory"),
                "kernel_results": [],
            },
        )
        record_id = submission.get("gateway_record_id")
        commit_id = submission.get("kernel_commit_id")
        identity = (scope, commit_id, record_id)
        result = result_by_identity.get(identity)
        if result is None:
            # A submission may bind a peer's central Evaluate without a local
            # evaluate call. Keep the verified performance, but never charge
            # the peer's Evaluate token checkpoint to this lineage.
            shared_evidence = submission.get("evidence_status") == "verified"
            result = {
                "kernel_commit_id": commit_id,
                "gateway_record_id": record_id,
                "evaluated_at": submission.get("evaluated_at") if shared_evidence else None,
                "performance": ({
                    "latency_us_geomean": submission.get("latency_us_geomean"),
                    "latency_us_arith_mean": submission.get("latency_us_arith_mean"),
                    "shape_count": submission.get("shape_count"),
                    "latency_us_by_shape": submission.get("latency_us_by_shape") or {},
                } if shared_evidence else None),
                "evaluate_token_usage": None,
                "evaluate_token_usage_availability": (
                    "shared_evidence_no_local_evaluate" if shared_evidence else "not_available"
                ),
                "submissions": [],
            }
            item["kernel_results"].append(result)
            result_by_identity[identity] = result

        accepted = submission.get("accepted") is True
        token_usage = None
        if accepted:
            token_reference = submission.get("token_checkpoint")
            token_reference = token_reference if isinstance(token_reference, dict) else {}
            token_usage = _checkpoint(
                scope=scope,
                event="accepted-submit-kernel",
                checkpoint_id=token_reference.get("checkpoint_id"),
                record_id=record_id,
                commit_id=commit_id,
                by_id=by_id,
                by_record=by_record,
                by_commit=by_commit,
            )
            if token_usage is None:
                missing_checkpoints.append({
                    "scope": scope,
                    "event": "accepted-submit-kernel",
                    "gateway_record_id": record_id,
                    "kernel_commit_id": commit_id,
                })
        result["submissions"].append(
            {
                "submission_id": submission.get("submission_id"),
                "submission_revision": submission.get("submission_revision", 1),
                "submission_index": submission.get("submission_index"),
                "episode": submission.get("episode"),
                "submitted_at": submission.get("submitted_at"),
                "accepted": accepted,
                "evidence_status": submission.get("evidence_status"),
                "source_contract": submission.get("source_contract"),
                "review": submission.get("review") or {},
                "implementation_review": submission.get("implementation_review") or {
                    "availability": "unavailable"
                },
                "diagnostics": submission.get("diagnostics") or [],
                "submit_token_usage": token_usage,
                "submit_token_usage_availability": (
                    "available"
                    if token_usage
                    else "missing_checkpoint"
                    if accepted
                    else "not_recorded"
                ),
            }
        )

    for item in by_scope.values():
        for result in item["kernel_results"]:
            result["submissions"].sort(
                key=lambda row: (
                    str(row.get("submitted_at") or ""),
                    row.get("submission_index") or 0,
                )
            )
        item["kernel_results"].sort(
            key=lambda row: (
                _result_event_time(row),
                str(row.get("gateway_record_id") or ""),
                str(row.get("kernel_commit_id") or ""),
            )
        )
        if last is not None:
            item["kernel_results"] = item["kernel_results"][-last:]
    return {
        "schema_version": 2,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "lineages": [by_scope[key] for key in sorted(by_scope)],
        "evidence_health": evidence_health(
            workspace,
            submissions,
            missing_token_checkpoints=missing_checkpoints,
        ),
        "detail_selection": {
            "lineages": list(dict.fromkeys(lineages)),
            "kernel_commit_ids": list(dict.fromkeys(commits)),
            "last_per_lineage": last,
        },
    }


def render(value: dict) -> str:
    rows = []
    for lineage in value["lineages"]:
        for row in lineage["kernel_results"]:
            performance = row.get("performance") or {}
            accepted = any(
                submission.get("accepted") is True
                for submission in row["submissions"]
            )
            technologies = []
            for submission in row["submissions"]:
                implementation = submission.get("implementation_review") or {}
                observed = implementation.get("technologies")
                if isinstance(observed, list):
                    for technology in observed:
                        if isinstance(technology, str) and technology not in technologies:
                            technologies.append(technology)
            token = row.get("evaluate_token_usage") or {}
            all_paths = token.get("all_paths") or {}
            rows.append(
                [
                    lineage["lineage"],
                    (
                        "full-eval+submit"
                        if accepted
                        else "full-eval"
                        if performance
                        else "submit-only"
                    ),
                    row["kernel_commit_id"],
                    (
                        round(performance["latency_us_geomean"], 6)
                        if isinstance(performance.get("latency_us_geomean"), (int, float))
                        else None
                    ),
                    performance.get("shape_count"),
                    ",".join(technologies) if technologies else "-",
                    all_paths.get("total_tokens"),
                    row["gateway_record_id"],
                ]
            )
    return "\n".join(
        [
            "AKA Light Kernel history",
            "",
            table(
                [
                    "Lineage",
                    "Event",
                    "Commit",
                    "GeoMean(us)",
                    "Shapes",
                    "Implementation",
                    "Tokens",
                    "Gateway Record",
                ],
                rows,
            ),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--lineage", action="append", default=[], help="lineage scope; repeatable")
    parser.add_argument("--kernel-commit-id", action="append", default=[], help="Agent commit; repeatable")
    parser.add_argument("--last", type=int, help="newest N Kernel results per lineage")
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(
        options.workspace,
        lineages=tuple(options.lineage),
        commits=tuple(options.kernel_commit_id),
        last=options.last,
    )
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-kernel-history", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
