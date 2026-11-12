#!/usr/bin/env python3
"""Read submit-kernel decisions and their exact full-Evaluate evidence."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.light.evaluation.provider import result_from_response  # noqa: E402
from aka.light.experiment.store import KERNEL_ID, RECORD_ID  # noqa: E402
from observability.common import (  # noqa: E402
    Scope, discover_scopes, existing_store, finite_positive, iter_records, read_json,
)


def eligible_full_eval(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the result only for a passing, complete standard full Evaluate."""
    request = record.get("request")
    if (record.get("operation") != "evaluate" or record.get("cacheable") is not True
            or not isinstance(request, dict) or request.get("operation") != "evaluate"):
        return None
    options = request.get("options")
    if not isinstance(options, dict):
        return None
    policy, command = options.get("evaluation_policy", {}), options.get("command", [])
    if not isinstance(policy, dict) or not isinstance(command, list):
        return None
    if ((options.get("evaluation_mode") or "full") != "full"
            or policy.get("mode", "full") != "full"
            or options.get("evaluation_input_path") or options.get("evaluation_shapes_path")
            or options.get("baseline_path") or command[:2] != ["python3", "test_kernel.py"]
            or any(isinstance(arg, str) and (arg == "--shape-id" or arg.startswith("--shape-id=")) for arg in command)):
        return None
    response = record.get("response")
    if not isinstance(response, dict) or response.get("exit_code") != 0 or response.get("truncated"):
        return None
    try:
        result = result_from_response(response, "evaluate")
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if (not isinstance(result, dict) or result.get("all_pass") is not True
            or result.get("error") or result.get("mode", "full") != "full"):
        return None
    latencies = result.get("latency_us_by_shape")
    if not isinstance(latencies, dict) or not latencies or any(finite_positive(value) is None for value in latencies.values()):
        return None
    return result


def _latency(result: dict[str, Any]) -> float | None:
    latency = finite_positive(result.get("latency_us_geomean"))
    if latency is not None:
        return latency
    values = [finite_positive(value) for value in (result.get("latency_us_by_shape") or {}).values()]
    if not values or any(value is None for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def full_eval_rows(scope: Scope) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in iter_records(scope):
        record = item.get("record")
        result = eligible_full_eval(record) if isinstance(record, dict) else None
        if result is None or (latency := _latency(result)) is None:
            continue
        identity = record.get("kernels", {}).get("candidate")
        identity = identity if isinstance(identity, dict) else {}
        commit_id = identity.get("kernel_id")
        if not isinstance(commit_id, str) or KERNEL_ID.fullmatch(commit_id) is None:
            continue
        rows.append({
            "scope": scope.label, "trajectory": scope.trajectory,
            "latency_us_geomean": latency,
            "latency_us_arith_mean": finite_positive(result.get("latency_us_arith_mean")),
            "kernel_commit_id": commit_id,
            "kernel_artifact_digest": identity.get("kernel_artifact_digest"),
            "gateway_record_id": item["gateway_record_id"],
            "shape_count": len(result.get("latency_us_by_shape") or {}),
            "latency_us_by_shape": dict(result.get("latency_us_by_shape") or {}),
            "measurement_workspace": str(scope.workspace),
        })
    return rows


def submission_rows(scope: Scope) -> list[dict[str, Any]]:
    """Return every submit-kernel audit entry from private Runtime Journals."""
    rows: list[dict[str, Any]] = []
    for path in sorted((scope.audit_root / "journals" / "episodes").glob("e*/journal.json")):
        journal = read_json(path, default={})
        if not isinstance(journal, dict):
            continue
        episode = journal.get("episode")
        if type(episode) is not int:
            try:
                episode = int(path.parent.name[1:])
            except ValueError:
                episode = None
        submissions = journal.get("kernel_submissions")
        if not isinstance(submissions, list):
            continue
        for index, submission in enumerate(submissions, start=1):
            if not isinstance(submission, dict):
                continue
            evidence = submission.get("evaluation_evidence")
            review = submission.get("review")
            diagnostics = submission.get("diagnostics")
            rows.append({
                "scope": scope.label, "trajectory": scope.trajectory, "episode": episode,
                "submission_index": index, "submission_id": submission.get("submission_id"),
                "submitted_at": submission.get("submitted_at"),
                "kernel_commit_id": submission.get("kernel_commit_id"),
                "accepted": submission.get("accepted") is True,
                "review": review if isinstance(review, dict) else {},
                "diagnostics": list(diagnostics) if isinstance(diagnostics, list) else [],
                "evaluation_evidence": evidence if isinstance(evidence, dict) else {},
                "token_checkpoint": submission.get("token_checkpoint"),
                "journal_path": str(path),
            })
    return rows


def _join_submission_evidence(scope: Scope, submission: dict[str, Any]) -> dict[str, Any]:
    row = dict(submission)
    evidence = submission.get("evaluation_evidence") or {}
    record_id = evidence.get("gateway_record_id")
    row.update({"gateway_record_id": record_id if isinstance(record_id, str) else None,
                "evidence_status": "not_accepted" if not submission.get("accepted") else "missing",
                "latency_us_geomean": None, "latency_us_arith_mean": None,
                "shape_count": None, "kernel_artifact_digest": None})
    if not submission.get("accepted"):
        return row
    if not isinstance(record_id, str) or RECORD_ID.fullmatch(record_id) is None:
        row["evidence_status"] = "gateway_record_id_missing"
        return row
    store = existing_store(scope)
    if store is None:
        row["evidence_status"] = "measurement_store_unavailable"
        return row
    try:
        record = store.read(record_id)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        row["evidence_status"] = "gateway_record_unreadable"
        return row
    result = eligible_full_eval(record)
    if result is None:
        row["evidence_status"] = "full_evaluate_ineligible"
        return row
    identity = record.get("kernels", {}).get("candidate")
    identity = identity if isinstance(identity, dict) else {}
    if identity.get("kernel_id") != submission.get("kernel_commit_id"):
        row["evidence_status"] = "kernel_identity_mismatch"
        return row
    row.update({"evidence_status": "verified", "latency_us_geomean": _latency(result),
                "latency_us_arith_mean": finite_positive(result.get("latency_us_arith_mean")),
                "shape_count": len(result.get("latency_us_by_shape") or {}),
                "latency_us_by_shape": dict(result.get("latency_us_by_shape") or {}),
                "kernel_artifact_digest": identity.get("kernel_artifact_digest")})
    return row


def campaign_submission_snapshot(workspace: Path) -> dict[str, Any]:
    submissions, evaluations = [], []
    for scope in discover_scopes(workspace):
        submissions.extend(_join_submission_evidence(scope, row) for row in submission_rows(scope))
        evaluations.extend(full_eval_rows(scope))
    accepted = [row for row in submissions if row["accepted"] and row["evidence_status"] == "verified"]
    by_lineage: dict[str, dict[str, Any]] = {}
    for row in accepted:
        previous = by_lineage.get(row["scope"])
        if previous is None or row["latency_us_geomean"] < previous["latency_us_geomean"]:
            by_lineage[row["scope"]] = row
    lineage_best = [by_lineage[key] for key in sorted(by_lineage)]
    global_best = min(lineage_best, key=lambda row: (row["latency_us_geomean"], row["kernel_commit_id"]), default=None) if len(lineage_best) > 1 else None
    return {"submissions": submissions, "accepted_submissions": accepted,
            "lineage_best": lineage_best, "global_best": global_best,
            "full_evaluations": evaluations}


def accepted_best(workspace: Path) -> dict[str, Any] | None:
    rows = campaign_submission_snapshot(workspace)["lineage_best"]
    return min(rows, key=lambda row: (row["latency_us_geomean"], row["kernel_commit_id"]), default=None)


def best_passing_full_eval(workspace: Path) -> dict[str, Any] | None:
    rows = campaign_submission_snapshot(workspace)["full_evaluations"]
    return min(rows, key=lambda row: (row["latency_us_geomean"], row["kernel_commit_id"], row["gateway_record_id"]), default=None)
