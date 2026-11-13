#!/usr/bin/env python3
"""Small factual projections shared by the operator-facing status commands."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable


TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)

SUCCESS_STATUSES = frozenset({"completed", "passed", "reused", "succeeded", "success"})
FAILURE_STATUSES = frozenset(
    {"cancelled", "canceled", "error", "failed", "failure", "rejected", "timed_out", "timeout"}
)
PENDING_STATUSES = frozenset(
    {"accepted", "pending", "queued", "running", "submitted", "validation_pending", "waiting"}
)
FILTER_STATUSES = ("succeeded", "failed", "pending", "unknown")


def normalized_outcome(*, success: object = None, status: object = None, error: object = None) -> str:
    if isinstance(error, dict):
        return "failed"
    if success is True:
        return "succeeded"
    if success is False:
        return "failed"
    value = str(status or "").casefold()
    if value in SUCCESS_STATUSES:
        return "succeeded"
    if value in FAILURE_STATUSES:
        return "failed"
    if value in PENDING_STATUSES:
        return "pending"
    return "unknown"


def usage_sum(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    usage: dict[str, Any] = {}
    for key in TOKEN_KEYS:
        values = [row.get(key) for row in materialized if type(row.get(key)) is int]
        usage[key] = sum(values) if values else None
    measurements = [row.get("measurement") for row in materialized]
    available = sum(type(row.get("total_tokens")) is int for row in materialized)
    usage["measurement"] = (
        "unavailable"
        if not materialized or not available
        else "exact"
        if available == len(materialized) and all(value == "exact" for value in measurements)
        else "estimated"
        if available == len(materialized) and all(value in {"exact", "estimated"} for value in measurements)
        else "partial"
    )
    return usage


def milestone_usage(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "checkpoint_id": row.get("checkpoint_id"),
        "occurred_at": row.get("occurred_at"),
        "all_paths": row.get("usage"),
        "main_agent": row.get("main_agent_usage"),
        "reviews": row.get("review_usage"),
        "reviews_by_kind": row.get("reviews_by_kind") or {},
        "warnings": row.get("warnings") or [],
    }

def checkpoint_indexes(rows: Iterable[dict[str, Any]]) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    by_record: dict[tuple, dict] = {}
    by_commit: dict[tuple, dict] = {}
    for row in rows:
        identity = row.get("identity") if isinstance(row.get("identity"), dict) else {}
        scope = row.get("scope")
        event = row.get("event")
        record_id = identity.get("gateway_record_id")
        commit_id = identity.get("kernel_commit_id")
        if isinstance(record_id, str):
            by_record[(scope, event, record_id)] = row
        if isinstance(commit_id, str):
            by_commit[(scope, event, commit_id)] = row
    return by_record, by_commit


def grouped_summary(
    rows: Iterable[dict[str, Any]],
    *,
    key: str,
    duration_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    result = []
    for name, items in sorted(groups.items()):
        counts = {status: sum(row.get("status") == status for row in items) for status in FILTER_STATUSES}
        summary: dict[str, Any] = {key: name, "calls": len(items), **counts}
        errors: dict[str, int] = defaultdict(int)
        for row in items:
            error = row.get("error") if isinstance(row.get("error"), dict) else {}
            label = (
                error.get("reason")
                or error.get("code")
                or error.get("error_class")
                or ("unknown_failure" if row.get("status") == "failed" else None)
            )
            if label:
                errors[str(label)] += 1
        summary["error_counts"] = dict(sorted(errors.items()))
        pending_ages = [
            float(row["pending_age_seconds"])
            for row in items
            if type(row.get("pending_age_seconds")) in {int, float}
            and math.isfinite(float(row["pending_age_seconds"]))
            and float(row["pending_age_seconds"]) >= 0
        ]
        summary["oldest_pending_seconds"] = max(pending_ages, default=None)
        for field in duration_fields:
            values = [
                float(row[field])
                for row in items
                if type(row.get(field)) in {int, float}
                and math.isfinite(float(row[field]))
                and float(row[field]) >= 0
            ]
            summary[f"average_{field}"] = sum(values) / len(values) if values else None
            summary[f"{field}_samples"] = len(values)
            summary[f"{field}_coverage"] = len(values) / len(items) if items else None
            ordered = sorted(values)
            for label, quantile in (("p50", 0.50), ("p95", 0.95)):
                if not ordered:
                    value = None
                elif len(ordered) == 1:
                    value = ordered[0]
                else:
                    position = (len(ordered) - 1) * quantile
                    lower = math.floor(position)
                    upper = math.ceil(position)
                    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (
                        position - lower
                    )
                summary[f"{label}_{field}"] = value
            summary[f"max_{field}"] = max(ordered, default=None)
        result.append(summary)
    return result


def evidence_health(
    workspace: Path,
    submissions: dict[str, Any],
    *,
    missing_token_checkpoints: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Report missing observability evidence instead of silently dropping it."""
    from observability._internal.common import (
        discover_scopes,
        existing_store,
        iter_records,
        read_json,
        runtime_audit_paths,
    )

    incomplete_records = []
    unreadable_records = []
    record_runtime_request_ids: dict[str, set[str]] = defaultdict(set)
    audit_request_ids: set[str] = set()
    unlinked_runtime_audits = []
    scopes = discover_scopes(workspace)
    for scope in scopes:
        if not hasattr(scope, "measurement_root") or not hasattr(scope, "audit_root"):
            continue
        known_records = set()
        for item in iter_records(scope, include_incomplete=True):
            record_id = item["gateway_record_id"]
            known_records.add(record_id)
            if item.get("read_error"):
                unreadable_records.append(
                    {
                        "scope": scope.label,
                        "gateway_record_id": record_id,
                        "error": item["read_error"],
                    }
                )
            elif not item.get("complete"):
                incomplete_records.append(
                    {"scope": scope.label, "gateway_record_id": record_id}
                )
            record = item.get("record") if isinstance(item.get("record"), dict) else {}
            request_id = record.get("runtime_request_id")
            if isinstance(request_id, str) and request_id:
                record_runtime_request_ids[scope.label].add(request_id)
        for path in runtime_audit_paths(scope):
            value = read_json(path, default={})
            if not isinstance(value, dict):
                continue
            request_id = value.get("request_id") or path.stem.removeprefix("request-")
            if isinstance(request_id, str) and request_id:
                audit_request_ids.add(f"{scope.label}:{request_id}")
            record_id = value.get("gateway_record_id")
            if isinstance(record_id, str) and record_id not in known_records:
                unlinked_runtime_audits.append(
                    {
                        "scope": scope.label,
                        "runtime_request_id": request_id,
                        "gateway_record_id": record_id,
                    }
                )

    missing_audits = [
        {"scope": scope, "runtime_request_id": request_id}
        for scope, request_ids in sorted(record_runtime_request_ids.items())
        for request_id in sorted(request_ids)
        if f"{scope}:{request_id}" not in audit_request_ids
    ]
    accepted_missing = [
        {
            "scope": row.get("scope"),
            "kernel_commit_id": row.get("kernel_commit_id"),
            "gateway_record_id": row.get("gateway_record_id"),
            "evidence_status": row.get("evidence_status"),
        }
        for row in submissions.get("submissions", [])
        if row.get("accepted") and row.get("evidence_status") != "verified"
    ]
    accepted_commits = {
        row.get("kernel_commit_id")
        for row in submissions.get("accepted_submissions", [])
        if isinstance(row.get("kernel_commit_id"), str)
    }
    stores = []
    store_roots: set[Path] = set()
    for scope in scopes:
        if not hasattr(scope, "measurement_root"):
            continue
        root = scope.measurement_root.resolve()
        if root in store_roots:
            continue
        store = existing_store(scope)
        if store is not None:
            store_roots.add(root)
            stores.append(store)
    unresolved_commits = []
    for commit_id in sorted(accepted_commits):
        resolved = False
        for store in stores:
            try:
                store.read_kernel(commit_id)
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                continue
            resolved = True
            break
        if not resolved:
            unresolved_commits.append(commit_id)

    missing_tokens = list(missing_token_checkpoints)
    issues = {
        "incomplete_gateway_records": incomplete_records,
        "unreadable_gateway_records": unreadable_records,
        "accepted_submissions_missing_evidence": accepted_missing,
        "missing_token_checkpoints": missing_tokens,
        "unresolvable_kernel_commits": unresolved_commits,
        "unlinked_runtime_audits": unlinked_runtime_audits,
        "gateway_records_missing_runtime_audit": missing_audits,
    }
    return {
        "complete": not any(issues.values()),
        "issue_counts": {key: len(value) for key, value in issues.items()},
        **issues,
    }
