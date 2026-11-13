#!/usr/bin/env python3
"""Build the durable Agate state consumed by the public call monitor."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import (
    add_output_arguments,
    age_seconds,
    discover_scopes,
    elapsed_seconds,
    emit,
    evaluation_result,
    iter_records,
    job_attempt_rows,
    job_error_details,
    job_identifier,
    job_state_rows,
    main_error,
    normalized_job_status,
    observed_at,
    operation_name,
    read_json,
    runtime_audit_paths,
    search_state,
    table,
)


_SUCCESS_JOB_STATUSES = {"completed", "succeeded"}
_SUCCESS_CALL_OUTCOMES = {"completed", "completed_not_cacheable"}
_FAILED_JOB_STATUSES = {"cancelled", "canceled", "error", "failed", "timed_out", "timeout"}
_PENDING_JOB_STATUSES = {
    "accepted",
    "pending",
    "queued",
    "running",
    "submitted",
    "validation_pending",
    "waiting",
}
_FILTER_STATUSES = {"succeeded", "failed", "pending", "unknown"}
_RECORD_ID = re.compile(r"gateway-[0-9a-f]{32}")
_COMPACT_JOB_FIELDS = (
    "scope",
    "trajectory",
    "source",
    "gateway_record_id",
    "runtime_request_ids",
    "operation",
    "job_id",
    "checkpoint_phase",
    "status",
    "execution_phase",
    "submission",
    "retry_count",
    "timing",
    "pending_seconds",
    "error",
)


def _retry_count(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    count = 0
    for item in value.values():
        if type(item) is int and item >= 0:
            count += item
        elif isinstance(item, list):
            count += len(item)
        elif isinstance(item, dict):
            count += 1
    return count


def _timing(job: object) -> dict[str, float | None]:
    if not isinstance(job, dict):
        return {"queue_seconds": None, "execution_seconds": None, "total_seconds": None}
    created = job.get("created_at") or job.get("submitted_at")
    started = job.get("started_at")
    finished = job.get("finished_at") or job.get("completed_at")
    return {
        "queue_seconds": elapsed_seconds(created, started),
        "execution_seconds": elapsed_seconds(started, finished),
        "total_seconds": elapsed_seconds(created, finished),
    }


def _pending_seconds(job: object, checkpoint_updated_at: object) -> float | None:
    if isinstance(job, dict):
        elapsed_ms = job.get("phase_elapsed_ms")
        if type(elapsed_ms) in {int, float} and float(elapsed_ms) >= 0:
            return float(elapsed_ms) / 1000.0
        created = job.get("created_at") or job.get("submitted_at")
        elapsed = age_seconds(created)
        if elapsed is not None:
            return elapsed
    return age_seconds(checkpoint_updated_at)


def _deduplicate_states(rows: list[dict]) -> tuple[list[dict], int]:
    """Collapse multiple checkpoint references to the same physical Agate Job."""
    grouped: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for index, row in enumerate(rows):
        job_id = job_identifier(row.get("job"))
        key = ("job", job_id) if job_id else (
            "checkpoint",
            row.get("path"),
            row.get("checkpoint_id"),
            row.get("submission"),
            index,
        )
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)

    unique = []
    for key in order:
        references = grouped[key]
        selected = max(
            references,
            key=lambda row: (
                type(row.get("submission")) is int and row.get("submission", 0) > 0,
                str(row.get("phase") or "") == "terminal",
                str(row.get("checkpoint_updated_at") or ""),
            ),
        )
        selected = dict(selected)
        selected["checkpoint_references"] = [
            {
                "checkpoint_id": row.get("checkpoint_id"),
                "checkpoint_path": row.get("checkpoint_path"),
                "submission": row.get("submission"),
                "phase": row.get("phase"),
                "path": row.get("path"),
            }
            for row in references
        ]
        unique.append(selected)
    return unique, len(rows)


def _physical_job(scope, state: dict, *, source: str, record_id: str | None,
                  operation: str, response: object = None) -> dict:
    job = state["job"]
    found = job_error_details(job, response)
    raw_status = job.get("status") if isinstance(job, dict) else None
    status = normalized_job_status(state)
    checkpoint_phase = state["phase"]
    terminal = checkpoint_phase == "terminal" or status in {
        "succeeded", "failed", "cancelled", "canceled", "completed"
    }
    return {
        "scope": scope.label,
        "trajectory": scope.trajectory,
        "source": source,
        "gateway_record_id": record_id,
        "runtime_request_ids": state.get("runtime_request_ids", []),
        "checkpoint_id": state.get("checkpoint_id"),
        "operation": operation,
        "job_id": job_identifier(job),
        "checkpoint_phase": checkpoint_phase,
        "status": status,
        "raw_status": raw_status,
        "execution_phase": job.get("execution_phase") if isinstance(job, dict) else None,
        "submission": state["submission"],
        "checkpoint_updated_at": state.get("checkpoint_updated_at"),
        "retry_count": _retry_count(state["retries"]),
        "checkpoint_references": state.get("checkpoint_references", []),
        "timing": _timing(job),
        "pending_seconds": None if terminal else _pending_seconds(
            job, state.get("checkpoint_updated_at")
        ),
        "error": found,
    }


def _batch_progress(record: dict, jobs: list[dict]) -> dict:
    result = evaluation_result(record)
    total = completed = None
    if isinstance(result, dict):
        total = result.get("shape_batch_count")
        completed = result.get("completed_shape_batch_count", total)
    batch_jobs = [
        row for row in jobs
        if isinstance(row.get("identity"), dict)
        and row["identity"].get("kind") in {"eval", "abba"}
    ]
    reused = [
        row for row in batch_jobs
        if row.get("submission") == 0
        and isinstance(row.get("identity"), dict)
        and row["identity"].get("batch_checkpoint")
    ]
    return {
        "shape_batch_count": total,
        "completed_shape_batch_count": completed,
        "physical_batch_jobs": len(batch_jobs),
        "reused_shape_batches": len(reused),
        "submitted_shape_batches": len(batch_jobs) - len(reused),
        "completed_result_reused": (
            bool(record.get("response", {}).get("reused"))
            if isinstance(record.get("response"), dict)
            else False
        ),
    }


def snapshot(workspace: Path) -> dict:
    calls = []
    physical_jobs = []
    submission_attempts = []
    task_markers = []
    audit_duplicates = 0
    audit_running = 0
    current_checkpoint_references = 0
    submission_references = 0
    scanned_measurement_roots: set[Path] = set()
    for scope in discover_scopes(workspace):
        measurement_root = scope.measurement_root.resolve()
        if measurement_root not in scanned_measurement_roots:
            scanned_measurement_roots.add(measurement_root)
            for marker in sorted((measurement_root / "tasks").glob("*.json")):
                value = read_json(marker, default={})
                if isinstance(value, dict):
                    task_markers.append({
                        "scope": "campaign",
                        "task_digest": marker.stem,
                        **value,
                    })
        for row in iter_records(scope, include_incomplete=True):
            record = row.get("record", {})
            request = row.get("request", {})
            response = record.get("response") if isinstance(record, dict) else None
            jobs, raw_job_count = _deduplicate_states(
                job_state_rows(row["directory"])
            )
            attempts, raw_attempt_count = _deduplicate_states(
                job_attempt_rows(row["directory"])
            )
            current_checkpoint_references += raw_job_count
            submission_references += raw_attempt_count
            errors = []
            retries = 0
            for state in jobs:
                retries += _retry_count(state["retries"])
                found = job_error_details(state["job"], response)
                if found:
                    errors.append(found)
                physical_jobs.append(_physical_job(
                    scope,
                    state,
                    source="measurement_record",
                    record_id=row["gateway_record_id"],
                    operation=operation_name(record or request),
                    response=response,
                ))
            for state in attempts:
                submission_attempts.append(_physical_job(
                    scope, state, source="measurement_record",
                    record_id=row["gateway_record_id"],
                    operation=operation_name(record or request), response=response,
                ))
            attempt_errors = [item["error"] for item in submission_attempts
                              if item["scope"] == scope.label
                              and item["gateway_record_id"] == row["gateway_record_id"]
                              and item.get("error")]
            primary_error = attempt_errors[-1] if attempt_errors else errors[-1] if errors else job_error_details(None, response)
            cacheable = record.get("cacheable") if isinstance(record, dict) else None
            if not row["complete"]:
                outcome = "running_or_interrupted"
            elif primary_error is not None:
                outcome = "failed"
            elif cacheable is True:
                outcome = "completed"
            else:
                outcome = "completed_not_cacheable"
            call = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "gateway_record_id": row["gateway_record_id"],
                "operation": operation_name(record or request),
                "created_at": record.get("created_at") if isinstance(record, dict) else None,
                "complete": row["complete"],
                "cacheable": cacheable,
                "outcome": outcome,
                "current_checkpoint_job_count": len(jobs),
                "current_checkpoint_reference_count": raw_job_count,
                "submission_attempt_count": len(attempts),
                "submission_reference_count": raw_attempt_count,
                "historical_retry_attempt_count": max(0, len(attempts) - len(jobs)),
                "retry_count": retries,
                "error": primary_error,
                "batch": _batch_progress(record, jobs) if isinstance(record, dict) else None,
            }
            calls.append(call)
        trusted_root = scope.audit_root / "trusted-jobs"
        trusted_jobs, trusted_job_references = _deduplicate_states(
            job_state_rows(trusted_root)
        )
        current_checkpoint_references += trusted_job_references
        for state in trusted_jobs:
            identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
            physical_jobs.append(_physical_job(
                scope,
                state,
                source="trusted",
                record_id=None,
                operation=operation_name({"operation": identity.get("kind")}),
            ))
        trusted_attempts, trusted_attempt_references = _deduplicate_states(
            job_attempt_rows(trusted_root)
        )
        submission_references += trusted_attempt_references
        for state in trusted_attempts:
            identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
            submission_attempts.append(_physical_job(
                scope, state, source="trusted", record_id=None,
                operation=operation_name({"operation": identity.get("kind")}),
            ))
        for audit in runtime_audit_paths(scope):
            value = read_json(audit, default={})
            if not isinstance(value, dict):
                continue
            text = str(value.get("stdout", "")) + "\n" + str(value.get("stderr", ""))
            audit_duplicates += text.count("duplicate_gateway_task")
            audit_running += text.count("gateway_task_running")

    operation_counts = Counter(call["operation"] for call in calls)
    job_counts = Counter((row["status"] or row["checkpoint_phase"] or "unknown") for row in physical_jobs)
    trusted_operation_counts = Counter(
        row["operation"] for row in physical_jobs if row["source"] == "trusted"
    )
    error_counts = Counter(row["error"]["category"] for row in submission_attempts if isinstance(row.get("error"), dict))
    timeout_counts = Counter()
    for row in submission_attempts:
        error = row.get("error") or {}
        text = " ".join(str(error.get(key) or "") for key in ("reason", "message")).casefold()
        for label, markers in {
            "queue_timeout": ("queue timeout", "queue_timeout"),
            "job_deadline": ("job deadline", "deadline exceeded"),
            "http_timeout": ("http timeout", "http_timeout"),
            "runtime_deadline": ("runtime_request_deadline", "gateway_deadline_exceeded"),
        }.items():
            if any(marker in text for marker in markers):
                timeout_counts[label] += 1

    search = search_state(Path(workspace))
    pause = None
    if isinstance(search, dict):
        binding = search.get("binding") if isinstance(search.get("binding"), dict) else {}
        runtime = binding.get("runtime") if isinstance(binding.get("runtime"), dict) else binding
        if isinstance(runtime, dict) and isinstance(runtime.get("pause_agent_during_requests"), bool):
            pause = runtime["pause_agent_during_requests"]
    running_markers = [row for row in task_markers if row.get("status") in {"running", "validation_pending"}]
    pending_jobs = [row for row in physical_jobs if row["pending_seconds"] is not None]
    result = {
        "schema_version": 3,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "summary": {
            "operation_counts": dict(sorted(operation_counts.items())),
            "trusted_operation_counts": dict(sorted(trusted_operation_counts.items())),
            "current_job_status_counts": dict(sorted(job_counts.items())),
            "error_counts": dict(sorted(error_counts.items())),
            "timeout_counts": dict(sorted(timeout_counts.items())),
            "logical_call_count": len(calls),
            "current_checkpoint_job_count": len(physical_jobs),
            "current_checkpoint_reference_count": current_checkpoint_references,
            "submission_attempt_count": len(submission_attempts),
            "submission_reference_count": submission_references,
            "historical_retry_attempt_count": sum(
                call["historical_retry_attempt_count"] for call in calls
            ),
            "retry_count": sum(row["retry_count"] for row in physical_jobs),
            "duplicate_cache_hits": audit_duplicates,
            "same_request_running": audit_running,
        },
        "running_task_markers": running_markers,
        "pending_jobs": pending_jobs,
        "calls": calls,
        "current_jobs": physical_jobs,
        "submission_attempts": submission_attempts,
    }
    if pause is not None:
        result["agent_paused_during_requests"] = pause
    return result


def _job_has_issue(row: object) -> bool:
    return _job_status(row) != "succeeded"


def _job_status(row: object) -> str:
    if not isinstance(row, dict):
        return "unknown"
    if isinstance(row.get("error"), dict):
        return "failed"
    if row.get("pending_seconds") is not None:
        return "pending"
    status = str(row.get("status") or "").casefold()
    if status in _SUCCESS_JOB_STATUSES:
        return "succeeded"
    if status in _FAILED_JOB_STATUSES:
        return "failed"
    if status in _PENDING_JOB_STATUSES:
        return "pending"
    return "unknown"


def _call_has_issue(row: object) -> bool:
    return _call_status(row) != "succeeded"


def _call_status(row: object) -> str:
    if not isinstance(row, dict):
        return "unknown"
    if isinstance(row.get("error"), dict) or row.get("outcome") == "failed":
        return "failed"
    if row.get("complete") is not True or row.get("outcome") == "running_or_interrupted":
        return "pending"
    if row.get("outcome") in _SUCCESS_CALL_OUTCOMES:
        return "succeeded"
    return "unknown"


def _compact_job(row: dict) -> dict:
    return {field: row.get(field) for field in _COMPACT_JOB_FIELDS}


def _created_at_key(row: dict) -> tuple[float, str]:
    value = row.get("created_at")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp(), str(row.get("gateway_record_id") or "")
        except ValueError:
            pass
    return float("-inf"), str(row.get("gateway_record_id") or "")


def project_details(
    value: dict,
    *,
    jobs: str = "issues",
    gateway_record_ids: tuple[str, ...] = (),
    last: int | None = None,
    statuses: tuple[str, ...] = (),
) -> dict:
    """Project a bounded default view while retaining explicit full-detail modes."""
    if jobs not in {"issues", "all", "none"}:
        raise ValueError("jobs must be one of: issues, all, none")
    if last is not None and (type(last) is not int or last <= 0):
        raise ValueError("--last must be a positive integer")

    selected_statuses = tuple(dict.fromkeys(statuses))
    invalid_statuses = set(selected_statuses) - _FILTER_STATUSES
    if invalid_statuses:
        raise ValueError(
            "--status must be one of: succeeded, failed, pending, unknown"
        )

    record_ids = tuple(dict.fromkeys(gateway_record_ids))
    for record_id in record_ids:
        if _RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("--gateway-record-id must be a complete gateway- ID")

    all_calls = list(value.get("calls") or [])
    all_current = list(value.get("current_jobs") or [])
    all_attempts = list(value.get("submission_attempts") or [])
    all_pending = list(value.get("pending_jobs") or [])

    explicit_filter = bool(record_ids or last is not None or selected_statuses)
    if record_ids:
        selected = set(record_ids)
        calls = [row for row in all_calls if row.get("gateway_record_id") in selected]
        current = [row for row in all_current if row.get("gateway_record_id") in selected]
        attempts = [row for row in all_attempts if row.get("gateway_record_id") in selected]
        pending = [row for row in all_pending if row.get("gateway_record_id") in selected]
        selection = "gateway_records"
    else:
        calls = all_calls
        current = all_current
        attempts = all_attempts
        pending = all_pending

    if selected_statuses:
        wanted = set(selected_statuses)
        current = [row for row in current if _job_status(row) in wanted]
        attempts = [row for row in attempts if _job_status(row) in wanted]
        pending = [row for row in pending if _job_status(row) in wanted]
        matching_record_ids = {
            row.get("gateway_record_id")
            for row in current + attempts + pending
            if row.get("gateway_record_id")
        }
        calls = [
            row for row in calls
            if _call_status(row) in wanted
            or row.get("gateway_record_id") in matching_record_ids
        ]

    if last is not None:
        calls = sorted(calls, key=_created_at_key, reverse=True)[:last]
        recent_record_ids = {
            row.get("gateway_record_id")
            for row in calls
            if row.get("gateway_record_id")
        }
        current = [row for row in current if row.get("gateway_record_id") in recent_record_ids]
        attempts = [row for row in attempts if row.get("gateway_record_id") in recent_record_ids]
        pending = [row for row in pending if row.get("gateway_record_id") in recent_record_ids]

    if explicit_filter:
        if jobs != "all" and not record_ids:
            current = [_compact_job(row) for row in current]
            attempts = [_compact_job(row) for row in attempts]
            pending = [_compact_job(row) for row in pending]
        if record_ids:
            selection = "gateway_records"
        elif last is not None and selected_statuses:
            selection = "recent_statuses"
        elif last is not None:
            selection = "recent"
        else:
            selection = "statuses"
    elif jobs == "all":
        selection = "all"
    elif jobs == "none":
        calls = []
        current = []
        attempts = []
        pending = []
        selection = "none"
    else:
        calls = [row for row in all_calls if _call_has_issue(row)]
        current = [
            _compact_job(row) for row in all_current if _job_has_issue(row)
        ]
        attempts = [
            _compact_job(row) for row in all_attempts if _job_has_issue(row)
        ]
        pending = [_compact_job(row) for row in all_pending]
        selection = "issues"

    projected = dict(value)
    projected["calls"] = calls
    projected["current_jobs"] = current
    projected["submission_attempts"] = attempts
    projected["pending_jobs"] = pending
    projected["detail_selection"] = {
        "mode": selection,
        "gateway_record_ids": list(record_ids),
        "last": last,
        "statuses": list(selected_statuses),
        "summary_scope": "campaign",
        "returned_counts": {
            "logical_calls": len(calls),
            "current_jobs": len(current),
            "submission_attempts": len(attempts),
        },
        "omitted_counts": {
            "logical_calls": len(all_calls) - len(calls),
            "current_jobs": len(all_current) - len(current),
            "submission_attempts": len(all_attempts) - len(attempts),
        },
        "full_detail_hint": (
            "Use --jobs all for full rows, --last N for recent logical calls, "
            "--status for Job outcomes, or --gateway-record-id gateway-... for "
            "one logical call."
        ),
    }
    return projected


def render(value: dict) -> str:
    summary = value["summary"]
    lines = ["AKA Light Agate 调用与故障", ""]
    lines.append(table(
        ["Operation", "Calls"],
        [[name, count] for name, count in summary["operation_counts"].items()],
    ))
    lines += ["", f"Current checkpoints: {summary['current_checkpoint_job_count']}  Historical submissions: {summary['submission_attempt_count']}  Historical retries: {summary['historical_retry_attempt_count']}  Retry counters: {summary['retry_count']}  Duplicate: {summary['duplicate_cache_hits']}  Running duplicate: {summary['same_request_running']}", ""]
    lines.append(table(
        ["Record", "Operation", "Complete", "Current", "Attempts", "Retry attempts", "Retry counters", "Error"],
        [[row["gateway_record_id"], row["operation"], row["complete"], row["current_checkpoint_job_count"], row["submission_attempt_count"],
          row["historical_retry_attempt_count"], row["retry_count"], (row["error"] or {}).get("category")]
         for row in value["calls"]],
    ))
    if summary["trusted_operation_counts"]:
        lines += ["", "Trusted Bootstrap/Baseline jobs"]
        lines.append(table(
            ["Operation", "Physical Jobs"],
            [[name, count] for name, count in summary["trusted_operation_counts"].items()],
        ))
    if value["pending_jobs"]:
        lines += ["", "Pending Agate jobs"]
        lines.append(table(
            ["Source", "Record", "Job", "Operation", "Status", "Execution Phase", "Wait(s)"],
            [[
                row["source"], row["gateway_record_id"], row["job_id"], row["operation"],
                row["status"] or row["checkpoint_phase"], row["execution_phase"],
                round(row["pending_seconds"], 1),
            ] for row in value["pending_jobs"]],
        ))
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--jobs",
        choices=("issues", "all", "none"),
        default="issues",
        help=(
            "physical Job detail: compact pending/failed details (issues, default), "
            "every Job and logical call (all), or summary only (none)"
        ),
    )
    parser.add_argument(
        "--gateway-record-id",
        action="append",
        default=[],
        help=(
            "expand all captured Job details for this Gateway Record; repeatable and "
            "takes precedence over --jobs"
        ),
    )
    parser.add_argument(
        "--last",
        type=int,
        help=(
            "show the newest N matching logical Gateway calls by created_at and "
            "their associated physical Jobs"
        ),
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=tuple(sorted(_FILTER_STATUSES)),
        default=[],
        help=(
            "filter physical Job outcomes; repeat for a union of succeeded, failed, "
            "pending, or unknown"
        ),
    )
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = project_details(
        snapshot(options.workspace),
        jobs=options.jobs,
        gateway_record_ids=tuple(options.gateway_record_id),
        last=options.last,
        statuses=tuple(options.status),
    )
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-agate-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
