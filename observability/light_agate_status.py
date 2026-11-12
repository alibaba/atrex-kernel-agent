#!/usr/bin/env python3
"""Report logical Agate calls, physical jobs, retries, batches and failures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments,
    age_seconds,
    discover_scopes,
    elapsed_seconds,
    emit,
    evaluation_result,
    existing_store,
    iter_records,
    job_identifier,
    job_state_rows,
    main_error,
    observed_at,
    operation_name,
    read_json,
    search_state,
    table,
)


def _error(job: object, response: object) -> dict | None:
    candidates = []
    if isinstance(job, dict):
        candidates.extend([job.get("error"), job.get("result", {}).get("error") if isinstance(job.get("result"), dict) else None])
    if isinstance(response, dict):
        for stream in (response.get("stderr"), response.get("stdout")):
            if isinstance(stream, str):
                try:
                    parsed = json.loads(stream.strip())
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed.get("error"))
                    if parsed.get("error_class") or parsed.get("code"):
                        candidates.append(parsed)
    value = next((item for item in candidates if isinstance(item, dict)), None)
    if value is None:
        return None
    error_class = value.get("error_class")
    reason = value.get("reason") or value.get("code")
    if error_class in {"candidate", "compilation", "correctness"}:
        category = "candidate"
    elif error_class == "infra" or reason in {"oss_unavailable", "gateway_unavailable"}:
        category = "infrastructure"
    elif error_class in {"validation", "policy"} or str(reason).startswith("source_"):
        category = "policy"
    else:
        category = "protocol" if error_class or reason else "unknown"
    return {
        "category": category,
        "error_class": error_class,
        "reason": reason,
        "message": value.get("message") or value.get("detail"),
    }


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


def _physical_job(scope, state: dict, *, source: str, record_id: str | None,
                  operation: str, response: object = None) -> dict:
    job = state["job"]
    found = _error(job, response)
    status = job.get("status") if isinstance(job, dict) else None
    checkpoint_phase = state["phase"]
    terminal = checkpoint_phase == "terminal" or status in {
        "succeeded", "failed", "cancelled", "canceled", "completed"
    }
    if isinstance(job, dict) and job.get('source') == 'recorded_execution_facts':
        terminal = job.get('execution_facts', {}).get('disposition') in {'terminal', 'not_submitted'}
    return {
        "scope": scope.label,
        "trajectory": scope.trajectory,
        "source": source,
        "gateway_record_id": record_id,
        "checkpoint_id": state.get("checkpoint_id"),
        "operation": operation,
        "job_id": job_identifier(job),
        "checkpoint_phase": checkpoint_phase,
        "status": status,
        "execution_phase": job.get("execution_phase") if isinstance(job, dict) else None,
        "submission": state["submission"],
        "retry_count": _retry_count(state["retries"]),
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
    task_markers = []
    audit_duplicates = 0
    audit_running = 0
    for scope in discover_scopes(workspace):
        store = existing_store(scope)
        legacy_root = scope.campaign_root / 'measurements'
        if store is None or store.legacy is not None or any((legacy_root / 'records').glob('gateway-*')):
            for marker in sorted((legacy_root / "tasks").glob("*.json")):
                value = read_json(marker, default={})
                if isinstance(value, dict):
                    task_markers.append({"scope": scope.label, "source": "legacy_task_marker", "task_digest": marker.stem, **value})
        for row in iter_records(scope, include_incomplete=True):
            record = row.get("record", {})
            shared = row.get("execution_record")
            if shared and shared.get('lifecycle') != 'terminal':
                task_markers.append({'scope': scope.label, 'source': 'shared_experiment',
                    'gateway_record_id': row['gateway_record_id'], 'status': 'validation_pending',
                    'lifecycle': shared['lifecycle'], 'phase': shared['phase'], 'updated_at': shared['updated_at']})
            request = row.get("request", {})
            response = record.get("response") if isinstance(record, dict) else None
            jobs = job_state_rows(row["directory"], record=shared)
            errors = []
            retries = 0
            for state in jobs:
                retries += _retry_count(state["retries"])
                found = _error(state["job"], response)
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
            primary_error = errors[-1] if errors else _error(None, response)
            cacheable = record.get("cacheable") if isinstance(record, dict) else None
            if not row["complete"] or shared and shared.get('lifecycle') != 'terminal':
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
                "physical_job_count": len(jobs),
                "retry_count": retries,
                "error": primary_error,
                "batch": _batch_progress(record, jobs) if isinstance(record, dict) else None,
            }
            calls.append(call)
        trusted_root = scope.audit_root / "trusted-jobs"
        for state in job_state_rows(trusted_root):
            identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
            physical_jobs.append(_physical_job(
                scope,
                state,
                source="trusted",
                record_id=None,
                operation=operation_name({"operation": identity.get("kind")}),
            ))
        for audit in sorted(scope.audit_root.glob("request-*.json")):
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
    error_counts = Counter(
        call["error"]["category"] for call in calls if isinstance(call.get("error"), dict)
    )
    timeout_counts = Counter()
    for row in physical_jobs:
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
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "summary": {
            "operation_counts": dict(sorted(operation_counts.items())),
            "trusted_operation_counts": dict(sorted(trusted_operation_counts.items())),
            "physical_job_status_counts": dict(sorted(job_counts.items())),
            "error_counts": dict(sorted(error_counts.items())),
            "timeout_counts": dict(sorted(timeout_counts.items())),
            "logical_call_count": len(calls),
            "physical_job_count": len(physical_jobs),
            "retry_count": sum(row["retry_count"] for row in physical_jobs),
            "duplicate_cache_hits": audit_duplicates,
            "same_request_running": audit_running,
        },
        "agent_paused_during_requests": pause,
        "availability": {
            "retry_backoff_seconds": (
                "actual sleep intervals are not durably persisted by current main; "
                "retry_count is available"
            ),
            "agent_paused_during_requests": (
                "reported only when the persisted search binding includes this setting"
            ),
        },
        "running_task_markers": running_markers,
        "pending_jobs": pending_jobs,
        "runtime_concurrency": {
            "unresolved_running_records": len(running_markers),
            "active_request_locks": None,
            "slot_limit": None,
            "waiting_queue_depth": None,
            "availability": (
                "live locks, slot limits and queue state are process-local and are not durably persisted; "
                "shared Experiment lifecycle and legacy task markers report unresolved Records, not lock ownership"
            ),
        },
        "calls": calls,
        "physical_jobs": physical_jobs,
    }


def render(value: dict) -> str:
    summary = value["summary"]
    lines = ["AKA Light Agate 调用与故障", ""]
    lines.append(table(
        ["Operation", "Calls"],
        [[name, count] for name, count in summary["operation_counts"].items()],
    ))
    lines += ["", f"Physical jobs: {summary['physical_job_count']}  Retries: {summary['retry_count']}  Duplicate: {summary['duplicate_cache_hits']}  Running duplicate: {summary['same_request_running']}", ""]
    lines.append(table(
        ["Record", "Operation", "Complete", "Jobs", "Retries", "Error"],
        [[row["gateway_record_id"], row["operation"], row["complete"], row["physical_job_count"],
          row["retry_count"], (row["error"] or {}).get("category")]
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
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-agate-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
