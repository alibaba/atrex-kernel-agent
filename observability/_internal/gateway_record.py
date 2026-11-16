"""Canonical Gateway Record and physical Agate-attempt projections."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from aka.evaluation.tool_results import compact_receipt
from observability._internal.common import (
    discover_scopes,
    existing_store,
    job_attempt_rows,
    job_error_details,
    job_identifier,
    normalized_job_status,
    operation_name,
    read_json,
    runtime_audit_paths,
)


def shape_ids(identity: object) -> list[str] | None:
    if not isinstance(identity, dict):
        return None
    direct = identity.get("shape_ids")
    if isinstance(direct, list):
        return [str(item) for item in direct]
    payload = identity.get("payload")
    if isinstance(payload, dict):
        for owner in (payload, payload.get("options"), payload.get("overrides")):
            if isinstance(owner, dict) and isinstance(owner.get("shape_ids"), list):
                return [str(item) for item in owner["shape_ids"]]
    return None


def backend_agate_job_ids(job: object) -> list[str]:
    if not isinstance(job, dict):
        return []
    values: list[str] = []
    for owner in (job, job.get("result")):
        if not isinstance(owner, dict):
            continue
        raw = owner.get("backend_job_ids")
        if isinstance(raw, list):
            values.extend(
                str(item) for item in raw if isinstance(item, str) and item
            )
    return list(dict.fromkeys(values))


def agent_visible_response(response: object) -> dict | None:
    """Project saved evidence using the current receipt contract as a fallback."""
    if not isinstance(response, dict):
        return None
    agent_result = response.get("agent_result")
    if not isinstance(agent_result, dict):
        # Legacy Records persisted the already-public response directly.
        return response
    return compact_receipt(agent_result)


def project_attempt(state: dict[str, Any]) -> dict[str, Any]:
    job = state.get("job")
    identity = state.get("identity")
    sdk_payload = identity.get("payload") if isinstance(identity, dict) else None
    reused = (
        state.get("submission") == 0
        and isinstance(identity, dict)
        and bool(identity.get("batch_checkpoint"))
    )
    return {
        "checkpoint_path": state.get("checkpoint_path"),
        "checkpoint_id": state.get("checkpoint_id"),
        "checkpoint_phase": state.get("phase"),
        "submission": state.get("submission"),
        "runtime_request_ids": state.get("runtime_request_ids", []),
        "agate_job_id": job_identifier(job),
        "backend_agate_job_ids": backend_agate_job_ids(job),
        "raw_status": job.get("status") if isinstance(job, dict) else None,
        "status": normalized_job_status(state),
        "reused": reused,
        "shape_ids": shape_ids(identity),
        "retry_counters": state.get("retries") or {},
        "error": job_error_details(job),
        "snapshot_path": state.get("path"),
        "agate_sdk_submission_payload": sdk_payload,
        "agate_response": job,
        "raw_process": state.get("process"),
    }


def gateway_record_locations(
    workspace: Path,
    record_id: str,
    *,
    discover: Callable[[Path], Iterable[Any]] = discover_scopes,
    attempt_reader: Callable[[Path], list[dict[str, Any]]] = job_attempt_rows,
) -> list[dict[str, Any]]:
    """Locate one canonical Record and parse every physical attempt once."""
    locations = []
    scopes = list(discover(workspace))
    grouped: dict[Path, list[Any]] = {}
    for scope in scopes:
        directory = scope.measurement_root / "records" / record_id
        if not directory.is_dir():
            continue
        grouped.setdefault(directory.resolve(), []).append(scope)
    for directory, source_scopes in grouped.items():
        request = read_json(directory / "request.json", default={})
        request = request if isinstance(request, dict) else {}
        record_path = directory / "record.json"
        record: dict[str, Any] = {}
        record_read_error = None
        store = existing_store(source_scopes[0])
        if record_path.is_file() and store is not None:
            try:
                stored = store.read(record_id)
            except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                record_read_error = str(error)
            else:
                record = stored if isinstance(stored, dict) else {}
        elif record_path.is_file():
            stored = read_json(record_path, default={})
            record = stored if isinstance(stored, dict) else {}
        embedded_request = record.get("request")
        if isinstance(embedded_request, dict):
            request = embedded_request
        operation = operation_name(record or request)
        attempts = [project_attempt(state) for state in attempt_reader(directory)]
        runtime_request_ids = {
            value
            for value in (
                record.get("runtime_request_id"),
                request.get("runtime_request_id"),
            )
            if isinstance(value, str) and value
        }
        for attempt in attempts:
            runtime_request_ids.update(
                value
                for value in attempt.get("runtime_request_ids", [])
                if isinstance(value, str) and value
            )
        audit_scopes = []
        for scope in scopes:
            matched = False
            for path in runtime_audit_paths(scope):
                audit = read_json(path, default={})
                if not isinstance(audit, dict):
                    continue
                request_id = audit.get("request_id") or path.stem.removeprefix("request-")
                if (
                    audit.get("gateway_record_id") == record_id
                    or request_id in runtime_request_ids
                ):
                    matched = True
                    break
            if matched:
                audit_scopes.append(scope)
        owner = next(
            (scope for scope in audit_scopes if scope.trajectory is not None),
            audit_scopes[0] if audit_scopes else source_scopes[0],
        )
        locations.append(
            {
                "scope": owner,
                "scope_name": owner.label,
                "trajectory": owner.trajectory,
                "workspace": str(owner.workspace),
                "audit_scopes": audit_scopes or source_scopes,
                "source_scopes": source_scopes,
                "record_directory": str(directory),
                "operation": operation,
                "complete": (directory / "record.json").is_file(),
                "record_read_error": record_read_error,
                "runtime_request_id": record.get("runtime_request_id"),
                "request": request,
                "record": record or None,
                "agent_response": (
                    agent_visible_response(record.get("response")) if record else None
                ),
                "attempts": attempts,
            }
        )
    return locations


def runtime_request_ids(location: dict[str, Any]) -> set[str]:
    request = location.get("request")
    request = request if isinstance(request, dict) else {}
    record = location.get("record")
    record = record if isinstance(record, dict) else {}
    values = {
        value
        for value in (
            record.get("runtime_request_id"),
            request.get("runtime_request_id"),
        )
        if isinstance(value, str) and value
    }
    for attempt in location.get("attempts", []):
        values.update(
            value
            for value in attempt.get("runtime_request_ids", [])
            if isinstance(value, str) and value
        )
    return values
