#!/usr/bin/env python3
"""List every Agate submission belonging to one logical Gateway Record."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aka.light.experiment.store import RECORD_ID  # noqa: E402
from observability.common import (  # noqa: E402
    add_output_arguments,
    discover_scopes,
    emit,
    job_attempt_rows,
    job_error_details,
    job_identifier,
    main_error,
    observed_at,
    operation_name,
    normalized_job_status,
    read_json,
    table,
)


def _shape_ids(identity: object) -> list[str] | None:
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


def _backend_job_ids(job: object) -> list[str]:
    if not isinstance(job, dict):
        return []
    values = []
    for owner in (job, job.get("result")):
        if not isinstance(owner, dict):
            continue
        raw = owner.get("backend_job_ids")
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if isinstance(item, str) and item)
    return list(dict.fromkeys(values))


def snapshot(workspace: Path, record_id: str) -> dict[str, Any]:
    if RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("--gateway-record-id must be a complete gateway- ID")
    locations = []
    attempts = []
    for scope in discover_scopes(workspace):
        directory = scope.measurement_root / "records" / record_id
        if not directory.is_dir():
            continue
        record = read_json(directory / "record.json", default={})
        request = read_json(directory / "request.json", default={})
        operation = operation_name(
            record if isinstance(record, dict) and record else request
        )
        rows = job_attempt_rows(directory)
        locations.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "workspace": str(scope.workspace),
                "record_directory": str(directory),
                "operation": operation,
                "complete": (directory / "record.json").is_file(),
                "submission_attempt_count": len(rows),
            }
        )
        for state in rows:
            job = state["job"]
            status = normalized_job_status(state)
            identity = state.get("identity")
            reused = (
                state["submission"] == 0
                and isinstance(identity, dict)
                and bool(identity.get("batch_checkpoint"))
            )
            attempts.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "operation": operation,
                    "checkpoint_path": state["checkpoint_path"],
                    "checkpoint_id": state["checkpoint_id"],
                    "submission": state["submission"],
                    "checkpoint_phase": state["phase"],
                    "job_id": job_identifier(job),
                    "raw_status": (
                        job.get("status") if isinstance(job, dict) else None
                    ),
                    "status": status,
                    "reused": reused,
                    "shape_ids": _shape_ids(identity),
                    "backend_job_ids": _backend_job_ids(job),
                    "retry_counters": state["retries"],
                    "error": job_error_details(job),
                    "snapshot_path": state["path"],
                }
            )
    if not locations:
        raise ValueError("Gateway Record is not available in this Campaign")
    status_counts = Counter(row["status"] for row in attempts)
    physical_ids = {
        item
        for row in attempts
        for item in [row["job_id"], *row["backend_job_ids"]]
        if item
    }
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(workspace.expanduser().resolve()),
        "gateway_record_id": record_id,
        "summary": {
            "record_location_count": len(locations),
            "submission_attempt_count": len(attempts),
            "physical_job_id_count": len(physical_ids),
            "reused_attempt_count": sum(row["reused"] for row in attempts),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "locations": locations,
        "attempts": attempts,
    }


def render(value: dict[str, Any]) -> str:
    summary = value["summary"]
    lines = [
        f"Gateway Record: {value['gateway_record_id']}",
        (
            f"Attempts: {summary['submission_attempt_count']}  "
            f"Physical Job IDs: {summary['physical_job_id_count']}  "
            f"Status: {summary['status_counts']}"
        ),
        "",
        table(
            [
                "Scope",
                "Checkpoint",
                "Try",
                "Job ID",
                "Status",
                "Reused",
                "Error",
                "Shapes",
            ],
            [
                [
                    row["scope"],
                    row["checkpoint_path"],
                    row["submission"],
                    row["job_id"],
                    row["status"],
                    row["reused"],
                    (row["error"] or {}).get("reason"),
                    ",".join(row["shape_ids"] or []),
                ]
                for row in value["attempts"]
            ],
        ),
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--gateway-record-id", required=True)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace, options.gateway_record_id)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-gateway-jobs", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
