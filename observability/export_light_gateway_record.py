#!/usr/bin/env python3
"""Export one Gateway Record's Runtime, Agate, and Agent-visible evidence chain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.experiment.measurement_store import RECORD_ID  # noqa: E402
from observability._internal.common import (  # noqa: E402
    discover_scopes,
    job_attempt_rows,
    observed_at,
    read_json,
    resolve_workspace,
    runtime_audit_paths,
)
from observability._internal.gateway_record import (  # noqa: E402
    gateway_record_locations,
    runtime_request_ids as location_runtime_request_ids,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_audits(
    scope: Any, record_id: str, runtime_request_ids: set[str]
) -> list[dict]:
    rows = []
    for path in runtime_audit_paths(scope):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        raw_request_id = value.get("request_id") or path.stem.removeprefix("request-")
        if (
            value.get("gateway_record_id") != record_id
            and raw_request_id not in runtime_request_ids
        ):
            continue
        rows.append({"audit_path": str(path), **value})
    return rows


def _location(location: dict[str, Any]) -> dict:
    runtime_request_ids = location_runtime_request_ids(location)
    audits = []
    for scope in location.get("audit_scopes", [location["scope"]]):
        audits.extend(
            _runtime_audits(
                scope, Path(location["record_directory"]).name, runtime_request_ids
            )
        )
    runtime_request_ids.update(
        row.get("request_id")
        for row in audits
        if isinstance(row.get("request_id"), str) and row.get("request_id")
    )
    return {
        "scope": location["scope_name"],
        "trajectory": location["trajectory"],
        "workspace": location["workspace"],
        "record_directory": location["record_directory"],
        "runtime_request_ids": sorted(runtime_request_ids),
        "runtime_request": location["request"],
        "runtime_audits": audits,
        "gateway_record": location["record"],
        "agent_response": location["agent_response"],
        "agate_attempts": [
            {
                key: attempt[key]
                for key in (
                    "checkpoint_path",
                    "checkpoint_id",
                    "checkpoint_phase",
                    "submission",
                    "runtime_request_ids",
                    "agate_job_id",
                    "backend_agate_job_ids",
                    "status",
                    "retry_counters",
                    "agate_response",
                    "raw_process",
                    "snapshot_path",
                )
            }
            for attempt in location["attempts"]
        ],
    }


def _matches(workspace: Path, record_id: str) -> list[dict]:
    return [
        _location(location)
        for location in gateway_record_locations(
            workspace,
            record_id,
            discover=discover_scopes,
            attempt_reader=job_attempt_rows,
        )
    ]


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_value(matches: list[dict], field: str) -> object:
    """Return one immutable logical value or reject conflicting copies."""
    first = matches[0][field]
    expected = _stable_json(first)
    for match in matches[1:]:
        if _stable_json(match[field]) != expected:
            raise ValueError(
                "gateway_record_integrity_conflict: "
                f"{field} differs across persisted copies of the same Gateway Record"
            )
    return first


def _merged_runtime_audits(matches: list[dict]) -> list[dict]:
    by_identity: dict[str, tuple[str, dict]] = {}
    for match in matches:
        for audit in match["runtime_audits"]:
            public = dict(audit)
            audit_path = public.pop("audit_path", None)
            request_id = public.get("request_id")
            identity = (
                f"request:{request_id}"
                if isinstance(request_id, str) and request_id
                else "content:" + _stable_json(public)
            )
            encoded = _stable_json(public)
            previous = by_identity.get(identity)
            if previous is not None and previous[0] != encoded:
                raise ValueError(
                    "runtime_audit_integrity_conflict: "
                    f"{request_id or 'audit without request ID'} has inconsistent copies"
                )
            if previous is None:
                if isinstance(audit_path, str) and audit_path:
                    public["source_audit_paths"] = [audit_path]
                by_identity[identity] = (encoded, public)
            elif isinstance(audit_path, str) and audit_path:
                paths = previous[1].setdefault("source_audit_paths", [])
                if audit_path not in paths:
                    paths.append(audit_path)
    return sorted(
        (row for _, row in by_identity.values()),
        key=lambda row: (
            str(row.get("started_at") or row.get("created_at") or ""),
            str(row.get("request_id") or ""),
        ),
    )


def _merged_agate_attempts(matches: list[dict]) -> list[dict]:
    by_identity: dict[tuple[object, ...], tuple[str, dict]] = {}
    for match in matches:
        for attempt in match["agate_attempts"]:
            public = dict(attempt)
            snapshot_path = public.pop("snapshot_path", None)
            runtime_request_ids = sorted({
                value
                for value in public.pop("runtime_request_ids", [])
                if isinstance(value, str) and value
            })
            identity = (
                public.get("checkpoint_id"),
                public.get("submission"),
                public.get("agate_job_id"),
                public.get("checkpoint_phase"),
            )
            encoded = _stable_json(public)
            previous = by_identity.get(identity)
            if previous is not None and previous[0] != encoded:
                raise ValueError(
                    "agate_attempt_integrity_conflict: one physical Agate attempt "
                    "has inconsistent persisted copies"
                )
            if previous is None:
                public["runtime_request_ids"] = runtime_request_ids
                if isinstance(snapshot_path, str) and snapshot_path:
                    public["source_snapshot_paths"] = [snapshot_path]
                by_identity[identity] = (encoded, public)
                continue
            row = previous[1]
            row["runtime_request_ids"] = sorted({
                *row.get("runtime_request_ids", []),
                *runtime_request_ids,
            })
            if isinstance(snapshot_path, str) and snapshot_path:
                paths = row.setdefault("source_snapshot_paths", [])
                if snapshot_path not in paths:
                    paths.append(snapshot_path)
    return sorted(
        (row for _, row in by_identity.values()),
        key=lambda row: (
            str(row.get("checkpoint_path") or ""),
            row.get("submission") if isinstance(row.get("submission"), int) else -1,
            str(row.get("agate_job_id") or ""),
        ),
    )


def export_gateway_record(workspace: Path, record_id: str, output: Path) -> dict:
    workspace = resolve_workspace(workspace)
    if RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("--gateway-record-id must be a complete gateway- ID")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    matches = _matches(workspace, record_id)
    if not matches:
        raise ValueError("Gateway Record is not available in this Campaign")

    runtime_request = _canonical_value(matches, "runtime_request")
    gateway_record = _canonical_value(matches, "gateway_record")
    agent_response = _canonical_value(matches, "agent_response")
    runtime_audits = _merged_runtime_audits(matches)
    agate_attempts = _merged_agate_attempts(matches)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".aka-gateway-export-", dir=output.parent))
    try:
        _write(temporary / "runtime-request.json", runtime_request)
        _write(temporary / "runtime-audits.json", runtime_audits)
        _write(temporary / "gateway-record.json", gateway_record)
        _write(temporary / "agent-response.json", agent_response)
        _write(temporary / "agate-attempts.json", agate_attempts)

        primary_ids = sorted(
            {
                attempt["agate_job_id"]
                for attempt in agate_attempts
                if attempt["agate_job_id"]
            }
        )
        backend_ids = sorted(
            {
                backend_agate_job_id
                for attempt in agate_attempts
                for backend_agate_job_id in attempt["backend_agate_job_ids"]
            }
        )
        runtime_request_ids = sorted(
            {
                runtime_request_id
                for match in matches
                for runtime_request_id in match["runtime_request_ids"]
            }
        )
        manifest = {
            "schema_version": 2,
            "exported_at": observed_at(),
            "campaign_workspace": str(workspace),
            "gateway_record_id": record_id,
            "source_location_count": len(matches),
            "runtime_request_ids": runtime_request_ids,
            "agate_job_ids": primary_ids,
            "backend_agate_job_ids": backend_ids,
            "runtime_audit_count": len(runtime_audits),
            "agate_attempt_count": len(agate_attempts),
            "agent_response_available": agent_response is not None,
            "source_locations": [
                {
                    "scope": match["scope"],
                    "trajectory": match["trajectory"],
                    "workspace": match["workspace"],
                    "record_directory": match["record_directory"],
                    "runtime_request_ids": match["runtime_request_ids"],
                    "runtime_audit_count": len(match["runtime_audits"]),
                    "agate_attempt_count": len(match["agate_attempts"]),
                }
                for match in matches
            ],
        }
        _write(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    manifest["output_directory"] = str(output)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--gateway-record-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    try:
        value = export_gateway_record(
            options.workspace, options.gateway_record_id, options.output
        )
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        print(f"export-light-gateway-record: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
