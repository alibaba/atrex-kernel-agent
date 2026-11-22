#!/usr/bin/env python3
"""Export one Agent commit, all bound measurements, and raw Agate identities."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.workflow.light.support.paths import supervisor_campaign_root
from observability._internal.common import (
    discover_scopes,
    evaluation_result,
    iter_records,
    job_attempt_rows,
    job_identifier,
    observed_at,
    operation_name,
    resolve_workspace,
)
from observability._internal.source_export import export_source, locate_git_commit
from observability._internal.submissions import campaign_submission_snapshot, eligible_full_eval
from observability._internal.view_helpers import evidence_health


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _agate_job_ids(job: object) -> tuple[str | None, list[str]]:
    if not isinstance(job, dict):
        return None, []
    primary = job_identifier(job)
    backend_values = []
    for owner in (job, job.get("result")):
        if not isinstance(owner, dict):
            continue
        backend = owner.get("backend_job_ids")
        if isinstance(backend, list):
            backend_values.extend(
                str(item) for item in backend if isinstance(item, str) and item
            )
    return primary, list(dict.fromkeys(backend_values))


def export_kernel(workspace: Path, commit_id: str, output: Path) -> dict:
    workspace = resolve_workspace(workspace)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    located = locate_git_commit(workspace, supervisor_campaign_root(workspace), commit_id)
    temporary = Path(tempfile.mkdtemp(prefix=".aka-kernel-export-", dir=output.parent))
    measurements = []
    all_primary_job_ids: set[str] = set()
    all_backend_agate_job_ids: set[str] = set()
    try:
        source_manifest = export_source(located, temporary / "kernel")
        source_manifest["output_directory"] = "kernel"
        for scope in discover_scopes(workspace):
            for item in iter_records(scope):
                record = item["record"]
                candidate = (record.get("kernels") or {}).get("candidate")
                if not isinstance(candidate, dict) or candidate.get("kernel_id") != commit_id:
                    continue
                record_id = item["gateway_record_id"]
                attempts = []
                for state in job_attempt_rows(item["directory"]):
                    job = state.get("job")
                    primary_id, backend_ids = _agate_job_ids(job)
                    if primary_id:
                        all_primary_job_ids.add(primary_id)
                    all_backend_agate_job_ids.update(backend_ids)
                    attempts.append(
                        {
                            "checkpoint_path": state.get("checkpoint_path"),
                            "checkpoint_phase": state.get("phase"),
                            "submission": state.get("submission"),
                            "agate_job_id": primary_id,
                            "backend_agate_job_ids": backend_ids,
                            "retries": state.get("retries"),
                            "runtime_request_ids": state.get("runtime_request_ids", []),
                            "agate_response": job,
                            "raw_process": state.get("process"),
                            "snapshot_path": state.get("path"),
                        }
                    )
                directory = temporary / "measurements" / record_id
                _write(directory / "record.json", record)
                _write(directory / "request.json", item.get("request") or {})
                _write(directory / "agent-response.json", record.get("response") or {})
                _write(directory / "agate-jobs.json", attempts)
                result = evaluation_result(record) or {}
                error = result.get("error")
                if not isinstance(error, dict):
                    error = None
                measurements.append(
                    {
                        "scope": scope.label,
                        "trajectory": scope.trajectory,
                        "gateway_record_id": record_id,
                        "operation": operation_name(record),
                        "created_at": record.get("created_at"),
                        "passing_standard_full_evaluate": eligible_full_eval(record) is not None,
                        "runtime_request_id": record.get("runtime_request_id"),
                        "result_summary": {
                            "all_pass": result.get("all_pass"),
                            "latency_us_geomean": result.get("latency_us_geomean"),
                            "latency_us_arith_mean": result.get("latency_us_arith_mean"),
                            "shape_count": len(result.get("latency_us_by_shape") or {}),
                            "performance_objective": result.get("performance_objective"),
                            "status": result.get("status") or result.get("outcome"),
                            "error": error,
                        },
                        "agate_job_ids": sorted(
                            {
                                row["agate_job_id"]
                                for row in attempts
                                if row["agate_job_id"]
                            }
                        ),
                        "backend_agate_job_ids": sorted(
                            {
                                value
                                for row in attempts
                                for value in row["backend_agate_job_ids"]
                            }
                        ),
                        "directory": f"measurements/{record_id}",
                    }
                )
        submissions = campaign_submission_snapshot(workspace)
        matching_submissions = [
            row for row in submissions["submissions"] if row.get("kernel_commit_id") == commit_id
        ]
        manifest = {
            "schema_version": 1,
            "exported_at": observed_at(),
            "campaign_workspace": str(workspace),
            "kernel_commit_id": commit_id,
            "source_manifest": source_manifest,
            "measurement_count": len(measurements),
            "measurements": measurements,
            "agate_job_ids": sorted(all_primary_job_ids),
            "backend_agate_job_ids": sorted(all_backend_agate_job_ids),
            "kernel_submissions": matching_submissions,
            "evidence_health": evidence_health(workspace, submissions),
            "files": {"kernel": "kernel/", "measurements": "measurements/"},
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
    parser.add_argument("--kernel-commit-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    try:
        value = export_kernel(options.workspace, options.kernel_commit_id, options.output)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        print(f"export-light-kernel: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
