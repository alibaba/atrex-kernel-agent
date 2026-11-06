#!/usr/bin/env python3
"""Export Kernel source, raw Agate jobs and Agent-visible response for one Record."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import job_identifier, job_state_rows, observed_at, resolve_workspace
from observability.export_light_source import (
    existing_stores,
    export_source,
    locate_gateway_record,
)
from aka.light.experiment.store import RECORD_ID  # noqa: E402
from aka.light.runtime.paths import supervisor_campaign_root  # noqa: E402


def _record(workspace: Path, campaign_root: Path, record_id: str):
    matches = []
    for owner, store in existing_stores(workspace, campaign_root):
        try:
            record = store.read(record_id)
        except (FileNotFoundError, OSError, ValueError):
            continue
        matches.append((owner, store, record))
    if not matches:
        raise ValueError("Gateway Record is not available in this Campaign")
    digests = {row[2].get("result_digest") for row in matches}
    if len(digests) != 1:
        raise ValueError("Gateway Record resolves to inconsistent results")
    return matches[0], matches


def export_evidence(workspace: Path, record_id: str, output: Path) -> dict:
    workspace = resolve_workspace(workspace)
    if RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("--gateway-record-id must be a complete gateway- ID")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    campaign_root = supervisor_campaign_root(workspace)
    (owner, store, record), locations = _record(workspace, campaign_root, record_id)
    located = locate_gateway_record(workspace, campaign_root, record_id)
    temporary = Path(tempfile.mkdtemp(prefix=".aka-evidence-export-", dir=output.parent))
    try:
        source_manifest = export_source(located, temporary / "kernel")
        source_manifest["output_directory"] = "kernel"
        rows = []
        seen = set()
        for matched_owner, matched_store, _ in locations:
            directory = matched_store.root / "records" / record_id
            for state in job_state_rows(directory):
                job = state["job"]
                key = json.dumps(job, sort_keys=True, ensure_ascii=False) if job is not None else state["path"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "workspace": str(matched_owner),
                        "job_id": job_identifier(job),
                        "checkpoint_phase": state["phase"],
                        "submission": state["submission"],
                        "retries": state["retries"],
                        "agate_response": job,
                        "raw_process": state["process"],
                    }
                )
        (temporary / "agent-response.json").write_text(
            json.dumps(record["response"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "agate-jobs.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "exported_at": observed_at(),
            "campaign_workspace": str(workspace),
            "gateway_record_id": record_id,
            "operation": record.get("operation"),
            "created_at": record.get("created_at"),
            "cacheable": record.get("cacheable"),
            "kernel": record.get("kernels", {}).get("candidate"),
            "agate_job_ids": sorted({row["job_id"] for row in rows if row["job_id"]}),
            "agate_job_count": len(rows),
            "files": {
                "kernel": "kernel/",
                "raw_agate_jobs": "agate-jobs.json",
                "agent_visible_response": "agent-response.json",
            },
            "source_manifest": source_manifest,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    try:
        value = export_evidence(options.workspace, options.gateway_record_id, options.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"export-light-evidence: {error}", file=__import__("sys").stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
