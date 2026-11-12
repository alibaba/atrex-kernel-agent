#!/usr/bin/env python3
"""Export a measured or accepted Light Kernel source artifact."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aka.light.experiment.store import (  # noqa: E402
    RECORD_ID,
    MeasurementStore,
)
from aka.light.runtime.paths import supervisor_campaign_root  # noqa: E402
from aka.light.workspace.sources import Artifact, from_git  # noqa: E402

from observability.light_latency_status import measurement_workspaces  # noqa: E402


GIT_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
STATUS_SELECTIONS = ("accepted_best", "best_passing_full_eval")


@dataclass(frozen=True)
class LocatedSource:
    snapshot: bytes
    identity: dict
    locations: tuple[str, ...]


def _git_has_commit(workspace: Path, commit: str) -> bool:
    if not (workspace / ".git").exists():
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _unique_source(matches: list[tuple[bytes, str]], identity: dict) -> LocatedSource:
    if not matches:
        raise ValueError("No matching source artifact exists in this Campaign")
    by_digest: dict[str, list[tuple[bytes, str]]] = {}
    for source, location in matches:
        by_digest.setdefault(hashlib.sha256(source).hexdigest(), []).append((source, location))
    if len(by_digest) != 1:
        raise ValueError("The same identity resolves to different source artifacts")
    rows = next(iter(by_digest.values()))
    return LocatedSource(rows[0][0], identity, tuple(sorted({location for _, location in rows})))


def locate_git_commit(workspace: Path, campaign_root: Path, commit: str) -> LocatedSource:
    if GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("--git-commit must be a complete 40- or 64-character lowercase Git commit")
    matches = []
    for owner, store in existing_stores(workspace, campaign_root):
        try:
            matches.append((store.read_kernel(commit), str(owner)))
        except (FileNotFoundError, OSError, ValueError):
            continue
    for owner in measurement_workspaces(workspace, campaign_root):
        if not _git_has_commit(owner, commit):
            continue
        try:
            matches.append((from_git(owner, commit), str(owner)))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return _unique_source(matches, {"kernel_commit_id": commit})


def existing_stores(workspace: Path, campaign_root: Path):
    for owner in measurement_workspaces(workspace, campaign_root):
        root = supervisor_campaign_root(owner) / "measurements"
        if (root / "identity.key").is_file():
            yield owner, MeasurementStore(root)


def locate_gateway_record(workspace: Path, campaign_root: Path, record_id: str) -> LocatedSource:
    if RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("--gateway-record-id must be a complete gateway- ID")
    matches = []
    kernel_commit_ids = set()
    artifact_digests = set()
    for owner, store in existing_stores(workspace, campaign_root):
        try:
            record = store.read(record_id)
            identity = record["kernels"]["candidate"]
            source = store.read_kernel(identity["kernel_id"])
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            continue
        kernel_commit_ids.add(identity["kernel_id"])
        artifact_digests.add(identity["kernel_artifact_digest"])
        matches.append((source, str(owner)))
    if len(kernel_commit_ids) > 1 or len(artifact_digests) > 1:
        raise ValueError("Gateway Record resolves to inconsistent candidate identities")
    return _unique_source(
        matches,
        {
            "gateway_record_id": record_id,
            "kernel_commit_id": next(iter(kernel_commit_ids), None),
            "kernel_artifact_digest": next(iter(artifact_digests), None),
        },
    )


def read_status(path: str) -> dict:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Status input must be a JSON object")
    return value


def locate_from_status(
    workspace: Path,
    campaign_root: Path,
    status: dict,
    selection: str,
) -> LocatedSource:
    value = status.get(selection)
    if not isinstance(value, dict):
        raise ValueError(f"Status has no {selection} result to export")
    if selection == "accepted_best":
        commit = value.get("kernel_commit_id")
        if not isinstance(commit, str):
            raise ValueError("accepted_best has no kernel_commit_id")
        return locate_git_commit(workspace, campaign_root, commit)

    record_id = value.get("gateway_record_id")
    if isinstance(record_id, str):
        return locate_gateway_record(workspace, campaign_root, record_id)
    commit = value.get("kernel_commit_id")
    if isinstance(commit, str):
        return locate_git_commit(workspace, campaign_root, commit)
    raise ValueError(
        "best_passing_full_eval has neither gateway_record_id nor kernel_commit_id"
    )


def source_kind(artifact: Artifact) -> str:
    if artifact.contract is None:
        return "single_file"
    if artifact.layout.framework == "cuda":
        return "cuda_manifest"
    return "source_tree"


def export_source(located: LocatedSource, destination: Path) -> dict:
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".aka-source-export-", dir=destination.parent))
    artifact = Artifact(located.snapshot)
    try:
        artifact.install(temporary, replace=False, bind=True)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "schema_version": 2,
        "output_directory": str(destination),
        "source_kind": source_kind(artifact),
        "source_snapshot_sha256": artifact.digest,
        "files": sorted(artifact.materialized_files),
        "identity": located.identity,
        "found_in_workspaces": list(located.locations),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an accepted or measured Light Kernel source package."
    )
    parser.add_argument("workspace", type=Path, help="top-level Light Campaign workspace")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument(
        "--kernel-commit-id", "--git-commit", dest="kernel_commit_id",
        help="Agent Git commit ID (the --git-commit spelling is retained as an alias)",
    )
    identity.add_argument("--gateway-record-id")
    identity.add_argument(
        "--status-json",
        metavar="PATH",
        help="JSON produced by light_latency_status.py, or - for stdin",
    )
    parser.add_argument(
        "--select",
        choices=STATUS_SELECTIONS,
        default="accepted_best",
        help="result to export when --status-json is used (default: accepted_best)",
    )
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    options = parser.parse_args(argv)
    if options.status_json is None and options.select != "accepted_best":
        parser.error("--select is only meaningful with --status-json")
    return options


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    workspace = options.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"export-light-source: Campaign workspace does not exist: {workspace}", file=sys.stderr)
        return 1
    campaign_root = supervisor_campaign_root(workspace)
    try:
        if options.kernel_commit_id is not None:
            located = locate_git_commit(workspace, campaign_root, options.kernel_commit_id)
        elif options.gateway_record_id is not None:
            located = locate_gateway_record(
                workspace, campaign_root, options.gateway_record_id
            )
        else:
            located = locate_from_status(
                workspace,
                campaign_root,
                read_status(options.status_json),
                options.select,
            )
        result = export_source(located, options.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"export-light-source: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
