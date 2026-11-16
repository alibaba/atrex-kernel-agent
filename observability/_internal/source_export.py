"""Materialize a measured or accepted Light Kernel source artifact."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aka.experiment.measurement_store import MeasurementStore  # noqa: E402
from aka.workflow.light.support.paths import supervisor_measurement_root  # noqa: E402
from aka.task.source.formats import Artifact, from_git  # noqa: E402
from aka.task.source.formats.registry import for_policy  # noqa: E402

from observability._internal.measurement_sources import measurement_workspaces  # noqa: E402


GIT_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


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
    root = supervisor_measurement_root(workspace)
    if (root / "identity.key").is_file():
        yield workspace, MeasurementStore(root)


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
        # This is a read-only evidence export, not an Agent workspace.  The
        # ordinary Artifact.install path binds a private SourceProvider contract
        # and therefore creates a sibling .atrex-supervisor-runtime directory.
        # Materialize the already validated artifact through its file-system
        # implementation without creating Campaign state.
        for_policy(artifact.contract).install(
            temporary,
            artifact.files,
            artifact.contract,
        )
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
