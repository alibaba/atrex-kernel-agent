"""Exact candidate-source Git history exchange between isolated Agent repositories."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from aka.core.filesystem import durable_write_json


def _run(workspace: Path, *args: str, input_bytes: bytes | None = None,
         check: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["git", *args], cwd=workspace, env=environment, input=input_bytes,
        capture_output=True, check=check,
    )


def _validate_source_history(workspace: Path, ref: str) -> int:
    """Ensure an exact-ID bundle contains only valid candidate-source commits."""
    from aka.task.source.formats import Artifact, from_git

    commits = _run(workspace, "rev-list", ref).stdout.decode().splitlines()
    for commit in commits:
        artifact = Artifact(from_git(workspace, commit))
        allowed = {*artifact.materialized_files, ".gitignore"}
        tree = set(
            _run(workspace, "ls-tree", "-r", "--name-only", commit)
            .stdout.decode()
            .splitlines()
        )
        unexpected = sorted(tree - allowed)
        if unexpected:
            raise ValueError(
                "Shared Git history contains non-Kernel paths: "
                + ", ".join(unexpected[:8])
            )
    return len(commits)


def export_publication(
    workspace: Path,
    destination: Path,
    *,
    trajectory: int,
    publication: int,
    lineage_winner_commit_id: str | None = None,
    audit_path: Path,
) -> dict:
    """Export one immutable publication without changing the source repository.

    The Agent-facing ref is trajectory-stable rather than publication-stable
    and always points at the latest imported HEAD. Publication and
    lineage-winner identity remain private audit/research metadata.
    """
    if type(publication) is not int or publication < 1:
        raise ValueError("Peer Git publication must be a positive integer")
    head = _run(workspace, "rev-parse", "HEAD").stdout.decode().strip()
    source_ref = f"refs/remotes/atrex/T{trajectory}"
    if lineage_winner_commit_id:
        if _run(
            workspace,
            "cat-file",
            "-e",
            f"{lineage_winner_commit_id}^{{commit}}",
            check=False,
        ).returncode != 0:
            raise ValueError("Lineage winner is not present in Agent Git")
    if destination.exists() or audit_path.exists():
        if not destination.is_file() or not audit_path.is_file():
            raise RuntimeError("Peer Git publication source is incomplete")
        try:
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Peer Git publication receipt is unreadable") from error
        immutable = {
            "schema_version": 3,
            "trajectory": trajectory,
            "publication": publication,
            "source_ref": source_ref,
            "original_agent_commit": head,
            "lineage_winner_commit_id": lineage_winner_commit_id,
        }
        if not isinstance(existing, dict) or any(
            existing.get(name) != value for name, value in immutable.items()
        ):
            raise RuntimeError("Peer Git publication source changed during recovery")
        heads = _run(
            workspace,
            "bundle",
            "list-heads",
            str(destination),
            check=False,
        )
        expected_head = f"{head} {source_ref}"
        if heads.returncode != 0 or expected_head not in heads.stdout.decode().splitlines():
            raise RuntimeError("Peer Git publication bundle lost its published HEAD")
        if existing.get("bytes") != destination.stat().st_size:
            raise RuntimeError("Peer Git publication bundle size changed during recovery")
        return existing
    # The Agent owns every ref in its repository. Build the namespaced export
    # in a private temporary repository instead of writing an export ref into
    # Agent Git.
    commit_count = _validate_source_history(workspace, head)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="aka-shared-export-") as private:
        private_repo = Path(private) / "repo"
        private_repo.mkdir()
        _run(private_repo, "init", "--bare")
        fetched = _run(
            private_repo,
            "-c",
            "protocol.file.allow=always",
            "fetch",
            str(workspace),
            f"+{head}:{source_ref}",
            check=False,
        )
        if fetched.returncode != 0:
            raise RuntimeError(fetched.stderr.decode(errors="replace").strip())
        result = _run(
            private_repo,
            "bundle",
            "create",
            str(temporary),
            source_ref,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
    size = temporary.stat().st_size
    os.replace(temporary, destination)
    receipt = {
        "schema_version": 3,
        "trajectory": trajectory,
        "publication": publication,
        "source_ref": source_ref,
        "original_agent_commit": head,
        "lineage_winner_commit_id": lineage_winner_commit_id,
        "bytes": size,
        "commit_count": commit_count,
    }
    durable_write_json(audit_path, receipt)
    return receipt


def import_exports_into_repository(
    workspace: Path,
    exports: Iterable[Path],
    *,
    notes: Mapping[str, Mapping[str, object]] | None = None,
    audit_path: Path,
) -> dict:
    """Fetch peer publications directly into one persistent Agent repository.

    Only Runtime-owned remote-tracking refs and measurement notes are updated.
    Local branches, the index, the worktree and HEAD are never changed.  The
    repository itself is durable, so no aggregate recovery bundle is needed.
    """
    exports = tuple(exports)
    if not (workspace / ".git").is_dir():
        raise ValueError("Peer Git import requires a persistent Agent repository")
    try:
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing_audit = None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Peer Git import receipt is unreadable") from error
    imported: list[str] = []
    mappings: list[dict[str, object]] = []
    for path in exports:
        audit = path.parent / "receipt.json"
        try:
            metadata = json.loads(audit.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        trajectory = metadata.get("trajectory")
        publication = metadata.get("publication")
        destination_ref = (
            f"refs/remotes/atrex/T{trajectory}"
            if type(trajectory) is int and trajectory > 0
            else None
        )
        source_ref = metadata.get("source_ref")
        original = metadata.get("original_agent_commit")
        if (
            metadata.get("schema_version") != 3
            or destination_ref is None
            or type(publication) is not int
            or publication < 1
            or not isinstance(source_ref, str)
            or not source_ref.startswith("refs/remotes/atrex/")
            or not isinstance(original, str)
        ):
            raise ValueError("Shared Git bundle has no valid namespaced source refs")
        if existing_audit is None:
            result = _run(
                workspace,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                str(path),
                f"+{source_ref}:{destination_ref}",
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
            resolved = _run(workspace, "rev-parse", destination_ref, check=False)
            imported_commit = (
                resolved.stdout.decode().strip() if resolved.returncode == 0 else None
            )
            if imported_commit != original:
                raise ValueError(
                    "Shared Git import changed the peer commit ID; Kernel identity is invalid"
                )
        else:
            imported_commit = original
        imported.append(f"T{trajectory}@p{publication:08d}")
        mappings.append(
            {
                "trajectory": trajectory,
                "publication": publication,
                "source_ref": destination_ref,
                "original_agent_commit": original,
                "imported_commit": imported_commit,
            }
        )
    if existing_audit is not None:
        expected = {
            "schema_version": 3,
            "imports": imported,
            "commit_mappings": mappings,
            "shared_refs": sorted(
                mapping["source_ref"] for mapping in mappings
            ),
        }
        if existing_audit != expected:
            raise RuntimeError("Peer Git import receipt changed during recovery")
        for mapping in mappings:
            present = _run(
                workspace,
                "cat-file",
                "-e",
                f"{mapping['imported_commit']}^{{commit}}",
                check=False,
            )
            if present.returncode != 0:
                raise RuntimeError("Persistent Agent Git lost previously imported history")
        return existing_audit
    for ref, note in (notes or {}).items():
        target = _run(workspace, "rev-parse", ref, check=False)
        if target.returncode:
            continue
        _run(
            workspace,
            "notes",
            "--ref=atrex-measurements",
            "add",
            "-f",
            "-m",
            json.dumps(note, ensure_ascii=False, sort_keys=True),
            target.stdout.decode().strip(),
        )
    receipt = {
        "schema_version": 3,
        "imports": imported,
        "commit_mappings": mappings,
        "shared_refs": sorted(mapping["source_ref"] for mapping in mappings),
    }
    durable_write_json(audit_path, receipt)
    return receipt
