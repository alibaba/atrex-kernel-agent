"""Exact candidate-source Git history exchange between isolated Agent repositories."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from aka.legacy.task.persistence import durable_write_json

def _run(workspace: Path, *args: str, input_bytes: bytes | None = None,
         check: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["git", *args], cwd=workspace, env=environment, input=input_bytes,
        capture_output=True, check=check,
    )


def _validate_source_history(workspace: Path, ref: str) -> int:
    """Ensure an exact-ID bundle contains only valid candidate-source commits."""
    from aka.light.workspace.sources import Artifact, from_git

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


def export_epoch(
    workspace: Path,
    destination: Path,
    *,
    trajectory: int,
    epoch: int,
    lineage_winner_commit_id: str | None = None,
    audit_path: Path,
) -> dict:
    """Export exact HEAD and submitted-winner objects for peer inspection."""
    head = _run(workspace, "rev-parse", "HEAD").stdout.decode().strip()
    refs = {"head": f"refs/atrex/shared/T{trajectory}/e{epoch:04d}/head"}
    commits = {"head": head}
    if lineage_winner_commit_id:
        if _run(
            workspace,
            "cat-file",
            "-e",
            f"{lineage_winner_commit_id}^{{commit}}",
            check=False,
        ).returncode != 0:
            raise ValueError("Lineage winner is not present in Agent Git")
        refs["lineage_winner"] = (
            f"refs/atrex/shared/T{trajectory}/e{epoch:04d}/lineage-winner"
        )
        commits["lineage_winner"] = lineage_winner_commit_id
    # The Agent owns every ref in its repository. Build the namespaced export
    # in a private temporary repository instead of writing an export ref into
    # Agent Git.
    for commit_id in set(commits.values()):
        _validate_source_history(workspace, commit_id)
    commit_count = len(
        set(
            _run(workspace, "rev-list", *sorted(set(commits.values())))
            .stdout.decode()
            .splitlines()
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="aka-shared-export-") as private:
        private_repo = Path(private) / "repo"
        private_repo.mkdir()
        _run(private_repo, "init", "--bare")
        for label, ref in refs.items():
            fetched = _run(
                private_repo,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                str(workspace),
                f"+{commits[label]}:{ref}",
                check=False,
            )
            if fetched.returncode != 0:
                raise RuntimeError(fetched.stderr.decode(errors="replace").strip())
        result = _run(
            private_repo,
            "bundle",
            "create",
            str(temporary),
            *refs.values(),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
    size = temporary.stat().st_size
    os.replace(temporary, destination)
    receipt = {
        "schema_version": 1,
        "trajectory": trajectory,
        "epoch": epoch,
        "source_ref": refs["head"],
        "source_refs": refs,
        "original_agent_commit": head,
        "original_agent_commits": commits,
        "bytes": size,
        "commit_count": commit_count,
    }
    durable_write_json(audit_path, receipt)
    return receipt


def import_exports(
    bundle: Path,
    exports: Iterable[Path],
    *,
    notes: Mapping[str, Mapping[str, object]] | None = None,
    audit_path: Path,
) -> dict:
    """Import exact, source-scoped peer commits without exposing peer repositories."""
    exports = tuple(exports)
    if not exports:
        return {"schema_version": 1, "imports": []}
    with tempfile.TemporaryDirectory(prefix="aka-shared-git-") as temporary:
        workspace = Path(temporary) / "repo"
        workspace.mkdir()
        _run(workspace, "init", "-b", "work")
        _run(workspace, "config", "user.name", "Atrex Runtime")
        _run(workspace, "config", "user.email", "runtime@localhost")
        if bundle.is_file():
            _run(
                workspace,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                str(bundle),
                "+refs/heads/work:refs/remotes/atrex-local/work",
                "+refs/tags/*:refs/tags/*",
                "+refs/atrex/*:refs/atrex/*",
                "+refs/notes/*:refs/notes/*",
            )
            if _run(
                workspace,
                "show-ref",
                "--verify",
                "refs/remotes/atrex-local/work",
                check=False,
            ).returncode == 0:
                _run(workspace, "checkout", "-B", "work", "refs/remotes/atrex-local/work")
        imported = []
        mappings = []
        for path in exports:
            audit = path.with_suffix(".json")
            try:
                metadata = json.loads(audit.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            source_refs = metadata.get("source_refs")
            originals = metadata.get("original_agent_commits")
            if not isinstance(source_refs, dict) or not isinstance(originals, dict):
                source_refs = {"head": metadata.get("source_ref")}
                originals = {"head": metadata.get("original_agent_commit")}
            if (
                not source_refs
                or "head" not in source_refs
                or set(source_refs) != set(originals)
                or any(
                    not isinstance(ref, str)
                    or not ref.startswith("refs/atrex/shared/")
                    for ref in source_refs.values()
                )
                or any(not isinstance(commit, str) for commit in originals.values())
            ):
                raise ValueError("Shared Git bundle has no valid namespaced source refs")
            result = _run(
                workspace,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                str(path),
                *(f"+{ref}:{ref}" for ref in source_refs.values()),
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
            imported.append(path.name)
            imported_commits = {}
            for label, source_ref in source_refs.items():
                resolved = _run(workspace, "rev-parse", source_ref, check=False)
                imported_commit = (
                    resolved.stdout.decode().strip()
                    if resolved.returncode == 0
                    else None
                )
                if imported_commit != originals[label]:
                    raise ValueError(
                        "Shared Git import changed the peer commit ID; Kernel identity is invalid"
                    )
                imported_commits[label] = imported_commit
            mapping = {
                "export": path.name,
                "source_ref": source_refs["head"],
                "original_agent_commit": originals["head"],
                "imported_commit": imported_commits["head"],
            }
            if len(source_refs) > 1:
                mapping.update(
                    source_refs=source_refs,
                    original_agent_commits=originals,
                    imported_commits=imported_commits,
                )
            mappings.append(mapping)
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
        bundle.parent.mkdir(parents=True, exist_ok=True)
        staged = bundle.with_suffix(".tmp")
        staged.unlink(missing_ok=True)
        _run(workspace, "bundle", "create", str(staged), "--all")
        os.replace(staged, bundle)
        refs = _run(workspace, "for-each-ref", "--format=%(refname)", "refs/atrex/shared").stdout.decode().splitlines()
    receipt = {
        "schema_version": 1,
        "imports": imported,
        "commit_mappings": mappings,
        "shared_refs": sorted(refs),
    }
    durable_write_json(audit_path, receipt)
    return receipt
