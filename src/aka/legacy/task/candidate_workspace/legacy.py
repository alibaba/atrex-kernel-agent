"""Original candidate worktree lifecycle and Git promotion transaction."""
from __future__ import annotations
import subprocess
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from aka.legacy.task.git import (_git, git_text, git_head, working_changes, changed_paths, ignored_evidence_files, protected_violation, _manifest_deleted)
from aka.legacy.task.io import atomic_write_json
from aka.contracts.content import WorkspaceFile
from .content import selected_file_bytes
from .policy import CANDIDATE_PATHS, TIMELINE_PROBE_MARKERS
from .excludes import ensure_excluded

@dataclass(frozen=True)
class EpisodeWorktree:
    episode: int
    base_commit: str
    branch: str
    path: Path

    @classmethod
    def plan(
        cls,
        incumbent_workspace: Path,
        episode: int,
        base_commit: str,
        root: Path | None = None,
    ) -> "EpisodeWorktree":
        branch = f"atrex/long-e{episode:04d}-{uuid.uuid4().hex[:8]}/candidate"
        worktree_root = root or (
            incumbent_workspace.parent
            / ".atrex_long_horizon_worktrees"
            / incumbent_workspace.name
        )
        worktree_root.mkdir(parents=True, exist_ok=True)
        path = worktree_root / f"e{episode:04d}-{uuid.uuid4().hex[:8]}"
        return cls(episode=episode, base_commit=base_commit, branch=branch, path=path)

    def materialize(self, incumbent_workspace: Path) -> None:
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "-b",
                self.branch,
                str(self.path),
                self.base_commit,
            ],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
        ensure_excluded(self.path)
        from .context import restore_context_projection, uses_context
        if uses_context(self.path):
            restore_context_projection(self.path, self.base_commit)

    @classmethod
    def create(
        cls,
        incumbent_workspace: Path,
        episode: int,
        base_commit: str,
        root: Path | None = None,
    ) -> "EpisodeWorktree":
        planned = cls.plan(incumbent_workspace, episode, base_commit, root)
        planned.materialize(incumbent_workspace)
        return planned

    def validate_candidate(self, candidate_commit: str) -> tuple[str, list[str]]:
        resolved = git_text(
            self.path,
            "rev-parse",
            "--verify",
            f"{candidate_commit}^{{commit}}",
            check=False,
        )
        if not resolved:
            return "candidate_commit does not resolve", []
        if resolved != git_head(self.path):
            return "candidate_commit must equal episode HEAD", []
        branch = git_text(
            self.path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
        )
        if branch != self.branch:
            return "episode worktree left its isolated branch", []
        ancestor = _git(
            self.path,
            "merge-base",
            "--is-ancestor",
            self.base_commit,
            resolved,
            check=False,
        )
        if ancestor.returncode:
            return "candidate_commit is not descended from incumbent", []
        from .context import context_changes, uses_context
        if uses_context(self.path):
            changed_context = context_changes(self.path, self.base_commit)
            if changed_context:
                return "candidate modified frozen workspace context: " + ", ".join(changed_context), []
        dirty = working_changes(self.path)
        violation = protected_violation(dirty)
        if violation:
            return violation, []
        if CANDIDATE_PATHS.intersection(dirty):
            return "worktree kernel.py and solution.json must match candidate_commit", []
        paths = changed_paths(self.path, self.base_commit, resolved)
        if not paths:
            return "candidate has no changes relative to incumbent", []
        if "kernel.py" not in paths or not set(paths).issubset(CANDIDATE_PATHS):
            return "candidate commit must change kernel.py and may also update solution.json", paths
        if _manifest_deleted(self.path, self.base_commit, resolved):
            return "candidate must preserve the incumbent solution.json manifest", paths
        kernel_text = (self.path / "kernel.py").read_text(encoding="utf-8", errors="replace")
        if any(marker in kernel_text for marker in TIMELINE_PROBE_MARKERS):
            return "candidate kernel.py still contains timeline profiling probes", paths
        return "", paths

    def archive(self, destination: Path, candidate_commit: str = "HEAD") -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        committed_patch = _git(
            self.path,
            "diff",
            "--binary",
            self.base_commit,
            candidate_commit,
            "--",
            binary=True,
        ).stdout
        (destination / "candidate.patch").write_bytes(committed_patch)
        worktree_patch = _git(
            self.path, "diff", "--binary", self.base_commit, "--", binary=True
        ).stdout
        (destination / "worktree.patch").write_bytes(worktree_patch)
        archived_files = destination / "worktree_files"
        paths = set(changed_paths(self.path, self.base_commit, candidate_commit))
        paths.update(working_changes(self.path))
        paths.update(ignored_evidence_files(self.path))
        for relative in sorted(paths):
            source = self.path / relative
            if not source.is_file() or source.is_symlink():
                continue
            target = archived_files / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        atomic_write_json(
            destination / "git.json",
            {
                "episode": self.episode,
                "base_commit": self.base_commit,
                "branch": self.branch,
                "head": git_head(self.path),
                "dirty_paths": working_changes(self.path),
            },
        )
        return destination

    def remove(self, incumbent_workspace: Path) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )


def promote_candidate(
    incumbent_workspace: Path,
    *,
    base_commit: str,
    candidate_commit: str,
    episode: int,
    record_files: tuple[WorkspaceFile, ...] = (),
    evidence: dict[str, Any] | None = None,
    memory_version: int | None = None,
    memory_record: dict[str, Any] | None = None,
) -> str:
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing promotion")
    from .context import context_changes, uses_context
    if uses_context(incumbent_workspace):
        changed_context = context_changes(incumbent_workspace, base_commit)
        if changed_context:
            raise RuntimeError("incumbent frozen workspace context changed: " + ", ".join(changed_context))
    candidate_paths = changed_paths(incumbent_workspace, base_commit, candidate_commit)
    if "kernel.py" not in candidate_paths or not set(candidate_paths).issubset(CANDIDATE_PATHS):
        raise RuntimeError("promotion requires kernel.py with only an optional solution.json update")
    if _manifest_deleted(incumbent_workspace, base_commit, candidate_commit):
        raise RuntimeError("candidate must preserve the incumbent solution.json manifest")
    if any(value is not None for value in (evidence, memory_version, memory_record)):
        if record_files or any(value is None for value in (evidence, memory_version, memory_record)):
            raise ValueError("promotion requires either opaque files or complete compatibility records")
        from aka.legacy.compat.workspace_records import promotion_record_files
        record_files = promotion_record_files(episode=episode, evidence=evidence,
            memory_version=memory_version, memory_record=memory_record)
    files = tuple(record_files)
    reserved = CANDIDATE_PATHS
    if uses_context(incumbent_workspace):
        from .context import MANIFEST, load_context
        value = load_context(incumbent_workspace)
        reserved = (*CANDIDATE_PATHS, MANIFEST, *value["context"], *value.get("namespaces", {}))
    selected = selected_file_bytes(incumbent_workspace, files, reserved=reserved)
    dirty = working_changes(incumbent_workspace)
    if set(candidate_paths).intersection(dirty):
        raise RuntimeError("incumbent candidate source has uncommitted changes")
    index_path = Path(git_text(incumbent_workspace, "rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = incumbent_workspace / index_path
    old_index = index_path.read_bytes() if index_path.exists() else None
    original = {}
    new_directories = set()
    for name in (*candidate_paths, *selected):
        target = incumbent_workspace / name
        original[name] = (target.read_bytes(), target.stat().st_mode) if target.is_file() else None
        parent = target.parent
        while parent != incumbent_workspace and not parent.exists():
            new_directories.add(parent)
            parent = parent.parent
    try:
        # Materialize only the validated candidate paths; unrelated index/worktree
        # entries remain untouched and commit --only excludes them.
        subprocess.run(["git", "checkout", candidate_commit, "--", *candidate_paths],
                       cwd=incumbent_workspace, check=True, capture_output=True, text=True)
        for item in files:
            target = incumbent_workspace / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(selected[item.path])
            if item.source is not None:
                target.chmod(item.source.stat().st_mode)
        if selected:
            subprocess.run(["git", "add", "--", *selected], cwd=incumbent_workspace, check=True)
        subprocess.run(["git", "-c", "user.name=atrex-long-horizon", "-c",
            "user.email=atrex-long-horizon@local", "commit", "--only", "-m",
            f"episode {episode}: promote verified long-horizon candidate", "--",
            *candidate_paths, *selected], cwd=incumbent_workspace, check=True,
            capture_output=True, text=True)
        return git_head(incumbent_workspace)
    except Exception:
        if git_head(incumbent_workspace) != base_commit:
            subprocess.run(["git", "reset", "--soft", base_commit], cwd=incumbent_workspace,
                           check=False, capture_output=True)
        for name, prior in original.items():
            target = incumbent_workspace / name
            if prior is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(prior[0])
                target.chmod(prior[1])
        if old_index is None:
            index_path.unlink(missing_ok=True)
        else:
            index_path.write_bytes(old_index)
        for directory in sorted(new_directories, key=lambda path: len(path.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
