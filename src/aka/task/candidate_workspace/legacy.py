"""Original candidate worktree lifecycle and Git promotion transaction."""
from __future__ import annotations
import subprocess
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from aka.task.git import (_git, git_text, git_head, working_changes, changed_paths, ignored_evidence_files, protected_violation, _manifest_deleted)
from aka.task.io import atomic_write_json
from aka.contracts.content import WorkspaceFile
from .content import selected_file_bytes
from .policy import CANDIDATE_PATHS, TIMELINE_PROBE_MARKERS

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


def promote_candidate(incumbent_workspace, *, base_commit, candidate_commit, episode,
                      record_files=(), evidence=None, memory_version=None, memory_record=None,
                      commit_message=None):
    if any(value is not None for value in (evidence, memory_version, memory_record)):
        if record_files or any(value is None for value in (evidence, memory_version, memory_record)):
            raise ValueError("promotion requires opaque files or complete compatibility records")
        from ._compat_records import promotion_record_files
        record_files = promotion_record_files(episode=episode, evidence=evidence,
                                             memory_version=memory_version, memory_record=memory_record)
    from .promotion import promote_candidate as promote
    return promote(incumbent_workspace, base_commit=base_commit, candidate_commit=candidate_commit,
                   episode=episode, record_files=record_files, commit_message=commit_message)
