from __future__ import annotations


import subprocess
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path

# Re-export shared facts/fields for existing callers; policy stays local.
from aka.legacy.task.git import (
    _git,
    git_text as git_text,
    git_head as git_head,
    working_changes as working_changes,
    changed_paths as changed_paths,
)

from aka.light.workflow.store import CampaignStore








def git_blob(workspace: Path, revision: str, path: str) -> bytes:
    """Read an exact controller-selected committed file."""
    return _git(workspace, "show", f"{revision}:{path}", binary=True).stdout


def warn_kernel_mismatch(workspace: Path, committed_source: bytes) -> bool:
    """Diagnose when mutable workspace bytes differ from a committed source."""
    try:
        from aka.light.workspace.sources import snapshot as source_snapshot
        mismatch = source_snapshot(workspace) != committed_source
    except (OSError, ValueError):
        mismatch = True
    if mismatch:
        logging.getLogger(__name__).warning(
            "Candidate Kernel integrity warning: source at %s is unreadable or differs "
            "from the committed source used by this explicit verification request",
            workspace,
        )
    return mismatch






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
        branch = f"atrex/long-e{episode:04d}-{uuid.uuid4().hex[:8]}"
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
        CampaignStore.ensure_excluded(self.path)

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



    def reset_scratch(self) -> None:
        from aka.light.workspace.auxiliary import reset_episode_scratch

        reset_episode_scratch(self.path)



    def remove(self, incumbent_workspace: Path) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
