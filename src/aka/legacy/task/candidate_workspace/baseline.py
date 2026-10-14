"""Check and restore original baseline files without deciding task acceptance."""
from pathlib import Path
import subprocess
from typing import Iterator

from aka.legacy.task.git import git_path_blob, git_worktree_blob
from .policy import IMMUTABLE_BASELINE_PATHS


def _changed_baseline_paths(workspace: Path, baseline_commit: str) -> Iterator[str]:
    for path in IMMUTABLE_BASELINE_PATHS:
        original = git_path_blob(workspace, baseline_commit, path)
        if original and original != git_worktree_blob(workspace, path):
            yield path


def baseline_changes(workspace: Path, baseline_commit: str) -> list[str]:
    """List baseline-owned files whose on-disk blobs differ, in policy order.

    Paths absent from the baseline are ignored, as in the original application.
    This checks content, not mode or index state, and does not modify the workspace.
    """
    return list(_changed_baseline_paths(workspace, baseline_commit))


def restore_baseline(workspace: Path, baseline_commit: str) -> list[str]:
    """Explicitly restore changed baseline files and return successful paths.

    Preserve the original Git checkout behavior: update worktree and index,
    continue after an individual checkout failure, and leave other files alone.
    """
    restored = []
    for path in _changed_baseline_paths(workspace, baseline_commit):
        checkout = subprocess.run(
            ["git", "checkout", baseline_commit, "--", path],
            cwd=str(workspace),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if checkout.returncode == 0:
            restored.append(path)
    return restored
