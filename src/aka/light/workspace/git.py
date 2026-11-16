"""Small Git read helpers used by the persistent Light workspace.

Episode worktree creation belongs to the Legacy workflow. Light keeps one
trajectory repository and uses these helpers only for committed reads and
diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aka.legacy.task.git import (
    _git,
    changed_paths as changed_paths,
    git_head as git_head,
    git_text as git_text,
    working_changes as working_changes,
)


def git_blob(workspace: Path, revision: str, path: str) -> bytes:
    """Read an exact Agent-selected committed file."""
    return _git(workspace, "show", f"{revision}:{path}", binary=True).stdout


def warn_kernel_mismatch(workspace: Path, committed_source: bytes) -> bool:
    """Diagnose when mutable workspace bytes differ from committed source."""
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
