"""Supervisor-owned Git excludes, shared by a Campaign and its worktrees."""

from __future__ import annotations

import fcntl
import subprocess
from pathlib import Path

from .constants import STALL_STATE_FILE
from .durable_state import durable_write_text

# Never exclude canonical memory/vN.json or candidate source. These rules replace
# the generated workspace .gitignore; they are not Agent-owned configuration.
WORKSPACE_EXCLUDES = (
    "__pycache__/",
    "*.pyc",
    "*.ncu-rep",
    "/scratch/",
    "/traces.jsonl",
    "/.finalize_traces.jsonl",
    "/submission.json",
    "/tools",
    "/reference",
    "/skills",
    "/reference-projects",
    "/gpu-wiki",
    "/.claude",
    "/.qoder",
    "/.agents",
    "/.orchestrator_mode.json",
    f"/{STALL_STATE_FILE}",
    "/.atrex_long_horizon/",
    "/verification_artifacts/.atrex_long_horizon_verify/",
    "/memory/live.json",
    "/trace-retention-manifest.json",
)


def install_git_excludes(workspace: Path, *, required: bool = False) -> None:
    """Idempotently append managed rules without overwriting custom exclusions."""
    if not (workspace / ".git").exists():
        if required:
            raise RuntimeError(f"Workspace Git metadata is missing: {workspace}")
        return  # Asset-only staging can precede Git initialization.
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "info/exclude"], cwd=workspace,
        capture_output=True, text=True, check=True,
    )
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = workspace / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("Git exclude path must not be a symlink")
    with (path.parent / "atrex-exclude.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        missing = [rule for rule in WORKSPACE_EXCLUDES if rule not in text.splitlines()]
        if missing:
            suffix = "" if not text or text.endswith("\n") else "\n"
            durable_write_text(path, text + suffix + "\n".join(missing) + "\n")


if __name__ == "__main__":
    import sys

    install_git_excludes(Path(sys.argv[1]), required=True)
