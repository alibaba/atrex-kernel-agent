"""Record non-promoted episode outcome in canonical Workflow history."""

from pathlib import Path
from typing import Any
import subprocess
from aka.legacy.task.git import git_head
from .protocol import atomic_write_json


def record_episode_outcome(
    incumbent_workspace: Path,
    *,
    base_commit: str,
    version: int,
    episode: int,
    status: str,
    memory_record: dict[str, Any],
) -> str:
    """Advance main-compatible version history without changing the incumbent kernel."""
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing outcome record")
    memory_path = incumbent_workspace / "memory" / f"v{version}.json"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(memory_path, memory_record)
    subprocess.run(
        ["git", "add", str(memory_path.relative_to(incumbent_workspace))],
        cwd=str(incumbent_workspace),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=atrex-long-horizon",
            "-c",
            "user.email=atrex-long-horizon@local",
            "commit",
            "--only",
            "-m",
            f"v{version}: long-horizon episode {episode} {status}",
            "--",
            str(memory_path.relative_to(incumbent_workspace)),
        ],
        cwd=str(incumbent_workspace),
        check=True,
        capture_output=True,
        text=True,
    )
    return git_head(incumbent_workspace)
