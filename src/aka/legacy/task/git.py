"""Git operations used by task-owned candidate workspaces."""
from __future__ import annotations
import subprocess
from pathlib import Path, PurePosixPath
from .candidate_workspace.policy import PROTECTED_PATHS, PROTECTED_PREFIXES, EPISODE_EVIDENCE_PREFIXES

def _git(workspace: Path, *args: str, check: bool = True, binary: bool = False):
    result = subprocess.run(
        ["git", *args], cwd=str(workspace), capture_output=True, text=not binary
    )
    if check and result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {str(stderr)[-1200:]}")
    return result


def git_text(workspace: Path, *args: str, check: bool = True) -> str:
    return _git(workspace, *args, check=check).stdout.strip()


def git_head(workspace: Path) -> str:
    return git_text(workspace, "rev-parse", "HEAD")


def git_path_blob(workspace: Path, ref: str, path: str) -> str:
    """Committed blob id of one path at one ref, or '' when it is absent there."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{ref}:{path}"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_worktree_blob(workspace: Path, path: str) -> str:
    """Blob id of the on-disk file, or '' when it is missing."""
    if not (workspace / path).is_file():
        return ""
    try:
        result = subprocess.run(
            ["git", "hash-object", "--", path],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def working_changes(workspace: Path) -> list[str]:
    # Porcelain status uses its first two columns for XY state.  Preserve the
    # leading space on an unstaged first entry; git_text().strip() would remove
    # it and shift the pathname by one character.
    output = _git(workspace, "status", "--porcelain", "--untracked-files=all").stdout
    return [line[3:].strip('"') for line in output.splitlines() if len(line) >= 4]


def changed_paths(
    workspace: Path, base_commit: str, candidate_commit: str = "HEAD"
) -> list[str]:
    output = git_text(
        workspace,
        "diff",
        "--name-only",
        "--no-renames",
        base_commit,
        candidate_commit,
        "--",
    )
    return sorted(path for path in output.splitlines() if path)


def ignored_evidence_files(workspace: Path) -> list[str]:
    output = git_text(
        workspace,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *[prefix.rstrip("/") for prefix in EPISODE_EVIDENCE_PREFIXES],
    )
    return sorted(path for path in output.splitlines() if path)


def protected_violation(paths: list[str]) -> str:
    for value in paths:
        normalized = PurePosixPath(value).as_posix()
        if normalized in PROTECTED_PATHS or normalized.startswith(PROTECTED_PREFIXES):
            return f"candidate modified protected path: {normalized}"
    return ""


def _manifest_deleted(workspace: Path, base_commit: str, candidate_commit: str) -> bool:
    return "solution.json" in git_text(
        workspace, "diff", "--no-renames", "--diff-filter=D", "--name-only",
        base_commit, candidate_commit, "--", "solution.json",
    ).splitlines()


