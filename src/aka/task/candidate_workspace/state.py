"""Workspace Git facts, pinned source identity and stall counter.

Git is the single source of truth for a committed win; everything here reads mechanical
facts rather than model-reported ones.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from aka.task.candidate_workspace.filenames import FRAMEWORK_BASELINE_FILE
from aka.task.candidate_workspace.excludes import STALL_STATE_FILE
from aka.task.git import git_path_blob, git_worktree_blob


# ── git is the SINGLE source of truth for a "committed win" ───────────────────
# A real win is a commit that CHANGES kernel.py. A dead-end "record" commit leaves kernel.py
# identical to its parent. Everything (stall counter, target-met, convert incumbent) keys off
# this git fact, NOT off the LLM-filled git_commit_hash / quality_gate in memory (which can drift
# from what actually got committed). One primitive, reused everywhere: commit_changed_kernel().


def git_head(workspace: Path) -> str:
    """Current HEAD sha, or '' if not a repo / no commits yet."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workspace),
                           capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def git_kernel_blob(workspace: Path) -> str:
    """Committed kernel.py blob id, stable across metadata-only commits."""
    return git_path_blob(workspace, "HEAD", "kernel.py")


def v0_baseline_commit(workspace: Path) -> str:
    """The commit that introduced ``kernel.py`` — the campaign's V0 baseline.

    A setup session may commit campaign scaffolding (README, profile driver) before V0, so the
    repository root commit is not necessarily V0 and may carry no kernel at all.
    """
    result = subprocess.run(
        ["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    commits = result.stdout.split()
    return commits[0] if commits else ""


def head_kernel_is_initial_baseline(workspace: Path) -> bool:
    """Whether committed HEAD still uses the repository's original V0 kernel.

    Production V0 is intentionally allowed to be a PyTorch reference wrapper.
    Later failed-policy/dead-end records may advance ``memory/vN.json`` and Git
    metadata without changing ``kernel.py``.  Resume must recognize that state
    by kernel provenance, not by assuming ``latest_version > 0`` means an
    optimized kernel was accepted.

    Once the framework-baseline stage lands v1 this returns False, because the
    accepted framework kernel is a real kernel change.  The framework baseline
    is tracked by ``framework_baseline.json``, not by this predicate.
    """
    baseline_commit = v0_baseline_commit(workspace)
    if not baseline_commit:
        return False
    baseline_blob = subprocess.run(
        ["git", "rev-parse", f"{baseline_commit}:kernel.py"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if baseline_blob.returncode != 0:
        return False
    return bool(
        baseline_blob.stdout.strip()
        and baseline_blob.stdout.strip() == git_kernel_blob(workspace)
    )


def read_framework_baseline(workspace: Path) -> Optional[dict]:
    """Read the framework-baseline marker from committed HEAD, never from the worktree.

    An interrupted session can leave an unstaged marker behind; trusting it would pin a kernel
    that was never validated.
    """
    show = subprocess.run(
        ["git", "show", f"HEAD:{FRAMEWORK_BASELINE_FILE}"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        return None
    try:
        marker = json.loads(show.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{workspace / FRAMEWORK_BASELINE_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"{workspace / FRAMEWORK_BASELINE_FILE} must contain a JSON object")
    return marker


def resolve_framework_baseline_commit(workspace: Path) -> tuple[str, int]:
    """Return the pinned (commit, version) of the framework baseline, or ("", 0) when unpinned.

    Verification is fail-closed and never consults HEAD's tree, so the pin survives HEAD
    advancing through later optimization versions. A broken marker is an error rather than a
    silent fallback to the root commit.
    """
    marker = read_framework_baseline(workspace)
    if marker is None:
        return "", 0
    commit = _normalize_commit_hash(marker.get("commit"))
    if not commit:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has no usable commit hash")
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} points at missing commit {commit[:12]}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"{FRAMEWORK_BASELINE_FILE} commit {commit[:12]} is not an ancestor of HEAD"
        )
    blob = subprocess.run(
        ["git", "rev-parse", f"{commit}:kernel.py"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    recorded_blob = str(marker.get("kernel_blob") or "").strip()
    if blob.returncode != 0 or not recorded_blob or blob.stdout.strip() != recorded_blob:
        raise RuntimeError(
            f"{FRAMEWORK_BASELINE_FILE} kernel blob does not match commit {commit[:12]}"
        )
    version_text = str(marker.get("version") or "")
    match = re.fullmatch(r"v(\d+)", version_text)
    if match is None:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has an unusable version {version_text!r}")
    return commit, int(match.group(1))


def _normalize_commit_hash(ref: object) -> str:
    """Return a safe git commit hash from an LLM-authored memory value.

    A short SHA containing only decimal digits can be serialized as a JSON
    number (for example ``7847485``).  Preserve that valid value by converting
    integers back to strings, while rejecting booleans, floats, and arbitrary
    git revisions/options.  Memory records are expected to contain commit
    hashes, not general revision expressions.
    """
    if isinstance(ref, bool):
        return ""
    if isinstance(ref, int):
        ref = str(ref)
    if not isinstance(ref, str):
        return ""
    ref = ref.strip()
    return ref if re.fullmatch(r"[0-9a-fA-F]{4,64}", ref) else ""


def commit_changed_kernel(workspace: Path, ref: object) -> bool:
    """True iff commit `ref` changed kernel.py vs its parent (i.e. a real win, not a dead-end
    record commit). The one git primitive the win/stall/incumbent logic all share.

    ``git_commit_hash`` is written by agents, so normalize it before passing it
    to subprocess.  In particular, all-decimal short SHAs may round-trip
    through JSON as integers.
    """
    commit_hash = _normalize_commit_hash(ref)
    if not commit_hash:
        return False
    try:
        r = subprocess.run(["git", "show", "--numstat", "--format=", commit_hash, "--", "kernel.py"],
                           cwd=str(workspace), capture_output=True, text=True)
    except (OSError, TypeError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def read_stall(workspace: Path) -> Optional[int]:
    """Persisted live stall counter, or None when absent (caller reconstructs)."""
    p = workspace / STALL_STATE_FILE
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text(encoding="utf-8")).get("stall")
    except (OSError, ValueError):
        return None
    return int(v) if isinstance(v, int) else None


def write_stall(workspace: Path, stall: int) -> None:
    """Persist the live stall counter so a restart resumes the exact value (survives git reset —
    the file is gitignored). This is the single source of truth for the stall->convert cooldown."""
    try:
        (workspace / STALL_STATE_FILE).write_text(
            json.dumps({"stall": int(stall)}, indent=2), encoding="utf-8")
    except OSError:
        pass


def reconstruct_stall(workspace: Path) -> int:
    """Best-effort rebuild of the live stall counter from git when no persisted state exists yet
    (e.g. a workspace from before state was tracked). Counts trailing commits from HEAD that did
    NOT change kernel.py; a win (kernel.py change) stops the count. read_stall() is authoritative —
    this only bootstraps it, so it does not attempt to replay convert-issued resets."""
    try:
        r = subprocess.run(["git", "rev-list", "HEAD"], cwd=str(workspace),
                           capture_output=True, text=True)
    except OSError:
        return 0
    if r.returncode != 0:
        return 0
    trailing = 0
    for h in r.stdout.split():
        if commit_changed_kernel(workspace, h):   # a win -> stop counting
            break
        trailing += 1
    return trailing
