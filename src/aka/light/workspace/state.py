"""Workspace state readers: version facts, optional legacy memory and Git state.

Git is the single source of truth for a committed win; everything here reads mechanical
facts rather than model-reported ones.
"""
from __future__ import annotations


import json
import math
import re
import subprocess
from pathlib import Path
from typing import Optional

# Re-export shared facts/fields for existing callers; policy stays local.
from aka.legacy.task.candidate_workspace.state import (
    git_head,
    v0_baseline_commit,
    read_framework_baseline,
    resolve_framework_baseline_commit,
    _normalize_commit_hash,
    commit_changed_kernel,
)
from aka.legacy.task.git import git_path_blob, git_worktree_blob

from aka.light.workflow.constants import FRAMEWORK_BASELINE_FILE, STALL_STATE_FILE
from aka.light.workspace.versions import STATE_DIRECTORY, version_record_path


def latest_version(workspace: Path) -> int:
    """Largest STRICT integer version in controller facts or legacy memory.

    Note: int('71_512') == 71512 in Python (PEP 515 underscores-in-numeric-literals).
    Experimental files named like v71_512.json (an experiment branch for the v71 round) must
    NOT count as v71512 — they're scratch variants of v71, not real iterations. We therefore
    accept only stems matching ^v\\d+\\.json$ exactly.
    """
    _STRICT = re.compile(r"^v(\d+)\.json$")
    vs = []
    for directory in (STATE_DIRECTORY, "memory"):
        for p in (workspace / directory).glob("v*.json"):
            m = _STRICT.match(p.name)
            if m is not None:
                vs.append(int(m.group(1)))
    return max(vs) if vs else -1


def read_memory(workspace: Path, n: int) -> Optional[dict]:
    """Read controller facts, falling back to pre-switch memory-only histories."""
    path = version_record_path(workspace, n)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def speedup_vs_reference(
    workspace: Path,
    candidate_latency_us: object,
    reported_speedup: object = None,
) -> Optional[float]:
    """Return a real reported speedup or derive it from canonical V0 latency."""

    def positive(value: object) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if number > 0.0 and math.isfinite(number) else None

    reported = positive(reported_speedup)
    if reported is not None:
        return reported
    candidate = positive(candidate_latency_us)
    baseline = read_memory(workspace, 0)
    performance = (
        baseline.get("performance")
        if isinstance(baseline, dict)
        and isinstance(baseline.get("performance"), dict)
        else {}
    )
    reference = positive(
        performance.get("latency_us_geomean", performance.get("latency_us"))
    )
    if candidate is None or reference is None:
        return None
    return reference / candidate


# ── git is the SINGLE source of truth for a "committed win" ───────────────────
# A real win is a commit that CHANGES kernel.py. A dead-end "record" commit leaves kernel.py
# identical to its parent. Everything (stall counter, target-met, convert incumbent) keys off
# this git fact, NOT off the LLM-filled git_commit_hash / quality_gate in memory (which can drift
# from what actually got committed). One primitive, reused everywhere: commit_changed_kernel().


def git_kernel_blob(workspace: Path) -> str:
    """Committed kernel.py blob id, stable across metadata-only commits."""
    return git_path_blob(workspace, "HEAD", "kernel.py")


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


def peak_util(mem: Optional[dict]) -> float:
    """Max of tflops / bandwidth peak utilization (%), 0 if unknown."""
    if not mem:
        return 0.0
    perf = mem.get("performance") or {}
    vals = [perf.get("tflops_peak_utilization_pct"), perf.get("bandwidth_peak_utilization_pct")]
    return max([float(v) for v in vals if isinstance(v, (int, float))] or [0.0])


def best_validated_latency_us(workspace: Path, *, from_version: Optional[int] = None) -> Optional[float]:
    """Best correctness-passing measured latency in a workspace.

    Versions before a pinned framework baseline are excluded by default: production cannot ship
    the PyTorch V0 wrapper, so treating its (typically much faster, library-backed) latency as the
    incumbent would reject every candidate a from-scratch DSL kernel can produce.
    """
    if from_version is None:
        _pinned_commit, pinned_version = resolve_framework_baseline_commit(workspace)
        from_version = pinned_version
    best: Optional[float] = None
    for version in range(from_version, latest_version(workspace) + 1):
        memory = read_memory(workspace, version)
        if not memory:
            continue
        gate = (memory.get("quality_gate") or {}).get("result")
        correctness = (memory.get("correctness") or {}).get("status")
        if gate != "PASS" and correctness != "PASS":
            continue
        latency = (memory.get("performance") or {}).get("latency_us")
        if isinstance(latency, (int, float)):
            best = float(latency) if best is None else min(best, float(latency))
    return best
