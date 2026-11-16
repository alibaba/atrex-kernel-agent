"""Workspace state readers: private version facts and Git state.

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
    git_head as git_head,
    v0_baseline_commit,
    _normalize_commit_hash,
)
from aka.legacy.task.git import git_path_blob as git_path_blob

from aka.light.workflow.constants import STALL_STATE_FILE
from aka.light.workspace.baseline import read_baseline_record
from aka.light.workspace.versions import STATE_DIRECTORY, version_record_path


def latest_version(workspace: Path) -> int:
    """Largest STRICT integer version in controller facts.

    Note: int('71_512') == 71512 in Python (PEP 515 underscores-in-numeric-literals).
    Experimental files named like v71_512.json (an experiment branch for the v71 round) must
    NOT count as v71512 — they're scratch variants of v71, not real iterations. We therefore
    accept only stems matching ^v\\d+\\.json$ exactly.
    """
    _STRICT = re.compile(r"^v(\d+)\.json$")
    vs = []
    for p in (workspace / STATE_DIRECTORY).glob("v*.json"):
        m = _STRICT.match(p.name)
        if m is not None:
            vs.append(int(m.group(1)))
    if read_baseline_record(workspace) is not None:
        vs.append(0)
    return max(vs) if vs else -1


def read_memory(workspace: Path, n: int) -> Optional[dict]:
    """Read Supervisor-owned controller facts."""
    if n == 0:
        return read_baseline_record(workspace)
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
# A real win changes the complete candidate source. A metadata-only record leaves source
# identical to its parent. Everything (stall counter, target-met, convert incumbent) keys off
# this git fact, NOT off the LLM-filled git_commit_hash / quality_gate in memory (which can drift
# from what actually got committed). One primitive, reused everywhere: commit_changed_kernel().


def git_kernel_blob(workspace: Path) -> str:
    """Committed candidate identity, stable across metadata-only commits."""
    from aka.light.workspace.sources import revision_identity
    return revision_identity(workspace)


def commit_changed_kernel(workspace: Path, commit: str) -> bool:
    """Count changes to the complete candidate through its file system."""
    from aka.light.workspace.sources import revision_changed
    commit = _normalize_commit_hash(commit)
    return bool(commit) and revision_changed(workspace, commit)


def head_kernel_is_initial_baseline(workspace: Path) -> bool:
    """Whether committed HEAD still uses the repository's original V0 kernel.

    Production V0 is intentionally allowed to be a PyTorch reference wrapper.
    Controller facts can advance without changing ``kernel.py``. Resume must
    recognize that state by Kernel provenance, not by assuming
    ``latest_version > 0`` means an optimized Kernel was accepted.

    It becomes false as soon as the Agent commits a different candidate source.
    """
    baseline_commit = v0_baseline_commit(workspace)
    if not baseline_commit:
        return False
    from aka.light.workspace.sources import revision_identity
    baseline_identity = revision_identity(workspace, baseline_commit)
    return bool(baseline_identity and baseline_identity == git_kernel_blob(workspace))



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
