"""Read-only source and measured-reference facts; no controller Git workflow."""
from pathlib import Path
import math
import subprocess


def git_head(workspace: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=workspace,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def git_path_blob(workspace: Path, commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=workspace, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def speedup_vs_reference(workspace, candidate_latency_us, reported_speedup=None):
    # A task's scoring normalization is not measured V0 evidence.
    from aka.workflow.light.initial_measurement import read_baseline_record
    baseline = read_baseline_record(workspace)
    latency = (baseline or {}).get("performance", {}).get("latency_us_geomean")
    def valid(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    return latency / candidate_latency_us if valid(latency) and valid(candidate_latency_us) else None
