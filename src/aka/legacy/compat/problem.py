"""Convert historical CLI and constructor paths into explicit Workflow inputs."""

import hashlib
from pathlib import Path

from aka.contracts.workflow import ProblemSource


def problem_request(directory, optimization_mode, *, workspace="", suffix=""):
    origin = Path(directory).resolve()
    snapshot_key = hashlib.sha256(
        (str(origin) + "\0" + optimization_mode + "\0" + suffix).encode()
    ).hexdigest()[:24]
    base = Path(workspace) if workspace else Path.cwd()
    return ProblemSource(
        origin, base / ".aka-problems" / snapshot_key
    )


def constructor_problem_request(configuration):
    name = configuration["name"]
    suffix = configuration.get("workspace_suffix", "")
    campaign_name = name + ("_" + suffix if suffix else "")
    work_dir = configuration.get("work_dir", "")
    base = Path(work_dir) if work_dir else Path.cwd()
    return ProblemSource(
        Path(configuration.get("kernel_demo", "")).resolve().parent,
        base / ".aka-problems" / campaign_name,
    )
