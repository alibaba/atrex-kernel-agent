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
