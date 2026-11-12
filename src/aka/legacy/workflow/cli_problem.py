"""Convert historical CLI and constructor paths into explicit Workflow inputs."""

import hashlib
from pathlib import Path

from aka.contracts.workflow import ProblemSource


def problem_request(
    directory,
    optimization_mode,
    *,
    workspace="",
    suffix="",
    expose_evaluator_shapes=True,
):
    origin = Path(directory).resolve()
    identity = str(origin) + "\0" + optimization_mode + "\0" + suffix
    if optimization_mode == "production" and expose_evaluator_shapes:
        identity += "\0expose-evaluator-shapes"
    snapshot_key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    base = Path(workspace) if workspace else Path.cwd()
    return ProblemSource(
        origin, base / ".aka-problems" / snapshot_key
    )
