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
    # Preserve the established default snapshot key. Only the opt-in visibility
    # mode needs a distinct namespace, so existing hidden-shape runs still resume.
    identity = str(origin) + "\0" + optimization_mode + "\0" + suffix
    if expose_evaluator_shapes:
        identity += "\0expose-evaluator-shapes"
    snapshot_key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    base = Path(workspace) if workspace else Path.cwd()
    return ProblemSource(
        origin, base / ".aka-problems" / snapshot_key
    )
