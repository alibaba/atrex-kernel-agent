"""Validate an explicit private measurement input association."""
from pathlib import Path
from aka.execution.gateway.sandbox import (
    _is_generalized_workspace, _json_object, PRIVATE_REFERENCE_ENV, AGENT_PROBLEM_FILENAME,
)
def private_reference_dir(workspace: Path, environment) -> Path | None:
    """Resolve private evaluator inputs only for a generalized production workspace."""
    if not _is_generalized_workspace(workspace):
        return None
    raw = environment.get(PRIVATE_REFERENCE_ENV, "")
    if not raw:
        raise ValueError(
            f"{PRIVATE_REFERENCE_ENV} is required for generalized Atrex-Bench evaluation"
        )
    private_dir = Path(raw).expanduser().resolve()
    if not private_dir.is_dir():
        raise ValueError("configured private Atrex-Bench reference directory is missing")
    private_problem = private_dir / AGENT_PROBLEM_FILENAME
    public_problem = workspace / AGENT_PROBLEM_FILENAME
    # A user-provided contract remains evaluator-owned and must match byte-for-byte.
    # An automatically authored production contract intentionally exists only in the
    # campaign workspace, so absence from the detailed-shape source is valid.
    if private_problem.is_file() and (
        private_problem.read_bytes() != public_problem.read_bytes()
    ):
        raise ValueError(
            "workspace agent_problem.json does not match the evaluator-owned public contract"
        )
    return private_dir
