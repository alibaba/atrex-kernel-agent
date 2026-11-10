"""Persisted Task input visibility shared by Workflow hosts."""
import json
from pathlib import Path

MODE_STATE_FILE = ".orchestrator_mode.json"

def requested_evaluator_shape_visibility(
    workspace: Path,
    mode: str,
    expose_evaluator_shapes: bool,
) -> str:
    """Return the visibility of the evaluator flow actually staged in workspace.

    Hidden-shape handling is active only after production staging has produced
    ``agent_problem.json``.  This mirrors the generalized-problem predicate and
    prevents SOL or non-Atrex workspaces from being mislabeled hidden merely
    because a visibility flag was false.
    """
    return (
        "hidden"
        if mode == "production"
        and not expose_evaluator_shapes
        and (workspace / "agent_problem.json").is_file()
        else "exposed"
    )

def historical_evaluator_shape_visibility(workspace: Path, mode: str) -> str:
    """Infer pre-state-file behavior without consulting a new CLI default."""
    return (
        "hidden"
        if mode == "production" and (workspace / "agent_problem.json").is_file()
        else "exposed"
    )

def workspace_uses_hidden_shapes(workspace: Path) -> bool:
    """Read persisted visibility, falling back to the historical file signal."""
    state_path = workspace / MODE_STATE_FILE
    if not state_path.is_file():
        state = {}
    else:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON in {MODE_STATE_FILE}: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError(f"{MODE_STATE_FILE} must contain a JSON object")
    visibility = state.get("evaluator_shape_visibility")
    if visibility in {"hidden", "exposed"}:
        return state.get("mode") == "production" and visibility == "hidden"
    return historical_evaluator_shape_visibility(
        workspace, str(state.get("mode") or "")
    ) == "hidden"

def visibility_mismatch_message(existing: str, requested: str) -> str:
    matching_flag = (
        "--no-expose-evaluator-shapes"
        if existing == "hidden"
        else "--expose-evaluator-shapes"
    )
    return (
        "workspace evaluator shape visibility mismatch: "
        f"recorded={existing}, requested={requested}; resume with "
        f"{matching_flag} to match the pinned workspace, or use a fresh campaign "
        "workspace to change visibility"
    )
