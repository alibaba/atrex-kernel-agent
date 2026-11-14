"""Persisted Task input visibility shared by Workflow hosts."""
import json
from pathlib import Path

MODE_STATE_FILE = ".orchestrator_mode.json"

def persist_workspace_policy_identity(
    workspace: Path, mode: str, framework: str, *, agent_runtime=None,
    source_checks_enabled=True, expose_evaluator_shapes=True,
) -> None:
    """Pin mode/framework/runtime and historical visibility before Agent launch."""
    if mode not in ("leaderboard", "production"):
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    requested_runtime = (
        str(agent_runtime).strip() if agent_runtime is not None else None
    )
    if agent_runtime is not None and not requested_runtime:
        raise ValueError("agent_runtime must be a non-empty runtime id")
    if type(expose_evaluator_shapes) is not bool:
        raise TypeError("expose_evaluator_shapes must be a boolean")
    workspace.mkdir(parents=True, exist_ok=True)
    requested_visibility = requested_evaluator_shape_visibility(
        workspace, mode, expose_evaluator_shapes
    )
    state_path = workspace / MODE_STATE_FILE
    state_changed = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid optimization-mode state: {state_path}") from exc
        if not isinstance(state, dict):
            raise RuntimeError(f"invalid optimization-mode state: {state_path}; expected a JSON object")
        existing_mode = state.get("mode")
        existing_framework = state.get("framework")
        if existing_mode != mode or existing_framework != framework:
            raise RuntimeError(
                "workspace policy mismatch: "
                f"recorded mode/framework={existing_mode}/{existing_framework}, "
                f"requested={mode}/{framework}"
            )
        existing_visibility = state.get("evaluator_shape_visibility")
        if existing_visibility is None:
            existing_visibility = historical_evaluator_shape_visibility(workspace, mode)
            state["evaluator_shape_visibility"] = existing_visibility
            state_changed = True
        if existing_visibility not in {"hidden", "exposed"}:
            raise RuntimeError(
                f"invalid evaluator shape visibility: {existing_visibility!r}"
            )
        if existing_visibility != requested_visibility:
            raise RuntimeError(visibility_mismatch_message(
                existing_visibility, requested_visibility
            ))
        existing_runtime = str(state.get("agent_runtime") or "").strip()
        if requested_runtime and existing_runtime and existing_runtime != requested_runtime:
            raise RuntimeError(
                "workspace agent runtime mismatch: "
                f"recorded={existing_runtime}, requested={requested_runtime}; "
                "use a fresh campaign workspace to change backend"
            )
        if requested_runtime and not existing_runtime:
            state["agent_runtime"] = requested_runtime
            state_changed = True
    else:
        state = {
            "mode": mode,
            "framework": framework,
            "evaluator_shape_visibility": requested_visibility,
        }
        if requested_runtime:
            state["agent_runtime"] = requested_runtime
        state_changed = True

    if state.get("source_checks_enabled") != source_checks_enabled:
        state["source_checks_enabled"] = source_checks_enabled
        state_changed = True
    if state_changed:
        state_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )


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
