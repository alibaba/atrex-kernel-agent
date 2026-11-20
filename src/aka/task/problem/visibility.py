"""Visibility of the input projection actually supplied to the Agent."""
from pathlib import Path


def requested_evaluator_shape_visibility(workspace: Path, mode: str, expose_evaluator_shapes: bool) -> str:
    if type(expose_evaluator_shapes) is not bool:
        raise TypeError("expose_evaluator_shapes must be a boolean")
    if expose_evaluator_shapes:
        return "exposed"
    if mode not in {"performance", "ultra"} or not (workspace / "agent_problem.json").is_file():
        raise ValueError("Hidden Shapes are incompatible with this input projection: "
                         "a Performance/Ultra agent_problem.json projection is required. "
                         "Use --expose-evaluator-shapes or a supported Native input.")
    return "hidden"


def workspace_uses_hidden_shapes(workspace: Path) -> bool:
    return (workspace / "agent_problem.json").is_file()


def visibility_mismatch_message(existing: str, requested: str) -> str:
    return (f"workspace evaluator shape visibility mismatch: recorded={existing}, requested={requested}; "
            "resume with the pinned visibility or use a new Campaign")
