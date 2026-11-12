"""Workflow policy implementation and callable interface."""
from __future__ import annotations

from aka.legacy.task.source.inspection import framework_key
import json
from pathlib import Path


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")


MODE_STATE_FILE = ".orchestrator_mode.json"


POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"


POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"


def requested_evaluator_shape_visibility(
    workspace: Path,
    mode: str,
    expose_evaluator_shapes: bool,
) -> str:
    """Return the visibility of the evaluator flow actually staged in workspace.

    Hidden-shape handling is active only after production staging has produced
    ``agent_problem.json``.  This prevents SOL or an incompletely staged native
    workspace from claiming a privacy boundary that does not actually exist.
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


def optimization_mode_directive(
    mode: str, framework: str, *, source_checks_enabled: bool = True,
    numerical_feedback_instruction: str | None = None,
    candidate_submission_instruction: str = "Before committing",
    correctness_instruction: str | None = None,
    numerical_validation_enabled: bool = True,
) -> str:
    """Shared policy; callers may adapt feedback/submission mechanics, not the gates."""
    if mode == "leaderboard":
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` exactly. Its existing framework guidance remains "
            "unchanged: the requested framework is a recommended direction, compatible mixed/alternate "
            "implementations are allowed when evidence supports them, and third-party helper/kernel "
            "libraries may be used.\n"
        )
    if mode != "production":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    if not source_checks_enabled:
        validation = (
            "Correctness, supplemental numerical probes, performance validation and promotion "
            "requirements remain unchanged. "
            if numerical_validation_enabled
            else "Correctness, performance validation and promotion requirements remain unchanged; "
            "supplemental numerical probes are disabled for this campaign. "
        )
        return (
            "## Optimization mode: production (multi-GPU Agate)\n\n"
            "This generated policy overrides conflicting source/framework restrictions elsewhere "
            "in the workspace. The evaluator requests at least two GPUs from Agate. Production "
            "source inspection and independent source-policy review are disabled for this campaign; "
            "communication libraries, PyTorch distributed operations and alternate implementations "
            "are allowed. No self-authored framework baseline or Triton-to-Gluon conversion is required.\n"
            + validation +
            "Use the immutable evaluator, full workload and required random seeds; preserve "
            "input/output semantics and include all per-call communication and synchronization in "
            "the measured work. Never modify trusted inputs, tolerances or evaluation records.\n"
        )
    if framework_key(framework) == "triton":
        framework_rule = (
            "- The submitted candidate's implementation framework is exactly **Triton**. After the orchestrator "
            "enters its mandatory Triton-to-Gluon conversion phase, a direct implementation in "
            "`triton.experimental.gluon` is allowed and becomes the required framework for later "
            "submitted candidates. Do not submit a candidate that switches early, switches back, "
            "mixes Triton and Gluon compute kernels, or uses another DSL.\n"
        )
        candidate_framework = "the active Triton/Gluon phase"
    else:
        framework_rule = (
            f"- The submitted candidate's implementation framework is exactly **{framework}**. "
            "This is a hard final-candidate constraint, not a development-method restriction. Do not "
            "submit another DSL, a mixed-framework compute path, or a prebuilt operator.\n"
        )
        candidate_framework = framework
    if numerical_feedback_instruction is None:
        numerical_feedback_instruction = (
            "Read verification_artifacts/.atrex_long_horizon_verify/numerical_feedback.json, "
            "fix the implementation, rerun ordinary validation and submit the updated candidate. "
        )
    if correctness_instruction is None:
        correctness_instruction = (
            "Validate correctness with the immutable evaluator's ordinary random input generator "
            "over the full workload set. Native Atrex-Bench uses allclose for ordinary operators "
            "and relative L2 error <= 0.2 for NVFP4 operators. Keep the configured allclose "
            "tolerances and required random-seed coverage; correctness must pass before promotion.\n"
        )
    numerical_rule = ""
    if numerical_validation_enabled:
        numerical_rule = (
            "- Numerical review produces targeted supplemental tests, not subjective rejection verdicts. "
            "The supervisor executes those probes; measured failures return to the coding Agent for repair "
            "before baseline admission or episode completion. "
            f"{numerical_feedback_instruction}"
            "Passing the same probes closes the suggestion. "
            "Incomplete probes remain validation blockers; never edit trusted inputs, probes or tolerances.\n"
        )
    return (
        "## Optimization mode: production (hard gate)\n\n"
        "This generated section overrides any conflicting permissive framework or third-party-library "
        "guidance elsewhere in `CLAUDE.md`.\n\n"
        f"{framework_rule}"
        "- During development, you may use any language or DSL, PyTorch/ATen compute, prebuilt "
        "kernels/operators/math implementations, third-party compute libraries, hidden or external "
        "dispatch, dynamically loaded external implementations, compiler probes, microbenchmarks, "
        "reference implementations, and temporary prototypes to make evidence-backed decisions. "
        "These are development aids, not production candidates: remove them from the submitted "
        "Kernel source tree and runtime compute path unless they themselves satisfy the active "
        "final-candidate policy.\n"
        "- The V0 PyTorch reference wrapper is the only admitted baseline exception. When the optional "
        "Framework Baseline stage is enabled, its accepted V1, and every optimized candidate submitted "
        "through a candidate_ready report, "
        f"must implement the GPU computation directly in **{candidate_framework}**. Ordinary Git "
        "checkpoints and temporary probes are not terminal submissions.\n"
        "- The supervisor sends a Framework Baseline candidate, when that stage is enabled, and each "
        "candidate_ready production candidate to a separate, read-only policy reviewer. "
        "The reviewer judges the complete candidate by actual use, not package names: compiler bindings, "
        "header discovery, ABI/launch plumbing, and ordinary non-compute support utilities may be accepted "
        "when they only build or launch the candidate's self-authored kernel. In the submitted candidate, "
        "prebuilt kernels/operators/math implementations, third-party core computation, alternate DSLs, "
        "hidden dispatch, PyTorch/ATen compute fallbacks, external implementation loading, and wrapping "
        "another DSL as the selected framework remain forbidden. Ambiguous evidence is rejected.\n"
        f"- {correctness_instruction}"
        f"{numerical_rule}"
        "- When the active source contract declares `solution.json` as editable candidate metadata, "
        "keep it consistent with the implementation; otherwise do not edit it. "
        f"{candidate_submission_instruction}, inspect the complete candidate sources "
        "and `solution.json` against these rules. The supervisor will reject a candidate that lacks an "
        "evidence-backed production-policy verdict, even if it is faster and correct.\n"
    )


def workspace_policy_block(
    mode: str, framework: str, *, source_checks_enabled: bool = True
) -> str:
    directive = optimization_mode_directive(
        mode, framework, source_checks_enabled=source_checks_enabled
    ).rstrip()
    return f"{POLICY_BEGIN}\n\n{directive}\n\n{POLICY_END}\n"


def install_workspace_policy(
    workspace: Path,
    mode: str,
    framework: str,
    *,
    agent_runtime: str | None = None,
    update_tracked_files: bool = True,
    source_checks_enabled: bool = True,
    expose_evaluator_shapes: bool = True,
) -> None:
    """Persist immutable mode, framework, shape visibility, and runtime identity.

    Existing workspaces without ``agent_runtime`` remain readable. Their first
    explicit post-upgrade runtime is adopted before a session starts; later
    attempts to resume with another backend fail closed.
    """
    if mode not in OPTIMIZATION_MODE_CHOICES:
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
    if state.get("evaluator_shape_visibility") != requested_visibility:
        state["evaluator_shape_visibility"] = requested_visibility
        state_changed = True
    if state_changed:
        state_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )

    if not update_tracked_files:
        # Episode checkouts preserve their protected Git boundary. The current
        # policy is supplied by optimization_mode_directive in every prompt.
        return

    claude_path = workspace / "CLAUDE.md"
    current = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
    generated = workspace_policy_block(
        mode, framework, source_checks_enabled=source_checks_enabled
    )
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        current = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    claude_path.write_text(current, encoding="utf-8")

    gitignore = workspace / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    entry = f"/{MODE_STATE_FILE}"
    if entry not in ignored.splitlines():
        with gitignore.open("a", encoding="utf-8") as handle:
            if ignored and not ignored.endswith("\n"):
                handle.write("\n")
            handle.write("\n# orchestrator optimization-mode identity (local policy state)\n")
            handle.write(entry + "\n")
