"""Legacy production policy with Light report and source-tree adapters."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from aka.legacy.acceptance import source_policy as legacy_policy
from aka.legacy.acceptance.source_policy import ProductionReviewer
from aka.legacy.task.source.inspection import framework_key, source_uses_gluon
from aka.legacy.workflow import policy as workflow_policy
from aka.legacy.workflow.policy import (
    MODE_STATE_FILE, OPTIMIZATION_MODE_CHOICES, POLICY_BEGIN, POLICY_END,
    historical_evaluator_shape_visibility,
    requested_evaluator_shape_visibility,
    visibility_mismatch_message,
)
from aka.light.acceptance.policy_modes import (
    canonical_submission_policy,
    dsl_policy,
)


def optimization_mode_directive(
    mode: str,
    framework: str,
    *,
    source_checks_enabled: bool = True,
    specify_dsl: bool = True,
    numerical_validation_enabled: bool = False,
) -> str:
    """Use the same production rules, without requiring Agent Git/private-file access."""
    if mode == "leaderboard":
        dsl = (
            "The requested framework is a recommended direction: compatible mixed or alternate "
            "implementations and third-party helper/kernel libraries are allowed when evidence "
            "supports them, unless a task-specific rule explicitly forbids them."
            if specify_dsl
            else "You may freely choose any evaluator-supported GPU implementation DSL."
        )
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` and the injected task constraints. " + dsl + "\n"
        )
    policy = canonical_submission_policy(mode, specify_dsl=specify_dsl)
    if policy == "fast":
        return (
            "## Optimization mode: fast\n\n"
            "Choose any evaluator-supported GPU implementation DSL or compatible mixture. "
            "Declared, reproducible third-party or prebuilt compute libraries may implement core "
            "computation. Dependencies, dynamic loading and manifests must remain explicit and "
            "auditable; hidden dispatch, undeclared downloads, secret external paths and unresolved "
            "implementation provenance are rejected.\n"
        )
    if policy == "performance":
        review_policy = (
            "The Supervisor independently reviews each submit-kernel candidate for source "
            "provenance, dependencies, loaders and manifest "
            "consistency without imposing a selected DSL.\n"
            if source_checks_enabled
            else "Independent source review is disabled for this task; correctness and the open "
            "DSL policy still apply.\n"
        )
        numerical_policy = (
            "Supplemental numerical review is enabled; measured probe failures are returned for repair.\n"
            if numerical_validation_enabled
            else "Supplemental numerical review is disabled for this campaign.\n"
        )
        return (
            (
                "## Optimization mode: production (open DSL)\n\n"
                if mode == "production"
                else "## Optimization mode: performance\n\n"
            )
            +
            "No DSL is prescribed. Choose a supported GPU implementation framework, or a compatible "
            "mixture, using the supplied expert guidance and measurement evidence. Recommendations "
            "inform the choice but are not hard submission constraints. Every final candidate submitted through "
            "submit-kernel must implement the "
            "operator's GPU computation directly with self-authored kernels. Compiler bindings, "
            "header discovery, ABI/launch plumbing and ordinary non-compute utilities are allowed; "
            "in the submitted candidate, PyTorch/ATen compute fallbacks, prebuilt "
            "kernels/operators/math implementations, third-party core computation, hidden dispatch "
            "and external implementation loading are forbidden. During development, these same "
            "mechanisms may be used as temporary references, probes, microbenchmarks or prototypes, "
            "but they must not remain in the submitted Kernel source tree or runtime compute path. "
            "When the active SourceProvider declares `solution.json` as editable candidate metadata, "
            "keep it consistent with the actual implementation; otherwise do not edit it.\n"
            "Validate correctness with the immutable evaluator over the full workload. Read "
            "`metadata.json` when present; every declared correctness condition must pass. "
            + review_policy + numerical_policy
        )
    if policy != "ultra":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    if not source_checks_enabled:
        return workflow_policy.optimization_mode_directive(
            "production",
            framework,
            source_checks_enabled=False,
            numerical_validation_enabled=numerical_validation_enabled,
        )

    framework_rule = (
        f"The submitted candidate's implementation framework is exactly **{framework}**. "
        "This is a hard final-candidate constraint, not a development-method restriction."
    )
    numerical_rule = (
        "Numerical review produces targeted Supervisor-run probes. A measured failure is returned "
        "for repair; fix the candidate, rerun ordinary validation, and resubmit without changing "
        "trusted probes or thresholds."
        if numerical_validation_enabled
        else "Supplemental numerical review is disabled for this campaign."
    )
    return (
        (
            "## Optimization mode: production (hard gate)\n\n"
            if mode == "production"
            else "## Optimization mode: ultra\n\n"
        )
        +
        f"- {framework_rule}\n"
        "- During development, you may use any language or DSL, PyTorch/ATen, prebuilt or "
        "third-party implementations, compiler probes, and temporary prototypes. Remove them from "
        "the submitted Kernel source tree and runtime compute path unless they satisfy the final policy.\n"
        "- V0 is the only admitted PyTorch baseline exception. Every submit-kernel production "
        "candidate must implement the GPU computation directly with self-authored Kernel code.\n"
        "- The independent Production Review examines actual final use. Compiler, header, ABI, "
        "launch, and non-compute support plumbing is allowed; PyTorch/ATen compute fallback, "
        "prebuilt Kernels, third-party compute libraries performing core computation, dynamically loaded external implementations, "
        "hidden dispatch, wrapping another DSL as the selected framework, and ambiguous provenance are rejected.\n"
        "- Validate the immutable full workload. Before reasoning about numerical acceptance, read "
        "workspace `metadata.json` when present; its benchmark correctness contract is authoritative. "
        "Do not assume, invent, or override tolerances; one passing check never overrides another reported failure.\n"
        f"- {numerical_rule}\n"
        "- Keep editable `solution.json`, when present, consistent with the implementation. The "
        "Supervisor returns review or validation failures for repair before accepting the submission.\n"
    )


def workspace_policy_block(
    mode: str,
    framework: str,
    *,
    source_checks_enabled: bool = True,
    specify_dsl: bool = True,
    numerical_validation_enabled: bool = False,
) -> str:
    directive = optimization_mode_directive(
        mode,
        framework,
        source_checks_enabled=source_checks_enabled,
        specify_dsl=specify_dsl,
        numerical_validation_enabled=numerical_validation_enabled,
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
    specify_dsl: bool = True,
    numerical_validation_enabled: bool = False,
    expose_evaluator_shapes: bool = True,
) -> None:
    """Persist immutable mode, framework, shape visibility, and runtime identity.

    Existing workspaces without ``agent_runtime`` remain readable. Their first
    explicit post-upgrade runtime is adopted before a session starts; later
    attempts to resume with another backend fail closed.
    """
    if mode not in OPTIMIZATION_MODE_CHOICES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    policy_framework = framework if specify_dsl else ""
    requested_runtime = (
        str(agent_runtime).strip() if agent_runtime is not None else None
    )
    if agent_runtime is not None and not requested_runtime:
        raise ValueError("agent_runtime must be a non-empty runtime id")
    if type(expose_evaluator_shapes) is not bool:
        raise TypeError("expose_evaluator_shapes must be a boolean")
    requested_visibility = "exposed" if expose_evaluator_shapes else "hidden"

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
        existing_specify_dsl = state.get("specify_dsl", True)
        existing_visibility = state.get("evaluator_shape_visibility")
        if existing_visibility is None:
            existing_visibility = "hidden" if (workspace / "agent_problem.json").is_file() else "exposed"
        if (
            existing_mode != mode
            or existing_framework != policy_framework
            or existing_specify_dsl != specify_dsl
            or existing_visibility != requested_visibility
        ):
            raise RuntimeError(
                "workspace policy mismatch: "
                "recorded mode/framework/specify_dsl/shape_visibility="
                f"{existing_mode}/{existing_framework}/{existing_specify_dsl}/{existing_visibility}, "
                f"requested={mode}/{policy_framework}/{specify_dsl}/{requested_visibility}"
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
            "framework": policy_framework,
            "specify_dsl": specify_dsl,
            "evaluator_shape_visibility": requested_visibility,
        }
        if requested_runtime:
            state["agent_runtime"] = requested_runtime
        state_changed = True

    if state.get("source_checks_enabled") != source_checks_enabled:
        state["source_checks_enabled"] = source_checks_enabled
        state_changed = True
    if state.get("numerical_validation_enabled") != numerical_validation_enabled:
        state["numerical_validation_enabled"] = numerical_validation_enabled
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
        # policy is supplied by the Runtime when it projects Agent CLAUDE.md.
        return

    claude_path = workspace / "CLAUDE.md"
    current = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
    generated = workspace_policy_block(
        mode,
        policy_framework,
        source_checks_enabled=source_checks_enabled,
        specify_dsl=specify_dsl,
        numerical_validation_enabled=numerical_validation_enabled,
    )
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        current = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    claude_path.write_text(current, encoding="utf-8")

def production_structure_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    specify_dsl: bool = True,
) -> list[str]:
    """Validate the editable Kernel entry point and source-tree boundary."""
    from aka.light.workspace.sources import policy_at, collect

    policy = policy_at(workspace)
    if policy is None:
        return legacy_policy.production_structure_violations(
            workspace,
            framework if specify_dsl else "Cuda",
            require_gluon=require_gluon if specify_dsl else False,
        )
    if (
        specify_dsl
        and framework_key(framework) not in legacy_policy.SUPPORTED_PRODUCTION_FRAMEWORKS
    ):
        return [f"unsupported production framework: {framework}"]
    try:
        files = collect(workspace, policy)
    except (OSError, ValueError) as exc:
        return [f"Invalid candidate source tree: {exc}"]
    for name, data in files.items():
        if Path(name).suffix != ".py":
            continue
        try:
            ast.parse(data.decode("utf-8"), filename=name)
        except SyntaxError as exc:
            return [f"{name} is not valid Python: {exc.msg} (line {exc.lineno})"]
    if specify_dsl and require_gluon and not any(
        source_uses_gluon(data.decode("utf-8", errors="ignore"))
        for data in files.values()
    ):
        return ["switching back from the accepted Gluon phase to Triton is forbidden"]
    return []


def production_kernel_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    specify_dsl: bool = True,
    production_reviewer: ProductionReviewer | None = None,
) -> list[str]:
    """Share Legacy's fail-closed reviewer gate for raw and tree candidates."""
    def validate_structure(path, selected_framework, *, require_gluon=False):
        return production_structure_violations(
            path,
            selected_framework,
            require_gluon=require_gluon,
            specify_dsl=specify_dsl,
        )

    return legacy_policy.production_kernel_violations(
        workspace, framework, require_gluon=require_gluon,
        production_reviewer=production_reviewer,
        structure_validator=validate_structure,
    )
