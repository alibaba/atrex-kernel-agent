"""Legacy production policy with Light report and source-tree adapters."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from aka.legacy.acceptance import source_policy as legacy_policy
from aka.legacy.acceptance.source_policy import ProductionReviewer, reject_production_commit
from aka.legacy.task.source.inspection import framework_key, source_uses_gluon
from aka.legacy.workflow import policy as workflow_policy
from aka.legacy.workflow.policy import (
    MODE_STATE_FILE, OPTIMIZATION_MODE_CHOICES, POLICY_BEGIN, POLICY_END,
)


def optimization_mode_directive(
    mode: str, framework: str, *, source_checks_enabled: bool = True
) -> str:
    """Use the same production rules, without requiring Agent Git/private-file access."""
    if mode == "leaderboard":
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` and the injected task constraints. The requested "
            "framework is a recommended direction: compatible mixed or alternate implementations "
            "and third-party helper/kernel libraries are allowed when evidence supports them, "
            "unless a task-specific rule explicitly forbids them.\n"
        )
    return workflow_policy.optimization_mode_directive(
        mode, framework, source_checks_enabled=source_checks_enabled,
        numerical_feedback_instruction=(
            "Use the Supervisor's returned numerical feedback and Gateway Records; "
            "fix the implementation, rerun ordinary validation and resubmit the report. "
            "Input/reference/probe failures are validation blockers, not Kernel failures. "
        ),
        candidate_submission_instruction="Before submitting the report",
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
) -> None:
    """Persist immutable mode, framework, and optional campaign runtime identity.

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

    workspace.mkdir(parents=True, exist_ok=True)
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
        state = {"mode": mode, "framework": framework}
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


def production_structure_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
) -> list[str]:
    """Delegate single-file checks; retain the controller-pinned tree boundary."""
    from aka.light.workspace.candidate import read, required_layout

    try:
        artifact = read(workspace)
    except (OSError, ValueError) as exc:
        return [f"Invalid candidate source declaration: {exc}"]
    selected = framework_key(framework)
    if artifact.layout.framework is not None:
        if selected != artifact.layout.framework or require_gluon:
            return ["Candidate source format does not implement the selected framework"]
        return []
    if required_layout(selected).typed_only:
        return [f"{framework} candidates require native sources and a versioned manifest; Python-inline kernels are unsupported"]
    if not artifact.layout.archived:
        return legacy_policy.production_structure_violations(
            workspace, framework, require_gluon=require_gluon,
        )
    if selected not in legacy_policy.SUPPORTED_PRODUCTION_FRAMEWORKS:
        return [f"unsupported production framework: {framework}"]
    entrypoint = artifact.layout.python_entrypoint
    source = artifact.files[entrypoint].decode()
    try:
        ast.parse(source, filename=entrypoint)
    except SyntaxError as exc:
        return [f"Candidate entry point is not valid Python: {exc.msg} (line {exc.lineno})"]
    if require_gluon and not source_uses_gluon(source):
        return ["switching back from the accepted Gluon phase to Triton is forbidden"]
    return []


def production_kernel_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    production_reviewer: ProductionReviewer | None = None,
) -> list[str]:
    """Share Legacy's fail-closed reviewer gate for single-file and tree candidates."""
    return legacy_policy.production_kernel_violations(
        workspace, framework, require_gluon=require_gluon,
        production_reviewer=production_reviewer,
        structure_validator=production_structure_violations,
    )
