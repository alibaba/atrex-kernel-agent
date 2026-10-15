"""Workflow policy implementation and callable interface."""
from __future__ import annotations

from aka.legacy.task.source.inspection import framework_key
import json
from pathlib import Path


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")


MODE_STATE_FILE = ".orchestrator_mode.json"


POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"


POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"


def optimization_mode_directive(
    mode: str, framework: str, *, source_checks_enabled: bool = True
) -> str:
    """Self-contained policy block injected into every coding-agent prompt."""
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
        return (
            "## Optimization mode: production (multi-GPU Agate)\n\n"
            "This generated policy overrides conflicting source/framework restrictions elsewhere "
            "in the workspace. The evaluator requests at least two GPUs from Agate. Production "
            "source inspection and independent source-policy review are disabled for this campaign; "
            "communication libraries, PyTorch distributed operations and alternate implementations "
            "are allowed. No self-authored framework baseline or Triton-to-Gluon conversion is required.\n"
            "Correctness, numerical probes, performance validation and promotion requirements remain "
            "unchanged. Use the immutable evaluator, full workload and required random seeds; preserve "
            "input/output semantics and include all per-call communication and synchronization in "
            "the measured work. Never modify trusted inputs, tolerances or evaluation records.\n"
        )
    if framework_key(framework) == "triton":
        framework_rule = (
            "- The initial implementation framework is exactly **Triton**. After the orchestrator "
            "enters its mandatory Triton-to-Gluon conversion phase, a direct implementation in "
            "`triton.experimental.gluon` is allowed and becomes the required framework for later "
            "iterations. Do not switch early, switch back, mix Triton and Gluon compute kernels, "
            "or use any other DSL.\n"
        )
        candidate_framework = "the active Triton/Gluon phase"
    else:
        framework_rule = (
            f"- The implementation framework is exactly **{framework}**. It is a hard constraint, "
            "not a recommendation. Do not switch to another DSL, mix another kernel framework "
            "into the candidate, or replace the implementation with a prebuilt operator.\n"
        )
        candidate_framework = framework
    return (
        "## Optimization mode: production (hard gate)\n\n"
        "This generated section overrides any conflicting permissive framework or third-party-library "
        "guidance elsewhere in `CLAUDE.md`.\n\n"
        f"{framework_rule}"
        "- The V0 PyTorch reference wrapper is the only baseline exception. Every optimized candidate "
        f"committed after V0 must implement the GPU computation directly in **{candidate_framework}**.\n"
        "- The supervisor sends every production candidate to a separate, read-only policy reviewer. "
        "The reviewer judges the complete candidate by actual use, not package names: compiler bindings, "
        "header discovery, ABI/launch plumbing, and ordinary non-compute support utilities may be accepted "
        "when they only build or launch the candidate's self-authored kernel. Prebuilt kernels/operators/math "
        "implementations, alternate DSLs, hidden dispatch, PyTorch compute fallbacks, and external "
        "implementation loading remain forbidden. Ambiguous evidence is rejected.\n"
        "- Validate correctness with the immutable evaluator's ordinary random input generator "
        "over the full workload set. Native Atrex-Bench uses allclose for ordinary operators "
        "and relative L2 error <= 0.2 for NVFP4 operators. Keep the configured allclose "
        "tolerances and required random-seed coverage; correctness must pass before promotion.\n"
        "- Numerical review produces targeted supplemental tests, not subjective rejection verdicts. "
        "The supervisor executes those probes; measured failures return to the coding Agent for repair "
        "before baseline admission or episode completion. Read verification_artifacts/"
        ".atrex_long_horizon_verify/numerical_feedback.json, fix the implementation, rerun ordinary "
        "validation and submit the updated candidate. Passing the same probes closes the suggestion. "
        "Incomplete probes remain validation blockers; never edit trusted inputs, probes or tolerances.\n"
        "- Keep `solution.json` consistent with the implementation. Before committing, inspect `kernel.py` "
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

