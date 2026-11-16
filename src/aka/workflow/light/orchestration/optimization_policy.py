"""Final submission policy, independent of development language and Git state."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from aka.acceptance import source_policy as shared_policy
from aka.acceptance.source_policy import ProductionReviewer
from aka.task.source.inspection import framework_key
from aka.task.problem.visibility import requested_evaluator_shape_visibility
from aka.workflow.light.long_horizon.policy_modes import SUBMISSION_POLICY_MODES as OPTIMIZATION_MODE_CHOICES

POLICY_BEGIN = "<!-- AKA SUBMISSION POLICY BEGIN -->"
POLICY_END = "<!-- AKA SUBMISSION POLICY END -->"
from aka.workflow.light.long_horizon.policy_modes import (
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
    policy = canonical_submission_policy(mode, specify_dsl=specify_dsl)
    if policy == "fast":
        return (
            "## Optimization mode: fast\n\n"
            "Choose any evaluator-supported GPU implementation DSL or compatible mixture. "
            "Declared, reproducible third-party or prebuilt compute libraries may implement core "
            "computation. Fast submissions do not run the independent final-source policy Review; "
            "they still require a valid source tree and a passing complete Evaluate.\n"
        )
    if policy == "performance":
        review_policy = (
            "The Supervisor independently reviews whether each submit-kernel candidate delegates "
            "target-operator computation to a framework operator, prebuilt Kernel or external "
            "math/operator library, without imposing a selected DSL.\n"
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
            "## Optimization mode: performance\n\n"
            +
            "No DSL is prescribed. Choose a supported GPU implementation framework, or a compatible "
            "mixture, using the supplied expert guidance and measurement evidence. Recommendations "
            "inform the choice but are not hard submission constraints. Every final candidate submitted through "
            "submit-kernel must implement the "
            "operator's GPU computation directly with self-authored kernels. Compiler bindings, "
            "header discovery, ABI/launch plumbing, allocation, metadata access and ordinary "
            "non-compute utilities are allowed; in the submitted candidate, calls to PyTorch/ATen "
            "operators, prebuilt kernels or external math/operator libraries that implement any "
            "part of the target result are forbidden. During development, these same "
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
        "## Optimization mode: ultra\n\n"
        +
        f"- {framework_rule}\n"
        "- During development, you may use any language or DSL, PyTorch/ATen, prebuilt or "
        "third-party implementations, compiler probes, and temporary prototypes. Remove them from "
        "the submitted Kernel source tree and runtime compute path unless they satisfy the final policy.\n"
        "- Every submit-kernel Ultra "
        "candidate must implement the GPU computation directly with self-authored Kernel code.\n"
        "- The independent final-source Review rejects calls to PyTorch/ATen operators, prebuilt "
        "Kernels or external math/operator libraries that implement any part of the target result. "
        "Compiler, header, ABI, launch, allocation, metadata and other non-compute support plumbing "
        "are allowed. The configured DSL remains a hard submission constraint.\n"
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
    """Pin policy in the Campaign manifest and render Agent guidance."""
    from aka.core.filesystem import durable_write_json
    from aka.workflow.light.support.paths import supervisor_campaign_base_root, _campaign_lock
    if mode not in OPTIMIZATION_MODE_CHOICES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    if mode == "ultra" and (not specify_dsl or not framework.strip()):
        raise ValueError("Ultra requires an explicit DSL")
    policy_framework = framework if mode == "ultra" else ""
    visibility = requested_evaluator_shape_visibility(workspace, mode, expose_evaluator_shapes)
    manifest_path = supervisor_campaign_base_root(workspace) / "campaign.json"
    with _campaign_lock(manifest_path.parent):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {"mode": mode, "framework": policy_framework,
                    "evaluator_shape_visibility": visibility, "agent_runtime": agent_runtime}
        previous = manifest.get("policy")
        if previous is not None and previous != identity:
            raise RuntimeError(f"workspace policy mismatch: recorded={previous}, requested={identity}")
        if previous is None:
            manifest["policy"] = identity
            durable_write_json(manifest_path, manifest)

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
    from aka.task.source.formats import policy_at, collect

    policy = policy_at(workspace)
    if policy is None:
        return shared_policy.production_structure_violations(
            workspace,
            framework if specify_dsl else "Cuda",
            require_gluon=require_gluon if specify_dsl else False,
        )
    if (
        specify_dsl
        and framework_key(framework) not in shared_policy.SUPPORTED_PRODUCTION_FRAMEWORKS
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
    return []


def production_kernel_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    specify_dsl: bool = True,
    production_reviewer: ProductionReviewer | None = None,
) -> list[str]:
    """Use the public reviewer gate with the complete candidate tree."""
    def validate_structure(path, selected_framework, *, require_gluon=False):
        return production_structure_violations(
            path,
            selected_framework,
            require_gluon=require_gluon,
            specify_dsl=specify_dsl,
        )

    return shared_policy.production_kernel_violations(
        workspace, framework, require_gluon=require_gluon,
        production_reviewer=production_reviewer,
        structure_validator=validate_structure,
    )
