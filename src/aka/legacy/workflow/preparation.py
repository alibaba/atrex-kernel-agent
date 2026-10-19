"""Prepare a selected Workflow launch using explicit inputs and host operations."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

from aka.legacy.task.candidate_workspace.filenames import FRAMEWORK_BASELINE_FILE
from aka.legacy.task.hardware import framework_workspace_suffix, supported_frameworks


def recorded_workspace_arch(workspace: Path) -> str:
    try:
        marker = json.loads((workspace / FRAMEWORK_BASELINE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    arch = str(marker.get("arch") or "")
    return arch if re.fullmatch(r"sm_\d+|gfx[0-9a-fA-F]+", arch) else ""


def prepare_request(launch, problem, *, frozen=False):
    request = launch.request
    settings, runtime = request.settings, request.runtime
    hardware = runtime.hardware or settings.platform
    base = launch.workspace_base
    from aka.legacy.execution.configuration import prepare_runtime
    arch, endpoint = prepare_runtime(launch, problem)
    suffix = launch.workspace_suffix or (framework_workspace_suffix(
        settings.framework, settings.platform, settings.optimization_mode) if settings.framework else "")
    if not arch and settings.framework:
        arch = recorded_workspace_arch(base / f"kernel_opt_{problem.name}_{suffix}")
        if arch:
            print(f"[orchestrator] sandbox arch detection failed; resuming with recorded {arch}",
                  file=sys.stderr, flush=True)
    frameworks = ((settings.framework,) if settings.framework
                  else supported_frameworks(settings.platform, arch))
    reviewers = ",".join(f"{stage}[{','.join(names)}]" for stage, names in (
        ("v1", settings.framework_baseline_reviewers),
        ("fast", settings.fast_episode_reviewers), ("full", settings.full_episode_reviewers)))
    print(f"[orchestrator] op={problem.name} agent_cli={settings.agent_cli} "
          f"optimization_mode={settings.optimization_mode} platform={settings.platform} "
          f"agent_problem={problem.public_contract_source} sandbox_hardware={hardware} "
          f"sandbox_endpoint={endpoint} "
          f"frameworks={','.join(frameworks)} reviewers={reviewers} "
          f"runtime_arch={arch or 'UNKNOWN (detect failed)'} "
          "(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)
    if frozen:
        if arch and settings.arch and arch != settings.arch:
            raise ValueError('checkpoint runtime architecture differs from its frozen request')
        return request, frameworks
    return replace(request, workspace=base / f"kernel_opt_{problem.name}_{suffix}",
                   name=problem.name + ("_" + suffix if suffix else ""),
                   settings=replace(settings, arch=arch)), frameworks
