"""Prepare a selected Workflow launch using explicit inputs and host operations."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

from aka.legacy.task.candidate_workspace.filenames import FRAMEWORK_BASELINE_FILE
from aka.legacy.execution.recovery import configure_recovery, raise_if_environment_blocked
from aka.legacy.task.hardware import framework_workspace_suffix, supported_frameworks
from aka.legacy.execution.sandbox import check_ssh_environment, detect_arch
from aka.legacy.resources.submodules import ensure_submodules
from aka.legacy.execution.health import runtime_health_command
from .dispatch import dispatch_framework_campaigns
from .invocation import encode_launch


class FrameworkDispatchComplete(Exception):
    def __init__(self, exit_code):
        self.exit_code = exit_code


def recorded_workspace_arch(workspace: Path) -> str:
    try:
        marker = json.loads((workspace / FRAMEWORK_BASELINE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    arch = str(marker.get("arch") or "")
    return arch if re.fullmatch(r"sm_\d+|gfx[0-9a-fA-F]+", arch) else ""


def prepare_request(launch, problem, optimizer_script):
    request = launch.request
    settings, runtime = request.settings, request.runtime
    hardware = runtime.hardware or settings.platform
    base = launch.workspace_base
    if runtime.ssh:
        configure_recovery(workspace_base=base, raw_argv=launch.restart_arguments or encode_launch(launch),
            optimize_script=optimizer_script, sandbox_hardware=hardware,
            ssh_target=runtime.ssh, ssh_init=runtime.ssh_init,
            ssh_runtime_binds=launch.ssh_runtime_binds, ssh_gpu=runtime.ssh_gpu,
            health_command=runtime.health_command, poll_interval=launch.environment_poll_interval,
            runtime_health_command=runtime_health_command(
                sol=problem.initial_source.kind == "sol", framework=settings.framework))
        raise_if_environment_blocked()
        check_ssh_environment(base, hardware, runtime.ssh, runtime.ssh_init, runtime.health_command)
    arch = settings.arch or detect_arch(hardware, runtime.profile, runtime.url,
                                       runtime.ssh, runtime.ssh_init, runtime.health_command)
    suffix = launch.workspace_suffix or (framework_workspace_suffix(
        settings.framework, settings.platform, settings.optimization_mode) if settings.framework else "")
    if not arch and settings.framework:
        arch = recorded_workspace_arch(base / f"kernel_opt_{problem.name}_{suffix}")
        if arch:
            print(f"[orchestrator] sandbox arch detection failed; resuming with recorded {arch}",
                  file=sys.stderr, flush=True)
    ensure_submodules(settings.platform, arch or "")
    frameworks = ((settings.framework,) if settings.framework
                  else supported_frameworks(settings.platform, arch))
    reviewers = ",".join(f"{stage}[{','.join(names)}]" for stage, names in (
        ("v1", settings.framework_baseline_reviewers),
        ("fast", settings.fast_episode_reviewers), ("full", settings.full_episode_reviewers)))
    print(f"[orchestrator] op={problem.name} agent_cli={settings.agent_cli} "
          f"optimization_mode={settings.optimization_mode} platform={settings.platform} "
          f"agent_problem={problem.public_contract_source} sandbox_hardware={hardware} "
          f"sandbox_endpoint={runtime.ssh or runtime.url or runtime.profile or 'agate-config'} "
          f"frameworks={','.join(frameworks)} reviewers={reviewers} "
          f"runtime_arch={arch or 'UNKNOWN (detect failed)'} "
          "(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)
    if not settings.framework:
        raise FrameworkDispatchComplete(dispatch_framework_campaigns(
            launch.child_arguments, optimizer_script, frameworks, base.resolve(), arch,
            settings.platform, settings.optimization_mode, launch=launch))
    return replace(request, workspace=base / f"kernel_opt_{problem.name}_{suffix}",
                   name=problem.name + ("_" + suffix if suffix else ""),
                   settings=replace(settings, arch=arch))
