"""Legacy startup preparation bound outside the public Workflow Interface."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
import sys


class FrameworkDispatchComplete(Exception):
    def __init__(self, exit_code):
        self.exit_code = exit_code


@dataclass
class StartupPreparation:
    module: object
    args: object
    raw_argv: list[str]
    arch: str = ""


_preparation = ContextVar("legacy_startup_preparation", default=None)


@contextmanager
def startup_preparation(module, args, raw_argv):
    state = StartupPreparation(module, args, raw_argv, args.arch)
    token = _preparation.set(state)
    try:
        yield state
    finally:
        _preparation.reset(token)


def prepare_request(request, problem):
    state = _preparation.get()
    if state is None:
        return request
    args, raw_argv = state.args, state.raw_argv
    sandbox_hardware = args.sandbox_hardware or args.platform
    if args.sandbox_ssh:
        workspace_base = Path(args.workspace) if args.workspace else Path.cwd()
        state.module.configure_recovery(
            workspace_base=workspace_base,
            raw_argv=raw_argv,
            optimize_script=state.module.optimizer_entrypoint(),
            sandbox_hardware=sandbox_hardware,
            ssh_target=args.sandbox_ssh,
            ssh_init=args.sandbox_ssh_init,
            ssh_runtime_binds=args.sandbox_ssh_runtime_bind,
            ssh_gpu=args.sandbox_ssh_gpu,
            health_command=args.sandbox_health_command,
            poll_interval=args.environment_poll_interval,
            runtime_health_command=state.module.runtime_health_command(
                sol=problem.initial_source.kind == "sol", framework=args.framework
            ),
        )
        state.module.raise_if_environment_blocked()
        state.module.check_ssh_environment(
            workspace_base, sandbox_hardware, args.sandbox_ssh,
            args.sandbox_ssh_init, args.sandbox_health_command,
        )

    arch = args.arch or state.module.detect_arch(
        sandbox_hardware,
        args.sandbox_profile,
        args.sandbox_url,
        args.sandbox_ssh,
        args.sandbox_ssh_init,
        args.sandbox_health_command,
    )
    if not arch and args.framework:
        suffix = args.workspace_suffix or state.module.framework_workspace_suffix(
            args.framework, args.platform, args.optimization_mode
        )
        base = Path(args.workspace) if args.workspace else Path.cwd()
        arch = state.module._recorded_workspace_arch(base / f"kernel_opt_{problem.name}_{suffix}")
        if arch:
            print(
                f"[orchestrator] sandbox arch detection failed; resuming with recorded {arch}",
                file=sys.stderr,
                flush=True,
            )
    state.module.ensure_submodules(args.platform, arch or "")
    frameworks = (
        (args.framework,)
        if args.framework
        else state.module.supported_frameworks(args.platform, arch)
    )
    print(
        f"[orchestrator] op={problem.name} agent_cli={args.agent_cli} "
        f"optimization_mode={args.optimization_mode} platform={args.platform} "
        f"agent_problem={problem.public_contract_source} "
        f"sandbox_hardware={sandbox_hardware} "
        "sandbox_endpoint="
        f"{args.sandbox_ssh or args.sandbox_url or args.sandbox_profile or 'agate-config'} "
        f"frameworks={','.join(frameworks)} "
        "reviewers="
        f"v1[codex={'on' if args.v1_ask_codex else 'off'},"
        f"qoder={'on' if args.v1_ask_qoder else 'off'}],"
        f"fast[codex={'on' if args.fast_episode_ask_codex else 'off'},"
        f"qoder={'on' if args.fast_episode_ask_qoder else 'off'}],"
        f"full[codex={'on' if args.full_episode_ask_codex else 'off'},"
        f"qoder={'on' if args.full_episode_ask_qoder else 'off'}] "
        "runtime_arch="
        f"{arch or 'UNKNOWN (detect failed)'} "
        f"(device name / vendor-smi may be desensitized; trusting the runtime API)",
        flush=True,
    )

    if not args.framework:
        base = Path(args.workspace).resolve() if args.workspace else Path.cwd()
        raise FrameworkDispatchComplete(state.module.dispatch_framework_campaigns(
            raw_argv,
            frameworks,
            base,
            arch,
            args.platform,
            args.optimization_mode,
        ))

    workspace_suffix = args.workspace_suffix or state.module.framework_workspace_suffix(
        args.framework, args.platform, args.optimization_mode
    )

    state.arch = arch
    base = Path(args.workspace) if args.workspace else Path.cwd()
    return replace(
        request,
        workspace=base / f"kernel_opt_{problem.name}_{workspace_suffix}",
        name=problem.name + ("_" + workspace_suffix if workspace_suffix else ""),
        settings=replace(request.settings, arch=arch),
    )
