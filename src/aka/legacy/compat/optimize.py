"""Historical Python entry surface; never selected by the normal runtime."""
from __future__ import annotations
import argparse, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
from typing import Optional
from orchestrator.process_launch import optimizer_entrypoint
from orchestrator import agent_runtime as _agent_runtime
from orchestrator.campaign import Campaign
from orchestrator.constants import (
    AGENT_CLI_CHOICES,
    DEFAULT_CONVERT_AFTER,
    DEFAULT_FAST_EPISODES,
    DEFAULT_FAST_TRIALS,
    DEFAULT_HANDOFF_RESUMES,
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_VERIFY_REPEATS,
    DEFAULT_VERIFY_RUN_TIMEOUT,
    FRAMEWORK_BASELINE_FILE,
    FRAMEWORK_BASELINE_MODES,
    FRAMEWORK_BASELINE_TIMEOUT_S,
    MAX_SANDBOX_TIMEOUT,
)
from orchestrator.environment_recovery import (
    DEFAULT_SSH_HEALTH_COMMAND,
    ENVIRONMENT_TEMPFAIL,
    EnvironmentUnavailable,
    configure_recovery,
    current_recovery_context,
    environment_is_blocked,
    launch_recovery_monitor,
    raise_if_environment_blocked,
    signal_restart_ready,
)
from orchestrator.hardware import (
    _workspace_slug,
    framework_workspace_suffix,
    supported_frameworks,
)
from orchestrator.trace_retention import (
    write_trace_retention_manifest,
)
from orchestrator.optimization_policy import (
    OPTIMIZATION_MODE_CHOICES,
)
from orchestrator.plugins import PluginError, PluginRegistry
from orchestrator.session_io import (
    check_ssh_environment,
    detect_arch,
    ensure_submodules,
)
from orchestrator.ssh_health import runtime_health_command
from orchestrator.workspace_state import (
    latest_version,
    read_memory,
)

def _without_cli_options(argv: list[str], option_names: tuple[str, ...]) -> list[str]:
    """Remove value-taking options from argv before adding canonical child values."""
    cleaned: list[str] = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg in option_names:
            skip_value = True
            continue
        if any(arg.startswith(name + "=") for name in option_names):
            continue
        cleaned.append(arg)
    return cleaned


def _recorded_workspace_arch(workspace: Path) -> str:
    try:
        marker = json.loads((workspace / FRAMEWORK_BASELINE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    arch = str(marker.get("arch") or "")
    return arch if re.fullmatch(r"sm_\d+|gfx[0-9a-fA-F]+", arch) else ""


def dispatch_framework_campaigns(
    argv: list[str],
    frameworks: tuple[str, ...],
    workspace_base: Path,
    arch: str,
    platform: str,
    optimization_mode: str = "leaderboard",
) -> int:
    """Launch explicit-framework optimizer children concurrently and wait for all.

    Each child receives a flat framework/hardware suffix, so its eventual
    workspace is ``<workspace_base>/kernel_opt_<op>_<framework>_<platform>``
    (with ``_production`` appended in production mode). Budgets remain per campaign; a failed
    framework does not cancel the other independent campaigns.
    """
    common_argv = _without_cli_options(
        argv, ("--framework", "--workspace", "--arch", "--workspace-suffix")
    )
    children: list[tuple[str, str, subprocess.Popen[str]]] = []
    workspace_base.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, int]] = []

    def stop_children() -> None:
        # Framework children and every coding-agent session deliberately use
        # separate process groups.  Capture the whole descendant group set
        # before signaling: once a framework coordinator exits, its Claude
        # session is reparented to PID 1 and can no longer be discovered from
        # the dispatcher, which previously left orphan optimizers behind.
        process_groups: set[int] = set()
        for _, _, proc in children:
            if proc.poll() is not None:
                continue
            try:
                process_groups.add(os.getpgid(proc.pid))
            except ProcessLookupError:
                continue
            for pid, _argv in _agent_runtime.descendant_process_commands(proc.pid):
                try:
                    process_groups.add(os.getpgid(pid))
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }

    def interrupt_dispatch(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for handled_signal in handled_signals:
        signal.signal(handled_signal, interrupt_dispatch)
    try:
        for framework in frameworks:
            workspace_suffix = framework_workspace_suffix(
                framework, platform, optimization_mode
            )
            cmd = [
                sys.executable,
                str(optimizer_entrypoint()),
                *common_argv,
                "--framework",
                framework,
                "--workspace",
                str(workspace_base),
                "--workspace-suffix",
                workspace_suffix,
            ]
            if arch:
                cmd += ["--arch", arch]
            from aka.legacy.application.processes import child_environment as launch_environment
            child_environment = os.environ.copy()
            for name, value in launch_environment().items():
                if value is None:
                    child_environment.pop(name, None)
                else:
                    child_environment[name] = value
            from orchestrator._bootstrap import execution_modules
            execution, _runtime = execution_modules()
            child_environment = execution.execution_environment(child_environment)
            child_environment["ATREX_ENVIRONMENT_RECOVERY_OWNER"] = "0"
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                text=True,
                env=child_environment,
            )
            children.append((framework, workspace_suffix, proc))
            print(
                f"[orchestrator] dispatched framework={framework} pid={proc.pid} "
                f"workspace_suffix={workspace_suffix} work_root={workspace_base}",
                flush=True,
            )

        remaining = {proc.pid: (framework, proc) for framework, _, proc in children}
        while remaining:
            for pid, (framework, proc) in list(remaining.items()):
                returncode = proc.poll()
                if returncode is None:
                    continue
                remaining.pop(pid)
                print(
                    f"[orchestrator] framework={framework} finished exit={returncode}",
                    flush=True,
                )
                if returncode != 0:
                    failed.append((framework, returncode))
                if returncode == ENVIRONMENT_TEMPFAIL or environment_is_blocked():
                    stop_children()
                    return ENVIRONMENT_TEMPFAIL
            if remaining:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print(
            "[orchestrator] interrupt: stopping framework campaigns",
            file=sys.stderr,
            flush=True,
        )
        stop_children()
        return 130
    except BaseException:
        stop_children()
        raise
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)

    if failed:
        summary = ", ".join(f"{name}={code}" for name, code in failed)
        print(
            f"[orchestrator] framework campaign failures: {summary}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def _run_main(argv: Optional[list[str]] = None, *, dependencies=None) -> int:
    if (dependencies or {}).get("workflow") is None:
        from aka.legacy.compat.assembly import default_workflow

        workflow = default_workflow(task=dependencies)
        try:
            return _run_main(
                argv, dependencies={**(dependencies or {}), "workflow": workflow}
            )
        finally:
            workflow.close()
    from .cli import parse_arguments
    args, raw_argv = parse_arguments(argv)
    sandbox_hardware = args.sandbox_hardware or args.platform

    from aka.legacy.compat.problem import problem_request
    from aka.legacy.compat.startup import startup_preparation, FrameworkDispatchComplete

    factory = dependencies["workflow"]
    source = problem_request(
        args.op_dir, args.optimization_mode, workspace=args.workspace,
        suffix=args.workspace_suffix or f"{args.framework}_{args.platform}",
    )
    from aka.legacy.compat.workflow import create_campaign as adapt_campaign
    create_campaign = lambda **options: adapt_campaign(factory, **options)
    with startup_preparation(sys.modules[__name__], args, raw_argv) as startup:
        try:
            campaign = create_campaign(
                task_dependencies=dependencies,
                problem=source,
                name=source.directory.name,
                platform=args.platform,
                framework=args.framework,
                notes=args.notes,
                arch=args.arch,
                sandbox_hardware=sandbox_hardware,
                sandbox_profile=args.sandbox_profile,
                sandbox_url=args.sandbox_url,
                sandbox_ssh=args.sandbox_ssh,
                sandbox_ssh_init=args.sandbox_ssh_init,
                sandbox_ssh_gpu=args.sandbox_ssh_gpu,
                sandbox_health_command=args.sandbox_health_command,
                sandbox_timeout=args.sandbox_timeout,
                agent_cli=args.agent_cli,
                optimization_mode=args.optimization_mode,
                work_dir=args.workspace,
                workspace_suffix=args.workspace_suffix,
                max_iters=args.max_iters,
                fast_episodes=args.fast_episodes,
                fast_trials=args.fast_trials,
                token_budget=args.token_budget,
                target_util=args.target_util,
                setup_timeout=args.setup_timeout,
                max_stall=args.max_stall,
                framework_baseline=args.framework_baseline,
                framework_baseline_timeout=args.framework_baseline_timeout,
                handoff_resumes=args.handoff_resumes,
                production_review_timeout=args.production_review_timeout,
                verify_repeats=args.verify_repeats,
                verify_run_timeout=args.verify_run_timeout,
                min_improvement_pct=args.min_improvement_pct,
                long_reviewer_session=args.long_reviewer_session,
                v1_ask_codex=args.v1_ask_codex,
                v1_ask_qoder=args.v1_ask_qoder,
                fast_episode_ask_codex=args.fast_episode_ask_codex,
                fast_episode_ask_qoder=args.fast_episode_ask_qoder,
                full_episode_ask_codex=args.full_episode_ask_codex,
                full_episode_ask_qoder=args.full_episode_ask_qoder,
                convert_after=args.convert_after,
            )
        except FrameworkDispatchComplete as completed:
            return completed.exit_code
        arch = startup.arch
    trace_status = "failed"
    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }

    def interrupt_campaign(signum: int, _frame: object) -> None:
        nonlocal trace_status
        trace_status = "interrupted"
        raise SystemExit(128 + signum)

    for handled_signal in handled_signals:
        signal.signal(handled_signal, interrupt_campaign)
    try:
        # Workflow completes setup before the host advertises recovery readiness.
        campaign.prepare(recovery_ready=signal_restart_ready if args.sandbox_ssh else None)
        campaign.run()
        trace_status = "completed"
        return 0
    except KeyboardInterrupt:
        trace_status = "interrupted"
        raise
    finally:
        write_trace_retention_manifest(
            campaign.workspace,
            trace_status,
            hardware={
                "platform": args.platform,
                "arch": arch or "",
                "sandbox_hardware": sandbox_hardware,
            },
        )
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def _run_application(argv: Optional[list[str]] = None, *, dependencies=None) -> int:
    try:
        result = _run_main(argv, dependencies=dependencies)
    except EnvironmentUnavailable:
        result = ENVIRONMENT_TEMPFAIL
    if result == ENVIRONMENT_TEMPFAIL or environment_is_blocked():
        context = current_recovery_context()
        supervised = os.environ.get("ATREX_ENVIRONMENT_RESTART_SUPERVISED") == "1"
        pid = (
            launch_recovery_monitor(context)
            if context is not None and not supervised
            else None
        )
        if pid is not None:
            print(
                "[orchestrator] remote GPU environment unavailable; optimization stopped; "
                f"recovery monitor pid={pid} state={context.directory}",
                file=sys.stderr,
                flush=True,
            )
        return ENVIRONMENT_TEMPFAIL
    return result


def main(argv: Optional[list[str]] = None) -> int:
    from orchestrator._bootstrap import run

    return run(argv)
