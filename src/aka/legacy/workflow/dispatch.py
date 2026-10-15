"""Supervise framework campaigns using explicitly supplied process arguments."""
from __future__ import annotations
import os, signal, subprocess, sys, time
from pathlib import Path
from aka.legacy.agent.process import descendant_process_commands
from aka.legacy.task.hardware import framework_workspace_suffix
from aka.legacy.execution.recovery import ENVIRONMENT_TEMPFAIL, environment_is_blocked


def dispatch_framework_campaigns(
    child_arguments: tuple[str, ...],
    optimizer_script: Path,
    frameworks: tuple[str, ...],
    workspace_base: Path,
    arch: str,
    platform: str,
    optimization_mode: str = "leaderboard",
    *, launch=None,
) -> int:
    """Launch explicit-framework optimizer children concurrently and wait for all.

    Each child receives a flat framework/hardware suffix, so its eventual
    workspace is ``<workspace_base>/kernel_opt_<op>_<framework>_<platform>``
    (with ``_production`` appended in production mode). Budgets remain per campaign; a failed
    framework does not cancel the other independent campaigns.
    """
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
            for pid, _argv in descendant_process_commands(proc.pid):
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
            substitutions = {"framework": framework, "workspace": str(workspace_base),
                             "suffix": workspace_suffix, "arch": arch or ""}
            if child_arguments:
                arguments = list(child_arguments)
                for index, key in launch.child_bindings:
                    arguments[index] = substitutions[key]
            else:
                from dataclasses import replace
                from .invocation import encode_launch
                child_request = replace(launch.request, settings=replace(
                    launch.request.settings, framework=framework, arch=arch or ""))
                arguments = encode_launch(replace(launch, request=child_request,
                    workspace_base=workspace_base, workspace_suffix=workspace_suffix,
                    restart_arguments=()))
            cmd = [sys.executable, str(optimizer_script), *arguments]
            from aka.legacy.application.processes import child_environment as launch_environment
            child_environment = os.environ.copy()
            for name, value in launch_environment().items():
                if value is None:
                    child_environment.pop(name, None)
                else:
                    child_environment[name] = value
            from aka.legacy.application import execution
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
