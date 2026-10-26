"""Prepare Workflow launch recovery and request the selected execution environment."""
import json
from aka.legacy.execution.configuration import legacy_settings, selected_ssh

def prepare_runtime(launch, problem):
    import os
    from .recovery_runtime import configure_recovery
    from aka.legacy.execution.recovery import raise_if_environment_blocked
    from aka.legacy.execution.health import runtime_health_command
    from aka.legacy.execution.sandbox import check_ssh_environment
    from aka.legacy.execution.selection import selected_execution
    from aka.contracts.workflow_launch import encode_launch
    from types import SimpleNamespace
    settings = legacy_settings() if selected_execution().get("profile", "legacy") == "legacy" else SimpleNamespace(
        profile="", url="", ssh="", ssh_init="", health_command="")
    hardware = launch.request.runtime.hardware or launch.request.settings.platform
    if selected_ssh():
        # Retired checkout-script recovery commands cannot identify an installed run.
        from aka.legacy.execution.recovery import environment_state_file
        state = environment_state_file()
        if state is not None and state.with_name("restart.json").is_file():
            record = json.loads(state.with_name("restart.json").read_text())
            command = record.get("command", [])
            if len(command) < 6 or command[1:6] != ["-m", "aka", "run", "--resume", "--"]:
                raise RuntimeError("retired script recovery command; use the installed profile continuation")
        raw_argv = launch.restart_arguments or encode_launch(launch)
        configure_recovery(workspace_base=launch.workspace_base,
            raw_argv=raw_argv, optimize_script=None,
            sandbox_hardware=hardware, ssh_target=settings.ssh, ssh_init=settings.ssh_init,
            ssh_runtime_binds=json.loads(os.environ.get("ATREX_SANDBOX_SSH_RUNTIME_BINDS", "[]")),
            ssh_gpu=settings.ssh_gpu, health_command=settings.health_command,
            poll_interval=int(os.environ.get("ATREX_ENVIRONMENT_POLL_INTERVAL", "60")),
            runtime_health_command=runtime_health_command(sol=problem.initial_source.kind == "sol",
                framework=launch.request.settings.framework))
        raise_if_environment_blocked()
        check_ssh_environment(launch.workspace_base, hardware, settings.ssh, settings.ssh_init, settings.health_command)
    from aka.contracts.execution_environment import EnvironmentQuery
    from aka.legacy.execution.selection import execution_scope
    from aka.legacy.execution.binding import sandbox_executor
    with execution_scope() as executor:
        bound = sandbox_executor(executor, hardware=hardware, timeout=launch.request.runtime.timeout)
        snapshot = bound.query_environment(EnvironmentQuery(hardware))
    arch = launch.request.settings.arch or str(snapshot.observed.get("arch") or "")
    return arch, settings.ssh or settings.url or settings.profile or "selected execution"
