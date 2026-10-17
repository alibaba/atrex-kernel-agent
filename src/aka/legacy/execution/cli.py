"""Main-compatible sandbox execution through the shared execution interface."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from aka.legacy.execution.legacy.runtime import ExecutionHooks, SandboxRequest


def main(argv: list[str] | None = None) -> int:
    from .selection import selected_execution
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = arguments[:arguments.index("--")] if "--" in arguments else arguments
    if selected_execution() == {"profile": "legacy"} and not any(
            value.split("=", 1)[0] in {"--profile", "--profiles-dir", "--patch"} for value in options):
        from .gateway.sandbox import main as transport_main
        return transport_main(arguments)
    # Keep the transport parser authoritative for its input/profiling flags.
    # Module selection must not silently narrow the installed CLI surface.
    from .gateway.sandbox import build_parser
    parser = build_parser()
    parser.set_defaults(health_command=os.environ.get("ATREX_SANDBOX_HEALTH_COMMAND", ""))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--wall-timeout", type=int)
    arguments = parser.parse_args(arguments)
    if not arguments.hardware:
        parser.error("--hardware or ATREX_SANDBOX_GPU is required")
    if arguments.no_sync and arguments.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    from .selection import selected_execution, read_execution_patches
    selection = dict(selected_execution())
    if arguments.profile is not None:
        selection["profile"] = arguments.profile
    if arguments.patch:
        selection["patches"] = [*selection.get("patches", ()), *read_execution_patches(arguments.patch)]
    if arguments.profiles_dir is not None:
        selection["profiles_dir"] = str(arguments.profiles_dir.expanduser().resolve())
    command = tuple(arguments.command)
    if command[:1] == ("--",):
        command = command[1:]
    if not command and not (arguments.preflight or arguments.check_health):
        parser.error("provide a command after -- or use --preflight")
    from aka.legacy.execution.recovery import raise_if_environment_blocked
    from aka.legacy.execution.processes import spawn_owned_session
    from aka.legacy.execution.configuration import queue_wait_grace
    from aka.legacy.execution.selection import execution_environment, execution_scope, run_sandbox
    hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
    with execution_scope(selection=selection) as executor:
        environment = execution_environment(os.environ)
        gateway_profile, url = arguments.gateway_profile or "", arguments.url or ""
        from .gateway.executor import GatewayExecutor
        from .legacy.executor import LegacyExecutor
        from aka.contracts.execution import UnsupportedExecution
        extra = []
        for flag, value in (
            ("--input", arguments.input), ("--env", arguments.env),
            ("--profile-counter", arguments.profile_counter),
            ("--ssh-runtime-bind", arguments.ssh_runtime_bind or []),
        ):
            for item in value:
                extra.extend((flag, str(item)))
        for flag, value in (
            ("--profiler", arguments.profiler), ("--profile-level", arguments.profile_level),
            ("--kernel-regex", arguments.kernel_regex), ("--top-kernels", arguments.top_kernels),
            ("--shape-batch-size", arguments.shape_batch_size),
            ("--max-input-file-mb", arguments.max_input_file_mb),
            ("--max-output-file-mb", arguments.max_output_file_mb),
            ("--ssh-gpu", arguments.ssh_gpu),
            ("--runtime-health-command", arguments.runtime_health_command or None),
        ):
            if value is not None:
                extra.extend((flag, str(value)))
        for flag, enabled in (
            ("--dry-run", arguments.dry_run), ("--inline-output", arguments.inline_output),
            ("--include-raw-profile", arguments.include_raw_profile),
            ("--keep-pod", arguments.keep_pod), ("--check-health", arguments.check_health),
        ):
            if enabled:
                extra.append(flag)
        if not isinstance(executor, (LegacyExecutor, GatewayExecutor)):
            if any(value.split("=", 1)[0] in {
                "--input", "--env", "-e", "--profiler", "--profile-level", "--profile-counter",
                "--kernel-regex", "--top-kernels", "--shape-batch-size", "--max-input-file-mb",
                "--max-output-file-mb", "--dry-run", "--inline-output", "--include-raw-profile",
                "--keep-pod", "--ssh-runtime-bind", "--ssh-gpu", "--runtime-health-command", "--check-health",
            } for value in options):
                raise UnsupportedExecution("selected executor does not support sandbox transport options")
            extra = []
        if isinstance(executor, GatewayExecutor):
            if arguments.ssh_runtime_bind or arguments.ssh_gpu is not None or arguments.runtime_health_command or arguments.check_health:
                raise UnsupportedExecution("gateway profile does not support SSH or health configuration")
            if gateway_profile and url:
                parser.error("--gateway-profile and --url are mutually exclusive")
            if gateway_profile or url:
                environment["ATREX_SANDBOX_PROFILE"] = gateway_profile
                environment["ATREX_SANDBOX_URL"] = url
                gateway_profile = url = ""
        request = SandboxRequest(
            workspace=Path(arguments.workspace).resolve(), launcher=(sys.executable, "-m", "aka.legacy.execution.gateway.sandbox"),
            hardware=arguments.hardware, profile=gateway_profile, url=url,
            timeout=arguments.timeout, command=command, environment=environment,
            ssh=arguments.ssh or "", ssh_init=arguments.ssh_init,
            health_command=(arguments.health_command if arguments.health_command is not None
                else os.environ.get("ATREX_SANDBOX_HEALTH_COMMAND", "")),
            sync=tuple(arguments.sync), wall_timeout=arguments.wall_timeout, gateway_kind=arguments.kind,
            preflight=arguments.preflight,
        )
        from .legacy.runtime import transport_arguments
        with transport_arguments(tuple(extra)):
            result = run_sandbox(executor, request, hooks=hooks)
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return result.returncode



if __name__ == "__main__":
    raise SystemExit(main())
