"""Main-compatible sandbox execution through the shared execution interface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from aka.legacy.execution.legacy.runtime import ExecutionHooks, SandboxRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hardware", default=os.environ.get("ATREX_SANDBOX_GPU", ""))
    parser.add_argument("--gateway-profile", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--ssh", default="")
    parser.add_argument("--ssh-init", default="")
    parser.add_argument("--health-command", default=None)
    parser.add_argument("--sync", action="append", default=[])
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("ATREX_SANDBOX_TIMEOUT", "600")))
    parser.add_argument("--wall-timeout", type=int)
    parser.add_argument("--kind", choices=("auto", "run", "profile", "dev"), default="auto")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    if not arguments.hardware:
        parser.error("--hardware or ATREX_SANDBOX_GPU is required")
    if arguments.no_sync and arguments.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    root = arguments.repo_root.expanduser().resolve()
    launcher = root / "tools/sandbox.py"
    if not launcher.is_file():
        parser.error("--repo-root must contain the matching tools/sandbox.py")
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
    if not command and not arguments.preflight:
        parser.error("provide a command after -- or use --preflight")
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        from aka.legacy.execution.recovery import raise_if_environment_blocked
        from aka.legacy.execution.processes import spawn_owned_session
        from aka.legacy.execution.configuration import queue_wait_grace
        from aka.legacy.execution.selection import execution_environment, execution_scope, run_sandbox
        hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
        with execution_scope(selection=selection) as executor:
            environment = execution_environment(os.environ)
            gateway_profile, url = arguments.gateway_profile, arguments.url
            from .gateway.executor import GatewayExecutor
            if isinstance(executor, GatewayExecutor):
                if gateway_profile and url:
                    parser.error("--gateway-profile and --url are mutually exclusive")
                if gateway_profile or url:
                    environment["ATREX_SANDBOX_PROFILE"] = gateway_profile
                    environment["ATREX_SANDBOX_URL"] = url
                    gateway_profile = url = ""
            request = SandboxRequest(
                workspace=arguments.workspace.resolve(), launcher=(sys.executable, str(launcher)),
                hardware=arguments.hardware, profile=gateway_profile, url=url,
                timeout=arguments.timeout, command=command, environment=environment,
                ssh=arguments.ssh, ssh_init=arguments.ssh_init,
                health_command=(arguments.health_command if arguments.health_command is not None
                    else os.environ.get("ATREX_SANDBOX_HEALTH_COMMAND", "")),
                sync=tuple(arguments.sync), wall_timeout=arguments.wall_timeout, gateway_kind=arguments.kind,
                preflight=arguments.preflight,
            )
            result = run_sandbox(executor, request, hooks=hooks)
        sys.stdout.write(result.stdout or "")
        sys.stderr.write(result.stderr or "")
        return result.returncode
    finally:
        if added:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(main())
