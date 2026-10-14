"""Main-compatible sandbox execution through the shared execution interface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from aka.legacy.execution.legacy.runtime import ExecutionHooks, SandboxRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("legacy",), default=None)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--gateway-profile", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--ssh", default="")
    parser.add_argument("--ssh-init", default="")
    parser.add_argument("--health-command", default=None)
    parser.add_argument("--sync", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--wall-timeout", type=int)
    parser.add_argument("--kind", choices=("auto", "run", "profile", "dev"), default="auto")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    root = arguments.repo_root.expanduser().resolve()
    launcher = root / "tools/sandbox.py"
    if not launcher.is_file():
        parser.error("--repo-root must contain the matching tools/sandbox.py")
    from .execution import selected_execution, read_execution_patches
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
    for name, module in tuple(sys.modules.items()):
        if name == "orchestrator" or name.startswith("orchestrator."):
            # Historical orchestrator entries can alias installed legacy modules.
            # Those implementations belong to the wheel, while real checkout
            # modules still have to come from the requested repository.
            if getattr(module, "__name__", "").startswith("aka.legacy."):
                continue
            location = getattr(module, "__file__", None)
            if location is not None and not Path(location).resolve().is_relative_to(root):
                parser.error("a different AKA checkout is already loaded")
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        from orchestrator.environment_recovery import raise_if_environment_blocked
        from orchestrator.recovery_processes import spawn_owned_session
        from orchestrator.sandbox_config import queue_wait_grace
        from aka.legacy.application.execution import execution_environment, application_execution, run_sandbox
        hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
        with application_execution(selection=selection) as executor:
            request = SandboxRequest(
                workspace=arguments.workspace.resolve(), launcher=(sys.executable, str(launcher)),
                hardware=arguments.hardware, profile=arguments.gateway_profile, url=arguments.url,
                timeout=arguments.timeout, command=command, environment=execution_environment(os.environ),
                ssh=arguments.ssh, ssh_init=arguments.ssh_init,
                health_command=(arguments.health_command if arguments.health_command is not None
                    else os.environ.get("ATREX_SANDBOX_HEALTH_COMMAND", "")),
                sync=tuple(arguments.sync), wall_timeout=arguments.wall_timeout, gateway_kind=arguments.kind,
                preflight=arguments.preflight,
                health_command_explicit=(arguments.health_command is not None
                    or "ATREX_SANDBOX_HEALTH_COMMAND" in os.environ),
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
