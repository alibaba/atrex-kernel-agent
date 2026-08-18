#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run one agent CLI headlessly, once, and wait for it.

This store's tools are standard-library only so that the wiki can ship and be
queried on its own. The natural-language front door needs to spawn a small agent,
which means it needs the headless invocation of each supported CLI -- so that
knowledge lives here, in a form deliberately narrower than the orchestrator's:

  * one prompt, one process, and no session resume. The Claude bridge uses a
    tool-free plain-JSON response; legacy backends retain the file handoff;
  * the flags below are a strict SUBSET of what
    ``orchestrator/agent_runtime/adapter.py`` passes for the same CLI, and
    ``tools/test_agent_launch_parity.py`` in the repo root asserts that. If the
    orchestrator changes a flag name, that test fails rather than this launcher
    silently drifting into an invocation nobody runs;
  * a timeout kills the whole process group, not just the child. An agent CLI
    spawns tool subprocesses, and killing only the parent leaves them holding the
    terminal and the GPU.

The prompt is always the LAST positional argument, which is what every supported
CLI expects; it is never piped, because stdin is closed to keep a headless agent
from blocking on a prompt nobody will answer.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

# Headless one-shot invocation per CLI. Subset of the orchestrator's adapters.
# {session} is substituted with a fresh id; CLIs that ignore sessions omit it.
HEADLESS_FLAGS: dict[str, list[str]] = {
    "qodercli": ["--print", "--dangerously-skip-permissions",
                 "--no-session-persistence", "--session-id", "{session}"],
    "claude": ["--print", "--dangerously-skip-permissions",
               "--session-id", "{session}"],
    "codex": ["exec", "--color", "never",
              "--dangerously-bypass-approvals-and-sandbox"],
}

SUPPORTED = tuple(sorted(HEADLESS_FLAGS))
DEFAULT_CLI = "claude"
DEFAULT_TIMEOUT = 600
CLAUDE_SETTINGS_ENV = "ATREX_CLAUDE_SESSION_SETTINGS"


class LaunchError(RuntimeError):
    """The CLI could not be started at all -- distinct from it failing a task."""


def build_command(cli: str, prompt: str, session_id: str | None = None,
                  settings: str | None = None) -> list[str]:
    if cli not in HEADLESS_FLAGS:
        raise LaunchError("unsupported agent cli %r (supported: %s)"
                          % (cli, ", ".join(SUPPORTED)))
    session = session_id or str(uuid.uuid4())
    flags = [f.replace("{session}", session) for f in HEADLESS_FLAGS[cli]]
    command = [cli] + flags
    if cli == "claude" and settings:
        command += ["--settings", settings]
    return command + [prompt]


def build_claude_json_command(
    prompt: str,
    session_id: str | None = None,
) -> list[str]:
    """Build the minimal, tool-free plain-JSON Claude bridge invocation."""
    session = session_id or str(uuid.uuid4())
    return [
        "claude",
        "--bare",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--no-session-persistence",
        "--session-id", session,
        "--effort", "low",
        "--tools", "",
        "--prompt-suggestions", "false",
        prompt,
    ]


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group; an agent CLI has children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _run_command(command: list[str], cwd: Path, timeout: int,
                 env: dict[str, str] | None) -> tuple[str, str, int, bool]:
    environment = dict(os.environ if env is None else env)
    # A nested agent must not inherit a parent's plan-mode or session state.
    for leaked in ("CLAUDE_SESSION_ID", "QODER_SESSION_ID", "CODEX_SESSION_ID",
                   "CLAUDECODE"):
        environment.pop(leaked, None)
    try:
        proc = subprocess.Popen(
            command, cwd=str(cwd), env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace",
            start_new_session=True,          # own process group, so we can kill it all
        )
    except FileNotFoundError as exc:
        raise LaunchError("agent cli not on PATH: %s (%s)" % (cli, exc)) from exc

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:            # unreachable in practice
            stdout, stderr = "", "killed after timeout; no output collected"
    return stdout or "", stderr or "", proc.returncode, timed_out


def run(cli: str, prompt: str, cwd: Path,
        timeout: int = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None) -> tuple[str, str, int, bool]:
    """Run the CLI once in ``cwd``. Returns (stdout, stderr, returncode, timed_out)."""
    environment = dict(os.environ if env is None else env)
    settings = environment.get(CLAUDE_SETTINGS_ENV) if cli == "claude" else None
    return _run_command(
        build_command(cli, prompt, settings=settings), cwd, timeout, environment
    )


def run_claude_json(prompt: str, cwd: Path,
                    timeout: int = DEFAULT_TIMEOUT,
                    env: dict[str, str] | None = None
                    ) -> tuple[str, str, int, bool]:
    """Run the minimal Claude bridge with plain JSON stdout and no tools."""
    return _run_command(
        build_claude_json_command(prompt), cwd, timeout, env
    )


def main(argv=None) -> int:
    """Smoke entry point: run a prompt and echo what came back."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--agent-cli", choices=SUPPORTED, default=DEFAULT_CLI)
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--print-command", action="store_true",
                    help="Show the argv that would run, and exit.")
    args = ap.parse_args(argv)
    if args.print_command:
        settings = os.environ.get(CLAUDE_SETTINGS_ENV) if args.agent_cli == "claude" else None
        print(" ".join(build_command(args.agent_cli, args.prompt, settings=settings)))
        return 0
    out, err, code, timed_out = run(args.agent_cli, args.prompt,
                                    Path(args.cwd).resolve(), args.timeout)
    sys.stderr.write(err)
    sys.stdout.write(out)
    if timed_out:
        print("TIMEOUT after %ds" % args.timeout, file=sys.stderr)
        return 124
    return code or 0


if __name__ == "__main__":
    raise SystemExit(main())
