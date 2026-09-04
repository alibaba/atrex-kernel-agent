#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Launch the store-blind natural-language intent bridge safely.

Only CLI protocols with an explicit no-tools mode are supported. Claude and
Qoder return a JSON envelope on stdout without approval
or sandbox bypasses. Codex is intentionally excluded: its read-only sandbox
still exposes the shell and readable files to prompt injection.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

SUPPORTED = ("claude", "qodercli")
DEFAULT_CLI = "claude"
DEFAULT_TIMEOUT = 600
EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
CLAUDE_SETTINGS_ENV = "ATREX_CLAUDE_SESSION_SETTINGS"

# Do not inherit arbitrary campaign credentials into the parser process. The
# CLI may read its own auth state under HOME; explicit provider auth and common
# TLS/proxy variables are the only environment credentials forwarded.
COMMON_ENVIRONMENT = (
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "ATREX_ENVIRONMENT_STATE_FILE", "ATREX_ENVIRONMENT_RESTART_HANDOFF_ID",
    "ATREX_ENVIRONMENT_RESTART_LOCK_FD",
)
AUTH_ENVIRONMENT = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
               "CLAUDE_CODE_OAUTH_TOKEN"),
    "qodercli": ("QODER_API_KEY", "QODER_AUTH_TOKEN", "QODER_ACCESS_TOKEN"),
}


class LaunchError(RuntimeError):
    """The bridge CLI could not be started."""


def build_claude_json_command(prompt: str, session_id: str | None = None) -> list[str]:
    """Build a non-interactive bridge with no tools or permission bypass."""
    session = session_id or str(uuid.uuid4())
    return [
        "claude", "--bare", "--print", "--output-format", "json",
        "--no-session-persistence", "--session-id", session, "--effort", "low",
        "--tools", "", "--prompt-suggestions", "false", prompt,
    ]


def build_qoder_json_command(prompt: str, session_id: str | None = None) -> list[str]:
    """Build Qoder's non-interactive, no-tools, empty-MCP JSON protocol."""
    session = session_id or str(uuid.uuid4())
    return [
        "qodercli", "--print", "--output-format", "json",
        "--no-session-persistence", "--session-id", session,
        "--reasoning-effort", "low", "--permission-mode", "default",
        "--tools", "", "--strict-mcp-config", "--mcp-config", EMPTY_MCP_CONFIG,
        "--setting-sources", "", "--max-output-tokens", "2048", "--", prompt,
    ]


def build_json_command(cli: str, prompt: str,
                       session_id: str | None = None) -> list[str]:
    if cli == "claude":
        return build_claude_json_command(prompt, session_id)
    if cli == "qodercli":
        return build_qoder_json_command(prompt, session_id)
    if cli == "codex":
        raise LaunchError(
            "unsupported bridge cli 'codex': read-only still permits shell/file reads; "
            "a verified no-tools protocol is required")
    raise LaunchError("unsupported bridge cli %r (supported: %s)" %
                      (cli, ", ".join(SUPPORTED)))


def build_command(cli: str, prompt: str, session_id: str | None = None,
                  settings: str | None = None) -> list[str]:
    """Compatibility launcher; query_nl uses the stricter run_json boundary."""
    if cli == "codex":
        return ["codex", "exec", "--color", "never", prompt]
    command = build_json_command(cli, prompt, session_id)
    if cli == "claude" and settings:
        command[-1:-1] = ["--settings", settings]
    return command


def bridge_environment(cli: str, source: dict[str, str] | None = None) -> dict[str, str]:
    """Return the minimal environment needed by one authenticated bridge CLI."""
    inherited = os.environ if source is None else source
    allowed = COMMON_ENVIRONMENT + AUTH_ENVIRONMENT.get(cli, ())
    environment = {name: inherited[name] for name in allowed if inherited.get(name)}
    environment["NO_COLOR"] = "1"
    environment["CI"] = "1"
    return environment


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _start_bridge_process(
    command: list[str], cwd: Path, environment: dict[str, str]
) -> subprocess.Popen:
    options = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "errors": "replace",
    }
    if not environment.get("ATREX_ENVIRONMENT_RESTART_HANDOFF_ID"):
        return subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
            close_fds=True,
            **options,
        )
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    try:
        from orchestrator.recovery_processes import spawn_owned_session
    except ImportError as exc:
        raise LaunchError(
            "recovery process registration is unavailable during restart handoff"
        ) from exc
    return spawn_owned_session(
        command,
        role="wiki-bridge-agent",
        environment=environment,
        **options,
    )


def _run_command(command: list[str], cwd: Path, timeout: int,
                 env: dict[str, str] | None) -> tuple[str, str, int, bool]:
    cli = Path(command[0]).name if command else ""
    environment = bridge_environment(cli, env)
    try:
        proc = _start_bridge_process(command, cwd, environment)
    except FileNotFoundError as exc:
        missing = command[0] if command else "<empty-command>"
        raise LaunchError(f"bridge cli not on PATH: {missing} ({exc})") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise LaunchError(f"cannot start recovery bridge process: {exc}") from exc

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "killed after timeout; no output collected"
    return stdout or "", stderr or "", proc.returncode, timed_out


def run_claude_json(prompt: str, cwd: Path, timeout: int = DEFAULT_TIMEOUT,
                    env: dict[str, str] | None = None
                    ) -> tuple[str, str, int, bool]:
    return _run_command(build_claude_json_command(prompt), cwd, timeout, env)


def run(cli: str, prompt: str, cwd: Path, timeout: int = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None) -> tuple[str, str, int, bool]:
    """Compatibility entry point without restoring permission-bypass flags."""
    settings = (env or os.environ).get(CLAUDE_SETTINGS_ENV) if cli == "claude" else None
    return _run_command(build_command(cli, prompt, settings=settings), cwd, timeout, env)


def run_json(cli: str, prompt: str, cwd: Path, timeout: int = DEFAULT_TIMEOUT,
             env: dict[str, str] | None = None) -> tuple[str, str, int, bool]:
    if cli == "claude":
        return run_claude_json(prompt, cwd, timeout, env)
    return run(cli, prompt, cwd, timeout, env)


if __name__ == "__main__":
    raise SystemExit("agent_launch.py is an internal library; use tools/query_nl.py")
