from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .adapter import (
    DEFAULT_BACKEND_REGISTRY,
    BackendAdapterRegistry,
    ClaudeAdapter,
    CodexAdapter,
    PiAdapter,
    QoderAdapter,
    token_usage_from_mapping,
    token_usage_from_model_usage,
)
from .codex_ledger import (
    CodexSessionLedgerObserver,
    CodexTemporaryHome,
    codex_home,
    codex_thread_id_from_stream,
    observe_codex_usage,
)
from .model import (
    AgentRunRequest,
    AgentRunResult,
    AgentRuntime,
    TokenUsage,
    sum_token_usages,
)
from .process import ProcessRunner, protected_gateway_identity, run_bounded

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_SHELL_GUARD = REPO_ROOT / "tools" / "session_shell_guard.sh"
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
SUPPORTED_RUNTIME_IDS = DEFAULT_BACKEND_REGISTRY.ids
REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def terminal_usage_from_stream(stdout: str) -> TokenUsage:
    """Parse the existing cross-backend terminal contract without event attribution."""
    terminal = TokenUsage.unavailable()
    deltas: list[TokenUsage] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if event.get("type") in {"result", "turn.completed"}:
            parsed = token_usage_from_mapping(event.get("usage"))
            if parsed.total_tokens is None:
                parsed = token_usage_from_model_usage(event.get("modelUsage"))
            if parsed.total_tokens is not None:
                terminal = parsed
            continue
        usage = event.get("usage")
        message = event.get("message")
        if usage is None and isinstance(message, Mapping):
            usage = message.get("usage")
        parsed = token_usage_from_mapping(usage)
        if parsed.total_tokens is not None:
            deltas.append(parsed)
    if terminal.total_tokens is not None:
        return terminal
    fallback = sum_token_usages(deltas)
    return (
        replace(fallback, measurement="partial")
        if fallback.total_tokens is not None
        else fallback
    )


def token_usage_from_stream(stdout: str) -> int:
    """Preserve the terminal-token compatibility contract for legacy callers."""
    return terminal_usage_from_stream(stdout).total_tokens or 0


def build_session_environment(runtime_id: str) -> dict[str, str]:
    """Build the current guarded environment for one coding-agent session."""
    environment = os.environ.copy()
    # Private Atrex-Bench evaluator inputs are campaign-scoped and must be reintroduced only by
    # the owning campaign, never inherited accidentally by an unrelated or legacy session.
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    environment["ATREX_AGENT_CLI"] = runtime_id
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [
        part
        for part in environment.get("PATH", "").split(os.pathsep)
        if part and part != python_bin
    ]
    environment["PATH"] = os.pathsep.join([python_bin, *path_parts])
    environment["PIP_ONLY_BINARY"] = ":all:"
    environment["PIP_INDEX_URL"] = PYPI_MIRROR
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["BASH_ENV"] = str(SESSION_SHELL_GUARD)
    protected_screen, protected_state = protected_gateway_identity(environment)
    environment["ATREX_PROTECTED_GATEWAY_SCREEN"] = protected_screen
    environment["ATREX_PROTECTED_GATEWAY_STATE_DIR"] = protected_state
    if runtime_id == "pi":
        environment["PI_SKIP_VERSION_CHECK"] = "1"
        environment["PI_TELEMETRY"] = "0"
    if runtime_id == "claude" and environment.get("ANTHROPIC_AUTH_TOKEN"):
        environment.pop("ANTHROPIC_API_KEY", None)
    return environment


class CliAgentRuntime:
    def __init__(
        self,
        adapter,
        *,
        process_runner: ProcessRunner = run_bounded,
    ) -> None:
        self._adapter = adapter
        self._process_runner = process_runner

    @property
    def id(self) -> str:
        return self._adapter.id

    def build_command(
        self, prompt: str, session_id: str, reasoning_effort: str
    ) -> list[str]:
        return self._adapter.build_command(
            prompt,
            session_id,
            reasoning_effort,
            self._session_settings(),
        )

    def _session_settings(self) -> str:
        return os.environ.get(self._adapter.settings_variable) or os.environ.get(
            "ATREX_SESSION_SETTINGS", ""
        )

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        if request.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"unsupported reasoning effort: {request.reasoning_effort!r}"
            )
        session_id = request.session_id or str(uuid.uuid4())
        command = self.build_command(
            request.prompt, session_id, request.reasoning_effort
        )
        environment = build_session_environment(self.id)
        environment["IS_SANDBOX"] = "1"
        if request.extra_environment:
            environment.update(
                {
                    str(key): str(value)
                    for key, value in request.extra_environment.items()
                }
            )
        if request.sandbox_hardware:
            environment["ATREX_SANDBOX_GPU"] = request.sandbox_hardware
        if request.sandbox_ssh:
            environment["ATREX_SANDBOX_SSH"] = request.sandbox_ssh
            environment.pop("ATREX_SANDBOX_URL", None)
            environment.pop("ATREX_SANDBOX_PROFILE", None)
        elif request.sandbox_url:
            environment["ATREX_SANDBOX_URL"] = request.sandbox_url
            environment.pop("ATREX_SANDBOX_SSH", None)
            environment.pop("ATREX_SANDBOX_PROFILE", None)
        elif request.sandbox_profile:
            environment["ATREX_SANDBOX_PROFILE"] = request.sandbox_profile
            environment.pop("ATREX_SANDBOX_SSH", None)
            environment.pop("ATREX_SANDBOX_URL", None)
        if request.sandbox_ssh_init:
            environment["ATREX_SANDBOX_SSH_INIT"] = request.sandbox_ssh_init
        if request.sandbox_health_command:
            environment["ATREX_SANDBOX_HEALTH_COMMAND"] = (
                request.sandbox_health_command
            )
        if request.environment_state_file:
            environment["ATREX_ENVIRONMENT_STATE_FILE"] = (
                request.environment_state_file
            )
        environment["ATREX_SANDBOX_TIMEOUT"] = str(request.sandbox_timeout_s)
        # The active backend is supervisor-owned. Plan helpers use it to avoid recursively
        # launching Codex or Qoder from an episode already owned by the matching backend.
        environment["ATREX_AGENT_CLI"] = self.id
        codex_observer = None
        codex_temporary_home = None
        pre_observation_errors: tuple[str, ...] = ()
        original_codex_home = environment.get("CODEX_HOME")
        isolated_home_ready = False
        if self.id == "codex":
            try:
                codex_temporary_home = CodexTemporaryHome(codex_home(environment))
                isolated_home = codex_temporary_home.open()
                isolated_home_ready = True
                environment["CODEX_HOME"] = str(isolated_home)
                codex_observer = CodexSessionLedgerObserver(isolated_home)
            except Exception as exc:
                pre_observation_errors = (
                    f"codex_ledger_setup_failed:{type(exc).__name__}",
                )
                if not isolated_home_ready:
                    if codex_temporary_home is not None:
                        cleanup_error = codex_temporary_home.close()
                        if cleanup_error:
                            pre_observation_errors += (cleanup_error,)
                    codex_temporary_home = None
                    if original_codex_home is None:
                        environment.pop("CODEX_HOME", None)
                    else:
                        environment["CODEX_HOME"] = original_codex_home
                try:
                    command.insert(command.index("--json") + 1, "--ephemeral")
                except ValueError:
                    pass
        try:
            stdout, stderr, exit_status, timed_out = self._process_runner(
                command,
                cwd=request.workspace,
                timeout=request.timeout_s,
                env=environment,
            )
        except BaseException:
            if codex_temporary_home is not None:
                codex_temporary_home.close()
            raise
        observation_errors: tuple[str, ...] = pre_observation_errors
        try:
            events, terminal_usage = self._adapter.normalize_stream(stdout)
        except Exception as exc:
            # Observation parsing must not turn a completed Agent run into a failure,
            # and the existing terminal token budget must remain available.
            events = ()
            terminal_usage = terminal_usage_from_stream(stdout)
            observation_errors = (f"stream_normalization_failed:{type(exc).__name__}",)
        capabilities = replace(
            self._adapter.capabilities,
            usage_delta_observed=any(event.kind == "usage_delta" for event in events),
        )
        if codex_observer is not None:
            observed_session_id = codex_thread_id_from_stream(stdout)
            try:
                if not observed_session_id:
                    observed_session_id = codex_observer.identify_new_thread(
                        request.workspace
                    )
                session_id = observed_session_id
                (
                    events,
                    terminal_usage,
                    capabilities,
                    ledger_errors,
                ) = observe_codex_usage(
                    codex_observer, observed_session_id, terminal_usage
                )
                observation_errors += ledger_errors
            except Exception as exc:
                observation_errors += (
                    f"codex_ledger_unavailable:{type(exc).__name__}",
                )
        if codex_temporary_home is not None:
            cleanup_error = codex_temporary_home.close()
            if cleanup_error:
                observation_errors += (cleanup_error,)
        return AgentRunResult(
            runtime_id=self.id,
            exit_status=exit_status,
            timed_out=timed_out,
            terminal_usage=terminal_usage,
            events=events,
            capabilities=capabilities,
            observation_errors=observation_errors,
            stdout_tail=stdout[-2000:],
            stderr_tail=stderr[-2000:],
            session_id=session_id,
        )


class ClaudeRuntime(CliAgentRuntime):
    def __init__(
        self,
        *,
        process_runner: ProcessRunner = run_bounded,
        runtime_assets_dir: Path | None = None,
    ) -> None:
        super().__init__(
            ClaudeAdapter(runtime_assets_dir), process_runner=process_runner
        )


class QoderRuntime(CliAgentRuntime):
    def __init__(
        self,
        *,
        process_runner: ProcessRunner = run_bounded,
        runtime_assets_dir: Path | None = None,
    ) -> None:
        super().__init__(
            QoderAdapter(runtime_assets_dir), process_runner=process_runner
        )


class PiRuntime(CliAgentRuntime):
    def __init__(
        self,
        *,
        process_runner: ProcessRunner = run_bounded,
        runtime_assets_dir: Path | None = None,
    ) -> None:
        super().__init__(PiAdapter(runtime_assets_dir), process_runner=process_runner)


class CodexRuntime(CliAgentRuntime):
    def __init__(
        self,
        *,
        process_runner: ProcessRunner = run_bounded,
        runtime_assets_dir: Path | None = None,
    ) -> None:
        super().__init__(
            CodexAdapter(runtime_assets_dir), process_runner=process_runner
        )


def build_agent_runtime(
    runtime_id: str,
    *,
    process_runner: ProcessRunner = run_bounded,
    runtime_assets_dir: Path | None = None,
    registry: BackendAdapterRegistry = DEFAULT_BACKEND_REGISTRY,
) -> AgentRuntime:
    adapter = registry.create(runtime_id, runtime_assets_dir)
    return CliAgentRuntime(adapter, process_runner=process_runner)


def build_session_command(
    runtime_id: str,
    prompt: str,
    session_id: str,
    reasoning_effort: str = "max",
    *,
    runtime_assets_dir: Path | None = None,
) -> list[str]:
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort!r}")
    runtime = build_agent_runtime(runtime_id, runtime_assets_dir=runtime_assets_dir)
    assert isinstance(runtime, CliAgentRuntime)
    return runtime.build_command(prompt, session_id, reasoning_effort)


def auth_hint(runtime_id: str) -> str:
    adapter = DEFAULT_BACKEND_REGISTRY.create(runtime_id)
    return adapter.auth_hint()
