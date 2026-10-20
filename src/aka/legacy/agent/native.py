"""Native session invocation; no handoff, retry budget or completion decisions."""

from dataclasses import replace
from pathlib import Path
from collections.abc import Mapping
from .adapter import DEFAULT_BACKEND_REGISTRY
from .runtime import (
    REASONING_EFFORTS,
    build_session_command,
    build_session_environment,
    terminal_usage_from_stream,
)
from .codex_ledger import (
    CodexSessionLedgerObserver,
    codex_home,
    codex_thread_id_from_stream,
    observe_codex_usage,
)
from aka.contracts.agent import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
    resequence_agent_events,
    subtract_token_usage,
    token_usage_exceeds,
    NativeInvocation,
)


def fresh_session_command(
    prompt: str,
    session_id: str,
    reasoning_effort: str,
    agent_cli: str = "claude",
    *,
    runtime=None,
) -> list[str]:
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort!r}")
    command = (
        runtime.build_command(prompt, session_id, reasoning_effort)
        if runtime is not None
        else build_session_command(agent_cli, prompt, session_id, reasoning_effort)
    )
    if agent_cli == "codex":
        if command[:2] != ["codex", "exec"] or command[-1] != prompt:
            raise RuntimeError("current main Codex command has no compatible exec seam")
        # Long Horizon recovery must keep the native Codex thread persistent.
        # The current adapter already omits this flag; retain the compatibility
        # guard in case the base command policy changes independently.
        try:
            command.remove("--ephemeral")
        except ValueError:
            pass
    return command


def resume_session_command(
    prompt: str,
    session_id: str,
    reasoning_effort: str,
    agent_cli: str = "claude",
    *,
    runtime=None,
) -> list[str]:
    if agent_cli == "codex":
        command = fresh_session_command(
            prompt, session_id, reasoning_effort, agent_cli, runtime=runtime
        )
        if command[:2] != ["codex", "exec"] or command[-1] != prompt:
            raise RuntimeError("current main Codex command has no compatible exec seam")
        command = command[:-1]
        try:
            color_index = command.index("--color")
        except ValueError:
            pass
        else:
            del command[color_index : color_index + 2]
        command.insert(2, "resume")
        command.extend([session_id, prompt])
        return command
    if agent_cli != "claude":
        raise RuntimeError(
            f"same-session handoff recovery is not supported by current main for {agent_cli}"
        )
    command = fresh_session_command(
        prompt, session_id, reasoning_effort, agent_cli, runtime=runtime
    )
    try:
        index = command.index("--session-id")
    except ValueError as exc:
        raise RuntimeError(
            "current main Claude command has no --session-id compatibility seam"
        ) from exc
    command[index : index + 2] = ["--resume", session_id]
    return command


def session_environment(agent_cli: str = "claude") -> dict[str, str]:
    return build_session_environment(agent_cli)


def supports_same_session_resume(agent_cli: str) -> bool:
    return agent_cli in {"claude", "codex"}


def session_id_from_stream(
    agent_cli: str, stdout: str, requested_session_id: str
) -> str:
    """Return the persistent CLI session id observed in one stream.

    Claude accepts the supervisor-provided id. Codex creates its own thread id
    and reports it in the initial ``thread.started`` JSONL event, so recovery
    must use that observed id rather than the supervisor's bookkeeping UUID.
    """
    if agent_cli != "codex":
        return requested_session_id
    return codex_thread_id_from_stream(stdout)


def normalize_stream(
    agent_cli: str,
    stdout: str,
    *,
    session_id: str = "",
    codex_observer: CodexSessionLedgerObserver | None = None,
    adapter=None,
) -> tuple[
    tuple[NormalizedAgentEvent, ...],
    TokenUsage,
    AgentRuntimeCapabilities,
    tuple[str, ...],
]:
    """Observe one long-session invocation through main's backend adapter."""
    adapter = (
        adapter if adapter is not None else DEFAULT_BACKEND_REGISTRY.create(agent_cli)
    )
    observation_errors: tuple[str, ...] = ()
    try:
        events, terminal_usage = adapter.normalize_stream(stdout)
    except Exception as exc:
        events = ()
        terminal_usage = terminal_usage_from_stream(stdout)
        observation_errors = (f"stream_normalization_failed:{type(exc).__name__}",)
    capabilities = replace(
        adapter.capabilities,
        usage_delta_observed=any(event.kind == "usage_delta" for event in events),
    )
    if agent_cli == "codex" and codex_observer is not None and session_id:
        try:
            (
                events,
                terminal_usage,
                capabilities,
                ledger_errors,
            ) = observe_codex_usage(codex_observer, session_id, terminal_usage)
            observation_errors += ledger_errors
        except Exception as exc:
            observation_errors += (f"codex_ledger_unavailable:{type(exc).__name__}",)
    return events, terminal_usage, capabilities, observation_errors


def _codex_invocation_usage(
    session_usage: TokenUsage, previous_session_usage: TokenUsage | None
) -> TokenUsage:
    if session_usage.total_tokens is None:
        return TokenUsage.unavailable()
    previous = previous_session_usage or TokenUsage.zero()
    if token_usage_exceeds(previous, session_usage):
        return TokenUsage.unavailable()
    return subtract_token_usage(session_usage, previous)


def _replace_terminal_usage(
    events: tuple[NormalizedAgentEvent, ...], usage: TokenUsage
) -> tuple[NormalizedAgentEvent, ...]:
    retained = [event for event in events if event.kind != "terminal_usage"]
    if usage.total_tokens is not None:
        retained.append(
            NormalizedAgentEvent(
                sequence=len(retained), kind="terminal_usage", usage=usage
            )
        )
    return resequence_agent_events(retained)


class NativeSession:
    def __init__(
        self,
        runtime_id,
        workspace,
        session_id,
        executor,
        environment_overrides=None,
        check_open=None,
        *,
        runtime=None,
    ):
        self._runtime = (
            runtime if runtime is not None else self._build_runtime(runtime_id)
        )
        is_codex = runtime_id == "codex"
        self.workspace = workspace
        environment = (
            session_environment()
            if runtime_id == "claude"
            else session_environment(runtime_id)
        )
        environment["IS_SANDBOX"] = "1"
        environment = self._prepare_environment(environment, environment_overrides, session_id)
        codex_observer = None
        codex_setup_errors: tuple[str, ...] = ()
        if is_codex:
            try:
                codex_observer = self._new_codex_observer(environment)
            except Exception as exc:
                codex_setup_errors = (
                    f"codex_ledger_setup_failed:{type(exc).__name__}",
                )
        codex_stdout_session_usage: TokenUsage | None = None
        codex_ledger_usable = codex_observer is not None
        self.runtime_id = runtime_id
        self.environment = environment
        self._observer = codex_observer
        self._setup_errors = codex_setup_errors
        self._ledger_usable = codex_ledger_usable
        self._stdout_usage = codex_stdout_session_usage
        self._session_id = "" if is_codex else session_id
        self._requested_session_id = session_id
        self._executor = executor
        self._closed = False
        self._invoked = False
        self._check_open = check_open

    # Execution/accounting policy hooks are used by the Light Agent. The session
    # lifecycle and resume cursor bookkeeping below have one implementation.
    @staticmethod
    def _build_runtime(runtime_id):
        from .runtime import build_agent_runtime
        return build_agent_runtime(runtime_id)

    def _prepare_environment(self, environment, overrides, session_id):
        if overrides:
            environment.update({str(key): str(value) for key, value in overrides.items()})
        return environment

    def _invocation_environment(self, overrides):
        if overrides:
            self.environment.update(overrides)
        return self.environment

    @staticmethod
    def _new_codex_observer(environment):
        return CodexSessionLedgerObserver(codex_home(environment))

    def _stream_usage(self, stdout):
        return terminal_usage_from_stream(stdout)

    def _observe(self, stdout):
        return normalize_stream(
            self.runtime_id, stdout, session_id=self._session_id,
            codex_observer=self._observer if self._ledger_usable else None,
            adapter=self._runtime._adapter,
        )

    def _check_invocation(self, resume):
        """Legacy permits repeated fresh invocations; providers may restrict it."""

    @property
    def supports_resume(self):
        return supports_same_session_resume(self.runtime_id)

    @property
    def session_id(self):
        return self._session_id

    def close(self):
        self._closed = True
        self._observer = None

    def invoke(self, prompt, *, resume=False, reasoning_effort="max", environment=None):
        if self._check_open is not None:
            self._check_open()
        if self._closed:
            raise RuntimeError("native session has been closed")
        self._check_invocation(resume)
        if resume:
            if not self.supports_resume or not self._session_id:
                raise RuntimeError(
                    "the existing backend has no resumable native session"
                )
            command = resume_session_command(
                prompt,
                self._session_id,
                reasoning_effort,
                self.runtime_id,
                runtime=self._runtime,
            )
        else:
            command = fresh_session_command(
                prompt,
                self._requested_session_id,
                reasoning_effort,
                self.runtime_id,
                runtime=self._runtime,
            )
        invocation_environment = self._invocation_environment(environment)
        stdout, stderr, exit_status, timed_out = self._executor(
            command, self.workspace, None, invocation_environment
        )
        is_codex = self.runtime_id == "codex"
        observed_session_id = session_id_from_stream(
            self.runtime_id, stdout, self._requested_session_id
        )
        if observed_session_id:
            self._session_id = observed_session_id
        stream_session_usage = self._stream_usage(stdout) if is_codex else None
        events, terminal_usage, capabilities, observation_errors = self._observe(stdout)
        if not self._invoked and self._setup_errors:
            observation_errors = self._setup_errors + observation_errors
        ledger_failed = any(
            value.startswith("codex_ledger_unavailable:")
            for value in observation_errors
        )
        if ledger_failed:
            self._ledger_usable = False
        ledger_usage_observed = bool(
            is_codex and capabilities.usage_delta_observed and not ledger_failed
        )
        resume_usage_qualified = bool(
            ledger_usage_observed
            and not any(value.startswith("codex_") for value in observation_errors)
        )
        if is_codex and not ledger_usage_observed:
            fallback_usage = _codex_invocation_usage(
                stream_session_usage or TokenUsage.unavailable(),
                self._stdout_usage,
            )
            if fallback_usage.total_tokens is not None:
                terminal_usage = fallback_usage
                events = _replace_terminal_usage(events, fallback_usage)
            else:
                terminal_usage = TokenUsage.unavailable()
                events = _replace_terminal_usage(events, terminal_usage)
                observation_errors += ("codex_cumulative_fallback_unavailable",)
        if (
            stream_session_usage is not None
            and stream_session_usage.total_tokens is not None
            and (
                self._stdout_usage is None
                or not token_usage_exceeds(self._stdout_usage, stream_session_usage)
            )
        ):
            self._stdout_usage = stream_session_usage
        self._invoked = True
        return NativeInvocation(
            exit_status,
            timed_out,
            self._session_id,
            stdout,
            stderr,
            terminal_usage,
            events,
            capabilities,
            observation_errors,
            resume_usage_qualified,
        )
