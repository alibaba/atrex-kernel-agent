"""Light session policy over the shared Agent lifecycle and resume mechanics."""
from dataclasses import replace
from aka.legacy.agent.native import (
    NativeSession as LegacyNativeSession,
    fresh_session_command as _fresh_session_command,
    resume_session_command as _resume_session_command,
    supports_same_session_resume, session_id_from_stream,
)
from aka.contracts.agent import AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage
from aka.light.agent.home import HOST_HOME_ENV, PREPARED_ENV, prepare_agent_environment
from aka.light.agent.runtime.adapter import DEFAULT_BACKEND_REGISTRY
from aka.light.agent.runtime.runtime import (
    build_agent_runtime,
    build_session_environment,
    terminal_usage_from_stream,
)
from aka.light.agent.runtime.codex_ledger import CodexSessionLedgerObserver, observe_codex_usage


def fresh_session_command(prompt, session_id, reasoning_effort, agent_cli="claude", *, runtime=None):
    return _fresh_session_command(prompt, session_id, reasoning_effort, agent_cli,
                                  runtime=runtime or build_agent_runtime(agent_cli))


def resume_session_command(prompt, session_id, reasoning_effort, agent_cli="claude", *, runtime=None):
    return _resume_session_command(prompt, session_id, reasoning_effort, agent_cli,
                                   runtime=runtime or build_agent_runtime(agent_cli))


def session_environment(agent_cli="claude"):
    return build_session_environment(agent_cli)


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
    adapter = adapter or DEFAULT_BACKEND_REGISTRY.create(agent_cli)
    observation_errors: tuple[str, ...] = ()
    try:
        events, terminal_usage = adapter.normalize_stream(stdout)
    except Exception as exc:
        events = ()
        terminal_usage = terminal_usage_from_stream(stdout, backend=agent_cli)
        observation_errors = (f"stream_normalization_failed:{type(exc).__name__}",)
    capabilities = replace(
        adapter.capabilities,
        usage_delta_observed=any(event.kind == "usage_delta" for event in events),
    )
    from aka.light.agent.session_capture import captured_observation

    captured = captured_observation(stdout, events, capabilities)
    if agent_cli == "codex" and codex_observer is not None and session_id:
        try:
            (
                events,
                terminal_usage,
                capabilities,
                ledger_errors,
            ) = observe_codex_usage(codex_observer, session_id, terminal_usage, captured=captured)
            observation_errors += ledger_errors
        except Exception as exc:
            codex_observer.invalidate()
            if captured is not None:
                events, terminal_usage, capabilities, capture_errors = captured[:4]
                observation_errors += capture_errors
            observation_errors += (f"codex_ledger_unavailable:{type(exc).__name__}",)
    elif captured is not None:
        return captured[:4]
    return events, terminal_usage, capabilities, observation_errors


class NativeSession(LegacyNativeSession):
    _HOME_BINDING_KEYS = (
        "HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "QODER_USER_DATA_DIR",
        "PI_CODING_AGENT_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
        "XDG_DATA_HOME", "XDG_STATE_HOME", HOST_HOME_ENV, PREPARED_ENV,
        "ATREX_AGENT_SANDBOX",
    )

    def __init__(
        self, runtime_id, workspace, session_id, executor,
        environment_overrides=None, check_open=None, *, runtime=None,
        output_directory=None,
    ):
        # The shared constructor has no environment-preparation hook. Prepare
        # explicitly before it binds the ledger observer, and retain this exact
        # Home for process launch, continuation and close/checkpoint capture.
        environment = session_environment(runtime_id)
        environment["IS_SANDBOX"] = "1"
        environment.update({
            str(key): str(value) for key, value in (environment_overrides or {}).items()
        })
        environment = prepare_agent_environment(workspace, environment, session_id)
        super().__init__(
            runtime_id, workspace, session_id, executor, environment, check_open,
            runtime=runtime if runtime is not None else build_agent_runtime(runtime_id),
            output_directory=output_directory,
        )
        self._home_binding = {
            key: self.environment.get(key) for key in self._HOME_BINDING_KEYS
        }

    def _invocation_environment(self, overrides):
        environment = dict(self.environment)
        for key, value in (overrides or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[str(key)] = str(value)
        for key, value in self._home_binding.items():
            if environment.get(key) != value:
                raise ValueError(f"Cannot change {key} inside an open Agent session")
        return environment

    def invoke(self, prompt, *, resume=False, reasoning_effort="max", environment=None):
        if self._check_open is not None:
            self._check_open()
        if self._closed:
            raise RuntimeError("native session has been closed")
        if not resume and self._invoked:
            raise RuntimeError("Session already invoked; use resume=True or open a new Session")
        self.environment = self._invocation_environment(environment)
        return super().invoke(prompt, resume=resume, reasoning_effort=reasoning_effort)
