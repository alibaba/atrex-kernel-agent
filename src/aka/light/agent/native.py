"""Light session policy over the shared Agent lifecycle and resume mechanics."""
from dataclasses import replace
from aka.legacy.agent.native import (
    NativeSession as LegacyNativeSession,
    fresh_session_command as _fresh_session_command,
    resume_session_command as _resume_session_command,
    supports_same_session_resume, session_id_from_stream,
)
from aka.contracts.agent import AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage
from aka.light.orchestrator.agent_runtime.adapter import DEFAULT_BACKEND_REGISTRY
from aka.light.orchestrator.agent_runtime.runtime import (
    build_agent_runtime, build_session_environment, terminal_usage_from_stream,
)
from aka.light.orchestrator.agent_runtime.codex_ledger import (
    CodexSessionLedgerObserver, codex_home, observe_codex_usage,
)


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
    from aka.light.orchestrator.session_capture import captured_observation

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
    _build_runtime = staticmethod(build_agent_runtime)

    def _prepare_environment(self, environment, overrides, session_id):
        from aka.light.orchestrator.agent_home import prepare_agent_environment
        for key, value in (overrides or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[str(key)] = str(value)
        return prepare_agent_environment(self.workspace, environment, session_id)

    def _invocation_environment(self, overrides):
        environment = dict(self.environment)
        for key, value in (overrides or {}).items():
            # Capture and the ledger remain bound to the Home chosen at creation.
            if key in {"HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "QODER_USER_DATA_DIR"}:
                if value != self.environment.get(key):
                    raise ValueError(f"Cannot change {key} inside an open Agent session")
            if value is None:
                environment.pop(key, None)
            else:
                environment[str(key)] = str(value)
        return environment

    @staticmethod
    def _new_codex_observer(environment):
        return CodexSessionLedgerObserver(codex_home(environment))

    def _stream_usage(self, stdout):
        return terminal_usage_from_stream(stdout, backend=self.runtime_id)

    def _observe(self, stdout):
        return normalize_stream(
            self.runtime_id, stdout, session_id=self._session_id,
            codex_observer=self._observer if self._ledger_usable else None,
            adapter=self._runtime._adapter,
        )

    def _check_invocation(self, resume):
        if not resume and self._invoked:
            raise RuntimeError("Session already invoked; use resume=True or open a new Session")
