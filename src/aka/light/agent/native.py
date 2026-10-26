"""Light session policy over the shared Agent lifecycle and resume mechanics."""
from dataclasses import replace
from aka.agent.native import (
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


def fresh_session_command(prompt, session_id, reasoning_effort, agent_cli="claude", *, runtime=None, environment=None):
    return _fresh_session_command(prompt, session_id, reasoning_effort, agent_cli,
                                  runtime=runtime or build_agent_runtime(agent_cli), environment=environment)


def resume_session_command(prompt, session_id, reasoning_effort, agent_cli="claude", *, runtime=None, environment=None):
    return _resume_session_command(prompt, session_id, reasoning_effort, agent_cli,
                                   runtime=runtime or build_agent_runtime(agent_cli), environment=environment)


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



from aka.agent.session_hooks import SessionHooks


class LightSessionHooks(SessionHooks):
    def prepare_environment(self, workspace, environment, overrides, session_id):
        from aka.light.orchestrator.agent_home import prepare_agent_environment
        return prepare_agent_environment(workspace, super().invocation_environment(environment, overrides), session_id)

    def invocation_environment(self, environment, overrides):
        for key, value in (overrides or {}).items():
            if key in {"HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "QODER_USER_DATA_DIR"} and value != environment.get(key):
                raise ValueError(f"Cannot change {key} inside an open Agent session")
        return super().invocation_environment(environment, overrides)

    def observer(self, environment):
        return CodexSessionLedgerObserver(codex_home(environment))

    def stream_usage(self, runtime_id, stdout):
        return terminal_usage_from_stream(stdout, backend=runtime_id)

    def observe(self, runtime_id, stdout, session_id, observer, backend):
        return normalize_stream(runtime_id, stdout, session_id=session_id,
                                codex_observer=observer, adapter=backend)

    def check_invocation(self, invoked, resume):
        if invoked and not resume:
            raise RuntimeError("Session already invoked; use resume=True or open a new Session")


def NativeSession(*args, **kwargs):
    kwargs["session_hooks"] = LightSessionHooks()
    return LegacyNativeSession(*args, **kwargs)
