"""Shared Agent exports are lazy so composition loads only selected capabilities."""

from importlib import import_module

_EXPORTS = {
    "DEFAULT_BACKEND_REGISTRY": (".adapter", "DEFAULT_BACKEND_REGISTRY"),
    "AgentBackendAdapter": (".adapter", "AgentBackendAdapter"),
    "BackendAdapterRegistry": (".adapter", "BackendAdapterRegistry"),
    "ClaudeAdapter": (".adapter", "ClaudeAdapter"),
    "CodexAdapter": (".adapter", "CodexAdapter"),
    "DshAdapter": (".adapter", "DshAdapter"),
    "PiAdapter": (".adapter", "PiAdapter"),
    "QoderAdapter": (".adapter", "QoderAdapter"),
    "codex_settings_args": (".adapter", "codex_settings_args"),
    "pi_settings_args": (".adapter", "pi_settings_args"),
    "token_usage_from_mapping": (".adapter", "token_usage_from_mapping"),
    "toml_config_value": (".adapter", "toml_config_value"),
    "AgentRunRequest": (".model", "AgentRunRequest"),
    "AgentRunResult": (".model", "AgentRunResult"),
    "AgentRuntime": (".model", "AgentRuntime"),
    "AgentRuntimeCapabilities": (".model", "AgentRuntimeCapabilities"),
    "NormalizedAgentEvent": (".model", "NormalizedAgentEvent"),
    "TokenUsage": (".model", "TokenUsage"),
    "subtract_token_usage": (".model", "subtract_token_usage"),
    "sum_token_usages": (".model", "sum_token_usages"),
    "token_usage_exceeds": (".model", "token_usage_exceeds"),
    "ProcessRunner": (".process", "ProcessRunner"),
    "descendant_process_commands": (".process", "descendant_process_commands"),
    "Agent": (".provider", "Agent"),
    "AgentOptions": (".provider", "AgentOptions"),
    "AgentCapabilities": (".model", "AgentCapabilities"),
    "AgentObservation": (".model", "AgentObservation"),
    "AgentAccess": ("aka.contracts.agent", "AgentAccess"),
    "AgentMount": ("aka.contracts.agent", "AgentMount"),
    "run_bounded": ("aka.agent.process", "run_bounded"),
    "PYPI_MIRROR": (".runtime", "PYPI_MIRROR"),
    "SESSION_SHELL_GUARD": (".runtime", "SESSION_SHELL_GUARD"),
    "SUPPORTED_RUNTIME_IDS": (".runtime", "SUPPORTED_RUNTIME_IDS"),
    "ClaudeRuntime": (".runtime", "ClaudeRuntime"),
    "CodexRuntime": (".runtime", "CodexRuntime"),
    "PiRuntime": (".runtime", "PiRuntime"),
    "QoderRuntime": (".runtime", "QoderRuntime"),
    "auth_hint": (".runtime", "auth_hint"),
    "build_agent_runtime": (".runtime", "build_agent_runtime"),
    "build_session_command": (".runtime", "build_session_command"),
    "build_session_environment": (".runtime", "build_session_environment"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
