"""Legacy Agent exports are lazy so composition does not load the checkout."""

from importlib import import_module

_EXPORTS = {
    "DEFAULT_BACKEND_REGISTRY": (".adapter", "DEFAULT_BACKEND_REGISTRY"),
    "AgentBackendAdapter": (".adapter", "AgentBackendAdapter"),
    "BackendAdapterRegistry": (".adapter", "BackendAdapterRegistry"),
    "ClaudeAdapter": (".adapter", "ClaudeAdapter"),
    "CodexAdapter": (".adapter", "CodexAdapter"),
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
    "DEFAULT_PROTECTED_GATEWAY_SCREEN": (
        "orchestrator.agent_runtime.process",
        "DEFAULT_PROTECTED_GATEWAY_SCREEN",
    ),
    "DEFAULT_PROTECTED_GATEWAY_STATE_NAME": (
        "orchestrator.agent_runtime.process",
        "DEFAULT_PROTECTED_GATEWAY_STATE_NAME",
    ),
    "ProcessRunner": ("orchestrator.agent_runtime.process", "ProcessRunner"),
    "descendant_process_commands": (
        "orchestrator.agent_runtime.process",
        "descendant_process_commands",
    ),
    "dependency_process_violation": (
        "orchestrator.agent_runtime.process",
        "dependency_process_violation",
    ),
    "protected_gateway_identity": (
        "orchestrator.agent_runtime.process",
        "protected_gateway_identity",
    ),
    "python_import_roots": (
        "orchestrator.agent_runtime.process",
        "python_import_roots",
    ),
    "run_bounded": ("orchestrator.agent_runtime.process", "run_bounded"),
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
    "token_usage_from_stream": (".runtime", "token_usage_from_stream"),
}
__all__ = [
    "AgentBackendAdapter",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRuntime",
    "AgentRuntimeCapabilities",
    "BackendAdapterRegistry",
    "ClaudeAdapter",
    "ClaudeRuntime",
    "CodexAdapter",
    "CodexRuntime",
    "DEFAULT_BACKEND_REGISTRY",
    "DEFAULT_PROTECTED_GATEWAY_SCREEN",
    "DEFAULT_PROTECTED_GATEWAY_STATE_NAME",
    "NormalizedAgentEvent",
    "PYPI_MIRROR",
    "PiAdapter",
    "PiRuntime",
    "ProcessRunner",
    "QoderAdapter",
    "QoderRuntime",
    "SESSION_SHELL_GUARD",
    "SUPPORTED_RUNTIME_IDS",
    "TokenUsage",
    "auth_hint",
    "build_agent_runtime",
    "build_session_command",
    "build_session_environment",
    "codex_settings_args",
    "descendant_process_commands",
    "dependency_process_violation",
    "pi_settings_args",
    "protected_gateway_identity",
    "python_import_roots",
    "run_bounded",
    "subtract_token_usage",
    "sum_token_usages",
    "token_usage_exceeds",
    "token_usage_from_mapping",
    "token_usage_from_stream",
    "toml_config_value",
]


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
