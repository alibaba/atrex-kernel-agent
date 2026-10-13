"""Compatibility import for the extracted legacy implementation."""

from orchestrator._bootstrap import task_modules

task_modules()
from aka.legacy.agent import __all__ as _exports

__all__ = [*_exports, "token_usage_from_stream"]


def __getattr__(name):
    from aka.legacy import agent

    if name in {"ClaudeRuntime", "CodexRuntime", "QoderRuntime", "PiRuntime"}:
        from aka.legacy.compat import agent_runtime

        return getattr(agent_runtime, name)
    if name == "token_usage_from_stream":
        from aka.legacy.compat.agent_runtime import token_usage_from_stream

        return token_usage_from_stream
    if name in {"AgentRunRequest", "AgentRunResult", "build_agent_runtime"}:
        from aka.legacy.compat import agent as compatibility

        return getattr(compatibility, name)
    return getattr(agent, name)
