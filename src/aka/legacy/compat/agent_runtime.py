"""Historical runtime import surface."""

from aka.legacy.agent.runtime import *
from .agent import build_agent_runtime, RuntimeAdapter


def token_usage_from_stream(stdout):
    return terminal_usage_from_stream(stdout).total_tokens or 0


def _historical_runtime(name):
    from aka.legacy.agent import runtime

    implementation = getattr(runtime, name)

    class HistoricalRuntime(RuntimeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(implementation(*args, **kwargs))

    HistoricalRuntime.__name__ = name
    return HistoricalRuntime


ClaudeRuntime = _historical_runtime("ClaudeRuntime")
CodexRuntime = _historical_runtime("CodexRuntime")
QoderRuntime = _historical_runtime("QoderRuntime")
PiRuntime = _historical_runtime("PiRuntime")
