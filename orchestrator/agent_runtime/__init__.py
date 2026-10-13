"""Compatibility import for the extracted legacy implementation."""
from orchestrator._bootstrap import task_modules
task_modules()
from aka.legacy.agent import __all__
def __getattr__(name):
    from aka.legacy import agent
    return getattr(agent, name)
