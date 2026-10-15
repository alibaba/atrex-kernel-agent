"""Compatibility import for aka.legacy.agent.process."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.agent.process")
sys.modules[__name__] = _implementation
