"""Compatibility import for aka.legacy.task.candidate_workspace.state."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.task.candidate_workspace.state")
sys.modules[__name__] = _implementation
