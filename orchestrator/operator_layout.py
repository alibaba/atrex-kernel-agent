"""Compatibility import for aka.legacy.task.problem."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.task.problem")
sys.modules[__name__] = _implementation
