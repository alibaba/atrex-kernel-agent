"""Compatibility import for aka.legacy.execution.configuration."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.execution.configuration")
sys.modules[__name__] = _implementation
