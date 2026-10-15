"""Compatibility import for aka.legacy.resources.runtime."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.resources.runtime")
sys.modules[__name__] = _implementation
