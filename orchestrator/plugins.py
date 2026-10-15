"""Compatibility import for aka.legacy.resources.plugins."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.resources.plugins")
sys.modules[__name__] = _implementation
