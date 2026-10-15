"""Compatibility import for aka.legacy.compat.launch."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.launch")
sys.modules[__name__] = _implementation
