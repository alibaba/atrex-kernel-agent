"""Compatibility import for aka.legacy.workflow.constants."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.constants")
sys.modules[__name__] = _implementation
