"""Compatibility import for aka.legacy.workflow.hardware."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.hardware")
sys.modules[__name__] = _implementation
