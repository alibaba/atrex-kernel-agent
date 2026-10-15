"""Compatibility import for aka.legacy.workflow.numerical_review."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.workflow.numerical_review")
sys.modules[__name__] = _implementation
