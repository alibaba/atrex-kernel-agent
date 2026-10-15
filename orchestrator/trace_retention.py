"""Compatibility import for aka.legacy.workflow.trace_retention."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.workflow.trace_retention")
sys.modules[__name__] = _implementation
