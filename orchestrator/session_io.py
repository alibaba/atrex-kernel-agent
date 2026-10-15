"""Compatibility import for aka.legacy.workflow.session_io."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.session_io")
sys.modules[__name__] = _implementation
