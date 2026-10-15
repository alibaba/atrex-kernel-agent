"""Compatibility import for aka.legacy.acceptance.source_policy."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.optimization_policy")
sys.modules[__name__] = _implementation
