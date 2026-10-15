"""Compatibility import for aka.legacy.experiment.retry."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.compat.infrastructure_retry")
sys.modules[__name__] = _implementation
