"""Compatibility entry for the extracted Workflow implementation."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
_implementation = importlib.import_module("aka.legacy.workflow.main_adapter")
sys.modules[__name__] = _implementation
