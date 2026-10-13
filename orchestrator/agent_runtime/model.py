"""Compatibility import for the extracted legacy implementation."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
sys.modules[__name__] = importlib.import_module("aka.contracts.agent")
