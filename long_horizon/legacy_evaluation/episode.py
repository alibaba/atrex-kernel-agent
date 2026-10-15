"""Compatibility import for aka.legacy.compat.legacy_evaluation.episode."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
sys.modules[__name__] = importlib.import_module("aka.legacy.compat.legacy_evaluation.episode")
