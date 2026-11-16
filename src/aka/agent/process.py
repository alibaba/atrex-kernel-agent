"""Compatibility import for the Agent-local process implementation."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("aka.agent.execution.process")
