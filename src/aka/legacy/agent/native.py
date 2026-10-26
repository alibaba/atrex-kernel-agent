"""Compatibility alias; implementation belongs to aka.agent."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("aka.agent.native")
