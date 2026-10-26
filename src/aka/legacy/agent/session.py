"""Compatibility module identity for Workflow-owned Agent session orchestration."""
import sys
from aka.legacy.workflow import agent_session as _implementation
sys.modules[__name__] = _implementation
