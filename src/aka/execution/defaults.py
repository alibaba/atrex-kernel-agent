"""Execution defaults implementation and callable interface."""
from __future__ import annotations


from aka.contracts.evaluation import derived_evaluation_timeout


DEFAULT_SANDBOX_TIMEOUT = derived_evaluation_timeout()
