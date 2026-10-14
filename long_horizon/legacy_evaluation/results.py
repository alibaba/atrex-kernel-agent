"""Historical result helpers routed to their Evaluation/Acceptance owners."""

from orchestrator._bootstrap import task_modules

task_modules()
from aka.legacy.evaluation.results import *
from aka.legacy.evaluation import results as _results
from aka.legacy.acceptance import LegacyAcceptancePolicy

_conversion_parity_passes = LegacyAcceptancePolicy().conversion_parity_passes


def __getattr__(name):
    return getattr(_results, name)
