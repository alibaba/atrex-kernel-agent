"""Legacy evaluator/sandbox configuration is confined to the application adapter."""

from __future__ import annotations
from dataclasses import dataclass, fields
from contextvars import ContextVar
from contextlib import contextmanager


from aka.contracts.workflow import CampaignRuntimeSettings as EvaluationSettings


_configuration = ContextVar("legacy_evaluation_configuration", default=None)


@contextmanager
def evaluation_settings(settings):
    token = _configuration.set(settings)
    try:
        yield
    finally:
        _configuration.reset(token)


def current_settings():
    return _configuration.get() or EvaluationSettings()
