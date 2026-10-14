"""Legacy evaluator/sandbox configuration is confined to the application adapter."""

from __future__ import annotations
from dataclasses import dataclass, fields
from contextvars import ContextVar
from contextlib import contextmanager


@dataclass(frozen=True)
class EvaluationSettings:
    hardware: str = ""
    profile: str = ""
    url: str = ""
    timeout: int = 600
    verify_repeats: int = 2
    verify_run_timeout: int = 120
    min_improvement_pct: float = 0.0
    ssh: str = ""
    ssh_init: str = ""
    ssh_gpu: int | None = None
    health_command: str = ""


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
