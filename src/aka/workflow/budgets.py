"""Workflow phase clocks, independent of Agent invocations and backend limits."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from threading import Event

from aka.core.filesystem import durable_write_json, read_regular_bytes

_current = ContextVar('workflow_phase_budget', default=None)


class PhaseBudget:
    def __init__(self, expires_at, parent=None):
        self.expires_at, self.parent = expires_at, parent

    def remaining(self):
        seconds = self.expires_at - time.time()
        if self.parent is not None:
            seconds = min(seconds, self.parent.remaining())
        if seconds <= 0:
            raise TimeoutError('Workflow phase wall-clock budget exhausted')
        return seconds

    def is_set(self):
        return time.time() >= self.expires_at or (self.parent is not None and self.parent.is_set())

    def wait(self, seconds):
        end = min(self.expires_at, time.time() + seconds)
        while not self.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                return False
            Event().wait(min(remaining, .1))
        return True


@contextmanager
def baseline_budget(campaign):
    path = Path(campaign.workspace).resolve() / '.atrex_long_horizon/framework_baseline/deadline.json'
    seconds = campaign.framework_baseline_timeout
    now = time.time()
    try:
        raw = read_regular_bytes(path, limit=4097)
        if len(raw) > 4096:
            raise ValueError('baseline deadline exceeds size limit')
        value = json.loads(raw)
    except FileNotFoundError:
        value = {'schema': 1, 'policy': 'wall-clock', 'seconds': seconds,
                 'started_at': now, 'expires_at': now + seconds}
        durable_write_json(path, value)
    if (not isinstance(value, dict) or value.get('schema') != 1 or value.get('policy') != 'wall-clock'
            or value.get('seconds') != seconds
            or any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
                   for k in ('started_at', 'expires_at'))
            or value['expires_at'] != value['started_at'] + seconds or now < value['started_at']):
        raise ValueError('baseline deadline is invalid or its frozen budget changed')
    budget = PhaseBudget(value['expires_at'], _current.get())
    budget.remaining()
    token = _current.set(budget)
    try:
        yield budget
        budget.remaining()
    finally:
        _current.reset(token)


def remaining_timeout(requested):
    budget = _current.get()
    if budget is None:
        return requested
    remaining = budget.remaining()
    return remaining if requested is None else min(requested, remaining)


def cancellation(parent=None):
    budget = _current.get()
    if budget is None:
        return parent
    class Combined:
        def is_set(self):
            return budget.is_set() or (parent is not None and parent.is_set())
        def wait(self, seconds):
            end = time.time() + seconds
            while not self.is_set():
                remaining = end - time.time()
                if remaining <= 0:
                    return False
                Event().wait(min(.1, remaining))
            return True
    return Combined()


def bound_executor(executor):
    budget = _current.get()
    if budget is None:
        return executor
    class PhaseExecutor:
        def __getattr__(self, name):
            return getattr(executor, name)
        def run(self, command):
            budget.remaining()
            # The parent cancellation narrows waiting/execution without rewriting
            # the immutable evaluator request or asserting remote termination.
            return executor.run(replace(command, cancel_event=cancellation(command.cancel_event)))
        def query_job(self, reference, *, timeout, **options):
            return executor.query_job(reference, timeout=min(timeout, budget.remaining()), **options)
        def replay_job(self, command, *, idempotency_key):
            budget.remaining()
            return executor.replay_job(replace(command, cancel_event=cancellation(command.cancel_event)),
                                       idempotency_key=idempotency_key)
    return PhaseExecutor()
