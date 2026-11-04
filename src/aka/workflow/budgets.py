"""Workflow-owned durable allowances for active Agent work, not backend wall time."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import time
from threading import Event

from aka.core.active_time import ActiveBudget, budget_scope
from aka.core.filesystem import durable_write_json, read_regular_bytes

_current = ContextVar('workflow_phase_budget', default=None)


class PhaseBudget(ActiveBudget):
    def require_remaining(self):
        remaining = self.remaining()
        if remaining <= 0:
            raise TimeoutError('Workflow active-time budget exhausted')
        return remaining


@contextmanager
def durable_budget(path, seconds, *, resumed=False):
    """Reopen used active time; never infer it from an old elapsed-time record.

    An in-flight checkpoint after abrupt host death is uncertain, so refuse
    automatic continuation rather than refunding unrecorded Agent work.
    """
    if path is None:
        if resumed:
            raise ValueError('resumed phase requires a durable budget directory')
        with budget_scope(PhaseBudget(seconds)) as budget:
            yield budget
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            raw = read_regular_bytes(path, limit=4097)
            if len(raw) > 4096:
                raise ValueError('active budget exceeds size limit')
            value = json.loads(raw)
        except FileNotFoundError:
            if resumed:
                raise ValueError('resumed phase has no frozen active budget')
            value = {'schema': 2, 'policy': 'active-time', 'seconds': seconds,
                     'used_seconds': 0.0, 'inflight': False}
        if (not isinstance(value, dict)
                or set(value) != {'schema', 'policy', 'seconds', 'used_seconds', 'inflight'}
                or type(value.get('schema')) is not int or value['schema'] != 2
                or type(value.get('seconds')) is bool
                or value.get('policy') != 'active-time' or value.get('seconds') != seconds
                or type(value.get('used_seconds')) not in (int, float)
                or not math.isfinite(value['used_seconds']) or value['used_seconds'] < 0
                or type(value.get('inflight')) is not bool):
            raise ValueError('active budget is invalid or incompatible; reconcile old wall-clock records explicitly')
        if value['inflight']:
            raise ValueError('active budget has interrupted in-flight work; reconcile used time before resuming')
        def checkpoint(used, inflight):
            durable_write_json(path, dict(value, used_seconds=used, inflight=inflight))
        budget = PhaseBudget(seconds, used=value['used_seconds'], checkpoint=checkpoint)
        budget.checkpoint()
        with budget_scope(budget):
            yield budget
    finally:
        os.close(fd)


@contextmanager
def baseline_budget(campaign):
    path = Path(campaign.workspace).resolve() / '.atrex_long_horizon/framework_baseline/deadline.json'
    with durable_budget(path, campaign.framework_baseline_timeout) as budget:
        budget.require_remaining()
        token = _current.set(budget)
        try:
            yield budget
            budget.require_remaining()
        finally:
            _current.reset(token)


def remaining_timeout(requested):
    budget = _current.get()
    if budget is None:
        return requested
    remaining = budget.require_remaining()
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
            budget.require_remaining()
            # The parent cancellation narrows waiting/execution without rewriting
            # the immutable evaluator request or asserting remote termination.
            return executor.run(replace(command, cancel_event=cancellation(command.cancel_event)))
        def query_job(self, reference, *, timeout, **options):
            return executor.query_job(reference, timeout=timeout, **options)
        def replay_job(self, command, *, idempotency_key):
            budget.require_remaining()
            return executor.replay_job(replace(command, cancel_event=cancellation(command.cancel_event)),
                                       idempotency_key=idempotency_key)
    return PhaseExecutor()
