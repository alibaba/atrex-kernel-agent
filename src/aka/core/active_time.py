"""In-process active clocks; callers own durable policy and process suspension."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
from threading import RLock
import time

_lock = RLock()
_budgets = ContextVar('active_time_budgets', default=())


class ActiveBudget:
    def __init__(self, seconds, *, used=0.0, checkpoint=None):
        if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('active budget must be finite and positive')
        if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
            raise ValueError("used active time must be finite and nonnegative")
        self.seconds, self.used = seconds, used
        self._checkpoint = checkpoint
        self._clocks = set()

    def consumed(self):
        with _lock:
            return self.used + sum(clock.active_seconds() for clock in self._clocks)

    def remaining(self):
        return float('inf') if self.seconds is None else max(0.0, self.seconds - self.consumed())

    def is_set(self):
        return self.remaining() <= 0

    def checkpoint(self):
        with _lock:
            if self._checkpoint is not None:
                self._checkpoint(self.consumed(), bool(self._clocks))


@contextmanager
def budget_scope(budget):
    existing = _budgets.get()
    token = _budgets.set(existing if budget in existing else (*existing, budget))
    try:
        yield budget
    finally:
        _budgets.reset(token)


class ActiveClock:
    """One invocation, charged to every enclosing budget, including on errors."""
    def __init__(self):
        self._since = None
        self._used = 0.0
        self._closed = False
        self._started = False
        self._budgets = ()
        self._last_checkpoint = 0.0

    def start(self):
        with _lock:
            if self._started or self._closed:
                raise RuntimeError('active clock cannot be restarted')
            budgets = _budgets.get()
            if any(b.is_set() for b in budgets):
                raise TimeoutError('active-time budget exhausted')
            self._budgets = budgets
            self._started = True
            self._since = time.monotonic()
            for budget in self._budgets:
                budget._clocks.add(self)
            self.checkpoint(force=True)

    def active_seconds(self):
        with _lock:
            return self._used + (0.0 if self._since is None else max(0.0, time.monotonic() - self._since))

    def remaining(self, limit):
        with _lock:
            own = float('inf') if limit is None else max(0.0, limit - self.active_seconds())
            return min((own, *(budget.remaining() for budget in self._budgets)))

    def pause(self):
        with _lock:
            if self._since is not None:
                self._used = self.active_seconds()
                self._since = None
                self.checkpoint(force=True)

    def resume(self):
        with _lock:
            if not self._closed and self._since is None:
                self._since = time.monotonic()

    def checkpoint(self, *, force=False):
        with _lock:
            now = time.monotonic()
            if force or now - self._last_checkpoint >= 1:
                for budget in self._budgets:
                    budget.checkpoint()
                self._last_checkpoint = now

    def close(self):
        with _lock:
            if self._closed:
                return
            self._used = self.active_seconds()
            self._since = None
            self._closed = True
            for budget in self._budgets:
                budget._clocks.discard(self)
                budget.used += self._used
            self.checkpoint(force=True)
