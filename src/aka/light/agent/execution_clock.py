"""Supervisor-owned active-time budgets and Runtime request suspension.

The HTTP handler lives outside the Provider's process tree. It stops that tree
only after receiving a complete authenticated request, and releases it before
writing the response. Client backgrounding/disconnection cannot release a wait.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
import signal
import subprocess
import threading
import time

from aka.light.agent.recovery_processes import session_process_snapshot

DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
_BUDGET = ContextVar("aka_agent_active_budget", default=None)


def validate_timeout(value, name="episode_timeout_seconds"):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


class ActiveBudget:
    def __init__(self, seconds=DEFAULT_TIMEOUT_SECONDS):
        self.seconds = validate_timeout(seconds)
        self.used = 0.0

    @property
    def remaining(self):
        return max(0.0, self.seconds - self.used)


@contextmanager
def episode_budget(seconds):
    """One budget across report repair and fresh output-limit recovery sessions."""
    budget = ActiveBudget(seconds)
    token = _BUDGET.set(budget)
    try:
        yield budget
    finally:
        _BUDGET.reset(token)


def current_budget():
    return _BUDGET.get()


class AgentProcessScope:
    """Track descendants, including detached children, without signaling peers."""
    def __init__(self, pid):
        if pid <= 1 or pid == os.getpid():
            raise ValueError("Agent process must be a separately spawned child")
        self.pid = pid
        self.identities = {}
        self.group_live = True
        self.empty_group_scans = 0
        root = session_process_snapshot().get(pid)
        self.root_start = root.start_token if root else None
        self.refresh()

    def refresh(self):
        table = session_process_snapshot()
        if not table:
            raise RuntimeError("Cannot inspect the Agent process tree for Runtime suspension")
        owned = {pid for pid, record in table.items()
                 if self.identities.get(pid) == record.start_token}
        root = table.get(self.pid)
        if root and self.root_start == root.start_token:
            owned.add(self.pid)
        # spawn_owned_session created this private PGID. The leader can exit
        # before attachment while its children still hold the output pipes.
        # Remaining members reserve the group ID; once it disappears (or its
        # leader identity changes), never adopt that numeric group again.
        if self.group_live:
            if root is not None and self.root_start != root.start_token:
                self.group_live = False
            else:
                members = {pid for pid, record in table.items() if record.pgid == self.pid}
                owned.update(members)
                # Process enumeration is not atomic: a parent may disappear
                # after the census while a just-forked child was not in it yet.
                self.empty_group_scans = 0 if members else self.empty_group_scans + 1
                if self.empty_group_scans >= 2:
                    self.group_live = False
        while True:
            children = {pid for pid, record in table.items() if record.ppid in owned}
            if children <= owned:
                break
            owned.update(children)
        for pid in owned:
            self.identities[pid] = table[pid].start_token
        return {pid: table[pid] for pid in owned}

    def signal(self, sig):
        records = self.refresh()
        # Stop the parent first, then any children already forking at that point.
        for pid in sorted(records, key=lambda value: value != self.pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


class SessionControl:
    """One Provider launch; overlapping requests pause the clock only once."""
    def __init__(self):
        self.condition = threading.Condition(threading.RLock())
        self.scope = None
        self.closed = False
        self.pending = 0
        self.started = None
        self.paused_since = None
        self.wait_seconds = 0.0
        self.finished_active_seconds = None

    def attach(self, pid):
        with self.condition:
            if self.closed:
                raise RuntimeError("Agent session has already closed")
            self.started = time.monotonic()
            self.scope = AgentProcessScope(pid)
            self.condition.notify_all()

    def active_seconds(self):
        with self.condition:
            if self.finished_active_seconds is not None:
                return self.finished_active_seconds
            if self.started is None:
                return 0.0
            now = self.paused_since if self.paused_since is not None else time.monotonic()
            return max(0.0, now - self.started - self.wait_seconds)

    def poll(self):
        with self.condition:
            if self.scope and not self.closed:
                if self.pending:
                    self.scope.signal(signal.SIGSTOP)
                else:
                    self.scope.refresh()

    @contextmanager
    def request(self):
        with self.condition:
            while self.scope is None and not self.closed:
                self.condition.wait(0.05)
            if self.closed:
                raise RuntimeError("Agent session closed before Runtime dispatch")
            if not self.pending:
                self.paused_since = time.monotonic()
            self.pending += 1
            try:
                self.scope.signal(signal.SIGSTOP)
                # The first pass freezes known parents. Re-scan before dispatch
                # for children forked between that snapshot and the stop signals.
                self.scope.signal(signal.SIGSTOP)
            except BaseException:
                self.pending -= 1
                # A partially stopped tree must never be abandoned frozen.
                self.closed = True
                self.scope.signal(signal.SIGKILL)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.pending -= 1
                if not self.pending:
                    self.wait_seconds += time.monotonic() - self.paused_since
                    self.paused_since = None
                    if not self.closed:
                        try:
                            self.scope.signal(signal.SIGCONT)
                        except BaseException:
                            self.kill()
                            raise

    def close(self):
        """Never resume a cancelled Agent, even if a late request completes."""
        with self.condition:
            if self.finished_active_seconds is None:
                self.finished_active_seconds = self.active_seconds()
            self.closed = True
            self.condition.notify_all()

    def kill(self):
        with self.condition:
            self.close()
            if self.scope:
                try:
                    # Freeze parents before teardown so a forking child cannot
                    # disappear from the next ancestry scan when its parent dies.
                    self.scope.signal(signal.SIGSTOP)
                    self.scope.signal(signal.SIGSTOP)
                finally:
                    self.scope.signal(signal.SIGKILL)


class ActiveCommunication:
    """Drain once to EOF while the caller supervises active-time and suspension."""
    def __init__(self, communicate, process, control):
        self.process, self.control = process, control
        self.done = threading.Event()
        self.result = None
        self.error = None

        def drain():
            try:
                self.result = communicate()
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.reader = threading.Thread(target=drain, name=f"agent-io-{process.pid}", daemon=True)
        self.reader.start()

    def __call__(self, timeout=None):
        deadline = timeout  # Total invocation active time, including process setup.
        while not self.done.wait(0.05):
            self.control.poll()
            if deadline is not None and self.control.active_seconds() >= deadline:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
        self.reader.join()
        if self.error:
            raise self.error
        return self.result
