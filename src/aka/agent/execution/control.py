"""Suspend owned Agent work only for fully admitted host tool operations."""
from contextlib import contextmanager
import os
import signal
import threading
from aka.core.processes import process_snapshot
from aka.core.active_time import ActiveClock

class AgentProcessScope:
    """Track descendants, including detached children, without signaling peers."""
    def __init__(self, pid):
        if pid <= 1 or pid == os.getpid():
            raise ValueError("Agent process must be a separately spawned child")
        self.pid = pid
        self.identities = {}
        self.group_live = True
        self.empty_group_scans = 0
        root = process_snapshot().get(pid)
        self.root_start = root.start_token if root else None
        self.refresh()

    def refresh(self):
        table = process_snapshot()
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
    def __init__(self, *, suspend=True):
        self.suspend = suspend
        self.condition = threading.Condition(threading.RLock())
        self.scope = None
        self.closed = False
        self.pending = 0
        self.clock = ActiveClock()

    def attach(self, pid):
        with self.condition:
            if self.closed:
                raise RuntimeError("Agent session has already closed")
            self.scope = AgentProcessScope(pid)
            self.condition.notify_all()

    def poll(self):
        with self.condition:
            if self.scope and not self.closed:
                if self.pending:
                    self.scope.signal(signal.SIGSTOP)
                else:
                    self.scope.refresh()
                self.clock.checkpoint()

    @contextmanager
    def request(self):
        if not self.suspend:
            # A trusted custom runner owns execution; without a verifiable local
            # process tree, none of its elapsed time may be excluded.
            with self.condition:
                if self.closed:
                    raise PermissionError("Agent invocation is no longer active")
            yield
            return
        with self.condition:
            while self.scope is None and not self.closed:
                self.condition.wait(.05)
            if self.closed:
                raise PermissionError("Agent invocation is no longer active")
            try:
                self.scope.signal(signal.SIGSTOP)
                self.scope.signal(signal.SIGSTOP)
                if not self.pending:
                    self.clock.pause()
                self.pending += 1
            except BaseException:
                self.kill()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.pending -= 1
                if not self.pending and not self.closed:
                    try:
                        self.scope.signal(signal.SIGCONT)
                        self.clock.resume()
                    except BaseException:
                        self.kill()
                        raise

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
            self.clock.close()

    def kill(self):
        with self.condition:
            try:
                if self.scope:
                    try:
                        self.scope.signal(signal.SIGSTOP)
                        self.scope.signal(signal.SIGSTOP)
                    finally:
                        self.scope.signal(signal.SIGKILL)
            finally:
                self.close()
