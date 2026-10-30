"""Cancellation ownership for one local Agent invocation."""
from __future__ import annotations

import os
import signal
import threading

from .process import descendant_process_groups, signal_process_groups


class ProcessControl:
    """Attach an owned Popen once; cancellation never starts or resumes work."""
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._groups = set()
        self._closed = False

    def attach(self, process):
        if process.pid <= 1 or process.pid == os.getpid():
            raise ValueError("Agent process must be a separately spawned child")
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            group = process.pid
        if group != process.pid:
            raise ValueError("Agent process must own a private process group")
        with self._lock:
            if self._process is not None:
                raise RuntimeError("Agent process has already attached")
            if self._closed:
                signal_process_groups(descendant_process_groups(process.pid) | {process.pid}, signal.SIGKILL)
                raise RuntimeError("Agent invocation closed before process attachment")
            self._process = process
            self._groups = {process.pid}
            self.poll()

    def poll(self):
        with self._lock:
            if self._process is not None and not self._closed and self._process.poll() is None:
                self._groups.update(descendant_process_groups(self._process.pid))

    def kill(self):
        with self._lock:
            if self._closed:
                return
            self.poll()
            self._closed = True
            signal_process_groups(self._groups, signal.SIGKILL)

    def close(self):
        """Release settled ownership; late revocation must not signal old groups."""
        with self._lock:
            self._closed = True
            self._process = None
            self._groups.clear()
