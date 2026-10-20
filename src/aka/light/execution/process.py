"""Bounded executor process lifecycle, independent of Agent and HTTP state."""
from dataclasses import dataclass
from collections.abc import Callable
import logging
import os
import signal
import subprocess
import tempfile
import time
from aka.light.runtime.errors import RequestDispatchTimeout, SessionRevokedError, GatewayExecutionUnavailable


@dataclass(frozen=True)
class ExecutionControl:
    deadline: float
    request_id: str
    is_live: Callable[[], bool]
    acquire_slot: Callable[[], None]
    release_slot: Callable[[], None]


def run_gateway(command, cwd, environment, control):
    deadline = control.deadline
    control.acquire_slot()
    try:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            if not control.is_live():
                raise SessionRevokedError("Session revoked")
            if time.monotonic() >= deadline:
                raise RequestDispatchTimeout("Supervisor request deadline exceeded; no executor started")
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=errors, start_new_session=True)
            try:
                started = time.monotonic()
                next_notice = started + 60
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= next_notice:
                        logging.getLogger(__name__).warning(
                            "Supervisor Runtime request %s still running after %.0fs; %.0fs until deadline",
                            control.request_id or "direct", now - started, max(0, deadline - now))
                        next_notice = now + 60
                    oversized = os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > 64 * 1024 * 1024
                    if now >= deadline or not control.is_live() or oversized:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        # Kill surviving descendants even when leader exited.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise GatewayExecutionUnavailable("Request interrupted; outcome may be unknown")
                    time.sleep(0.1)
                output.seek(0)
                errors.seek(0)
                if os.fstat(output.fileno()).st_size > 4 * 1024 * 1024:
                    raise GatewayExecutionUnavailable(
                        "GPU output exceeds the parsing limit; outcome cannot be projected safely"
                    )
                return subprocess.CompletedProcess(command, process.returncode,
                    output.read(4 * 1024 * 1024 + 1).decode("utf-8", "replace"),
                    errors.read(128 * 1024 + 1).decode("utf-8", "replace"))
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
    finally:
        control.release_slot()
