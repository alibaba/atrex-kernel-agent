"""Call one selected Agent tool; lost replies never authorize new work."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
from threading import Event, Thread

from .protocol import ENDPOINT_ENV, TOKEN_ENV, receive, receive_json, send, send_json


class ToolSessionError(RuntimeError):
    def __init__(self, code, payload=None):
        self.code = code
        self.payload = payload
        super().__init__("Agent tool transport: " + code)


@contextmanager
def _controls(writer, cancel_event):
    done = Event()
    def cancel():
        try:
            send_json(writer, "cancel", {})
        except OSError:
            pass
    def monitor():
        while not done.wait(0.05):
            if cancel_event.is_set():
                cancel()
                return
    thread = Thread(target=monitor, daemon=True) if cancel_event is not None else None
    if thread is not None:
        thread.start()
    try:
        yield cancel
    finally:
        done.set()
        if thread is not None:
            thread.join()


def call_tool(name, payload, *, workspace, environment=None, cancel_event=None):
    if not isinstance(payload, bytes):
        raise TypeError("Agent tool payload must be immutable bytes")
    if cancel_event is not None and cancel_event.is_set():
        raise ToolSessionError("cancelled")
    values = os.environ if environment is None else environment
    endpoint, token = values.get(ENDPOINT_ENV), values.get(TOKEN_ENV)
    if not endpoint or not token:
        raise ToolSessionError("session_unavailable")
    received = None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(30)  # Admission IO only; operation deadlines belong to the host.
            connection.connect(endpoint)
            with connection.makefile("rb", buffering=0) as reader, connection.makefile("wb", buffering=0) as writer:
                send_json(writer, "session", {"version": 1, "token": token, "workspace": str(Path(workspace).absolute())})
                send_json(writer, "tool", {"name": name})
                send(writer, "request", payload)
                connection.settimeout(None)
                with _controls(writer, cancel_event) as cancel:
                    try:
                        channel, response = receive(reader)
                    except KeyboardInterrupt:
                        cancel()
                        channel, response = receive(reader)
                    if channel == "error":
                        error = json.loads(response)
                        raise ToolSessionError(error.get("code", "outcome_unknown"))
                    if channel != "response":
                        raise ValueError("invalid Agent tool response")
                    received = response
                    if receive_json(reader, "complete") != {}:
                        raise ValueError("invalid Agent tool completion")
                    return response
    except (OSError, ValueError, ConnectionError):
        raise ToolSessionError("outcome_unknown", received) from None
