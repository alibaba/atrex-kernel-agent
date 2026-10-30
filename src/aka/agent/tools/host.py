"""Scoped local transport lifetime, independent of operation contents and results."""
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import secrets
import socket
import socketserver
import tempfile
from threading import Event, RLock, Thread, Timer
import time
from types import MappingProxyType

from aka.contracts.agent import AgentToolBinding, AgentToolContext
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.contracts.filesystem_access import require_path_access
from .protocol import ENDPOINT_ENV, TOKEN_ENV, FRAME_BYTES, receive, receive_json, send, send_json


def _pinned_directory(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("tool workspace must be absolute without parent traversal")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_access(access):
    if not isinstance(access, FilesystemAccess):
        raise TypeError("Agent tools require an explicit filesystem scope")
    targets = set()
    for mount in access.mounts:
        for path in (mount.source, mount.target):
            if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
                raise PermissionError("Agent tool scope must contain absolute paths without symlinks")
        if not mount.source.exists() or mount.target in targets:
            raise PermissionError("Agent tool scope has missing or ambiguous grants")
        targets.add(mount.target)
    for path in (*access.hidden_paths, *access.read_only_paths):
        if not path.is_absolute() or ".." in path.parts:
            raise PermissionError("Agent tool restrictions must contain absolute paths")


@dataclass(frozen=True)
class ToolSession:
    environment: Mapping[str, str] = field(repr=False)
    filesystem_access: FilesystemAccess


@dataclass
class _Registration:
    workspace: Path
    source: Path
    descriptor: int
    identity: tuple
    access: FilesystemAccess
    bindings: dict = field(repr=False)
    deadline: float | None
    operations: dict = field(default_factory=dict, repr=False)
    connections: set = field(default_factory=set, repr=False)


class ToolHost:
    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="at-")
        self.directory = Path(self._temporary.name).resolve()
        self.endpoint = self.directory / "s"
        os.chmod(self.directory, 0o700)
        self._lock = RLock()
        self._registrations, self._connections = {}, set()
        self._closed = False
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                registration = None
                cancel, done = Event(), Event()
                monitor = None
                try:
                    with owner._lock:
                        if owner._closed:
                            raise PermissionError()
                        owner._connections.add(self.request)
                    self.request.settimeout(30)
                    token, registration = owner._authenticate(receive_json(self.rfile, "session", max_bytes=FRAME_BYTES))
                    with owner._lock:
                        owner._check(token, registration)
                        registration.connections.add(self.request)
                    selection = receive_json(self.rfile, "tool", max_bytes=FRAME_BYTES)
                    if not isinstance(selection, dict) or set(selection) != {"name"} or selection["name"] not in registration.bindings:
                        raise PermissionError()
                    channel, payload = receive(self.rfile)
                    if channel != "request":
                        raise ValueError()
                    self.request.settimeout(None)
                    with owner._lock:
                        owner._check(token, registration)
                        registration.operations[cancel] = done
                    def controls():
                        try:
                            while not done.is_set():
                                if receive_json(self.rfile, "cancel", max_bytes=FRAME_BYTES) != {}:
                                    raise ValueError()
                                cancel.set()
                        except (OSError, ValueError, ConnectionError):
                            pass  # Passive disconnect alone does not cancel admitted work.
                    monitor = Thread(target=controls, daemon=True)
                    monitor.start()
                    context = AgentToolContext(registration.workspace, registration.access, cancel, registration.deadline)
                    response = registration.bindings[selection["name"]].handle(payload, context)
                    if not isinstance(response, bytes):
                        raise TypeError()
                    send(self.wfile, "response", response)
                    send_json(self.wfile, "complete", {})
                except Exception as error:
                    code = "session_unavailable" if isinstance(error, PermissionError) else "invalid_request" if isinstance(error, (ValueError, TypeError)) else "outcome_unknown"
                    try:
                        send_json(self.wfile, "error", {"code": code})
                    except OSError:
                        pass
                finally:
                    done.set()
                    try:
                        self.request.shutdown(socket.SHUT_RD)
                    except OSError:
                        pass
                    if monitor is not None:
                        monitor.join()
                    with owner._lock:
                        owner._connections.discard(self.request)
                        if registration is not None:
                            registration.connections.discard(self.request)
                            registration.operations.pop(cancel, None)

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = False
            block_on_close = True
        try:
            self._server = Server(str(self.endpoint), Handler)
            os.chmod(self.endpoint, 0o600)
            self._thread = Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            self._thread.start()
        except BaseException:
            self._temporary.cleanup()
            raise

    def _check(self, token, registration):
        if self._closed or self._registrations.get(token) is not registration:
            raise PermissionError("Agent tool session is unavailable")
        if registration.deadline is not None and time.monotonic() >= registration.deadline:
            raise PermissionError("Agent tool deadline expired")
        current = _pinned_directory(registration.source)
        try:
            info = os.fstat(current)
            if (info.st_dev, info.st_ino) != registration.identity:
                raise PermissionError("Agent tool workspace was replaced")
        finally:
            os.close(current)

    def _authenticate(self, value):
        if (not isinstance(value, dict) or set(value) != {"version", "token", "workspace"}
                or type(value["version"]) is not int or value["version"] != 1
                or not isinstance(value["token"], str) or not isinstance(value["workspace"], str)):
            raise ValueError("invalid Agent tool session")
        with self._lock:
            registration = self._registrations.get(value["token"])
            if registration is None or Path(value["workspace"]) != registration.workspace:
                raise PermissionError("Agent tool session is unavailable")
            self._check(value["token"], registration)
            return value["token"], registration

    def validate_environment(self, workspace, environment):
        token = environment.get(TOKEN_ENV)
        if token is None and ENDPOINT_ENV not in environment:
            return {}
        if environment.get(ENDPOINT_ENV) != str(self.endpoint):
            raise PermissionError("Agent tool endpoint cannot change")
        self._authenticate({"version": 1, "token": token, "workspace": str(workspace)})
        return {ENDPOINT_ENV: str(self.endpoint), TOKEN_ENV: token}

    def revoke(self, token):
        with self._lock:
            registration = self._registrations.pop(token, None)
            if registration is None:
                return ()
            operations = tuple(registration.operations.items())
            for cancel, _ in operations:
                cancel.set()
            for connection in tuple(registration.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        return tuple(done for _, done in operations)

    @contextmanager
    def session(self, workspace, *, access, bindings, deadline=None):
        _validate_access(access)
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic()):
            raise ValueError("Agent tool deadline must be an unexpired monotonic time")
        selected = {}
        for binding in bindings:
            if (not isinstance(binding, AgentToolBinding) or not isinstance(binding.name, str)
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", binding.name) is None
                    or binding.name in selected or not callable(getattr(binding.handler, "handle", None))):
                raise ValueError("Agent tool bindings must have unique valid names and handlers")
            selected[binding.name] = binding.handler
        workspace = Path(workspace).absolute()
        source = require_path_access(access, workspace)
        descriptor = _pinned_directory(source)
        info = os.fstat(descriptor)
        registration = _Registration(workspace, source, descriptor, (info.st_dev, info.st_ino), access, selected, deadline)
        token = secrets.token_hex(32)
        with self._lock:
            if self._closed:
                os.close(descriptor)
                raise RuntimeError("Agent tool host is closed")
            self._registrations[token] = registration
        timer = Timer(deadline - time.monotonic(), self.revoke, args=(token,)) if deadline is not None else None
        if timer is not None:
            timer.daemon = True
            timer.start()
        try:
            yield ToolSession(MappingProxyType({ENDPOINT_ENV: str(self.endpoint), TOKEN_ENV: token}),
                FilesystemAccess((FilesystemMount(self.directory, self.directory, False),)))
        finally:
            pending = self.revoke(token)
            if timer is not None:
                timer.cancel()
                timer.join()
            with self._lock:
                settled = tuple(registration.operations.values())
            for done in (*pending, *settled):
                done.wait()
            os.close(descriptor)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tokens = tuple(self._registrations)
        for token in tokens:
            self.revoke(token)
        with self._lock:
            for connection in tuple(self._connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self._temporary.cleanup()
