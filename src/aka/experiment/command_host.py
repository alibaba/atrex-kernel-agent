"""Forward one Agent session's commands to its already selected Experiment."""
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import secrets
import socket
import socketserver
import stat
import tempfile
from threading import Event, Lock, Thread

from aka.contracts.execution import CommandSpec, ExecutionFile, JobResult, OutputOptions, ProfileOptions
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.contracts.filesystem_access import require_path_access
from aka.execution.configuration import transport_environment_key, validate_workload_environment
from .command_access import access_data, client_access, pinned, read_input
from .command_protocol import ENDPOINT_ENV, TOKEN_ENV, receive_json, receive, send, send_json
from .scoped_publication import validate_target

ADMISSION_READ_TIMEOUT = 30


@dataclass
class _Session:
    workspace: Path
    source: Path
    descriptor: int
    identity: tuple
    access: FilesystemAccess
    operations: dict = field(default_factory=dict)
    connections: set = field(default_factory=set)


class ExperimentCommandHost:
    def __init__(self, open_experiment, capabilities, *, hardware=""):
        if not callable(open_experiment):
            raise TypeError("command host requires an Experiment context opener")
        self.open_experiment, self.capabilities, self.hardware = open_experiment, capabilities, hardware
        self._temporary = tempfile.TemporaryDirectory(prefix="ax-")
        self.directory = Path(self._temporary.name).resolve()
        self.endpoint = self.directory / "s"
        os.chmod(self.directory, 0o700)
        self._lock = Lock()
        self._sessions, self._connections = {}, set()
        self._closed = False
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                session = None
                result = None
                cancel, done, send_lock = Event(), Event(), Lock()
                monitor = None

                def reply(channel, value):
                    with send_lock:
                        try:
                            send_json(self.wfile, channel, value)
                        except OSError:
                            pass  # Lost replies never authorize resubmission.

                def observe(result):
                    data = result.as_dict()
                    data["stdout"] = data["stderr"] = ""
                    reply("observation", data)

                try:
                    with owner._lock:
                        if owner._closed:
                            raise RuntimeError("command host is closed")
                        owner._connections.add(self.request)
                    self.request.settimeout(ADMISSION_READ_TIMEOUT)
                    hello = receive_json(self.rfile, "session")
                    token, session = owner._authenticate(hello)
                    with owner._lock:
                        owner._check(token, session)
                        session.connections.add(self.request)
                    reply("access", access_data(client_access(session.access)))
                    value = receive_json(self.rfile, "command")
                    inputs = []
                    for item in value.get("input_files", ()):
                        channel, content = receive(self.rfile)
                        if channel != "input" or not isinstance(item, dict) or set(item) != {"path", "mode"}:
                            raise ValueError("invalid command input stream")
                        inputs.append(ExecutionFile(item["path"], content, item["mode"]))
                    command = owner._command(session, value, tuple(inputs))
                    with owner._lock:
                        owner._check(token, session)
                        session.operations[cancel] = done
                    self.request.settimeout(None)
                    if owner.capabilities.cancellation:
                        command = replace(command, cancel_event=cancel)
                    command = replace(command, on_observation=observe)

                    def controls():
                        try:
                            while not done.is_set():
                                value = receive_json(self.rfile, "cancel")
                                if value != {}:
                                    raise ValueError("invalid cancellation control")
                                if owner.capabilities.cancellation:
                                    cancel.set()
                                else:
                                    reply("cancel_refused", {"code": "unsupported"})
                        except (OSError, ValueError, ConnectionError):
                            pass  # Passive EOF does not cancel an admitted Job.

                    monitor = Thread(target=controls, daemon=True)
                    monitor.start()
                    with owner.open_experiment(session.workspace, session.access) as experiment:
                        result = experiment.run_command(command)
                    if not isinstance(result, JobResult):
                        raise TypeError("Experiment.run_command must return JobResult")
                    owner._result_files(session, command, result)
                    with send_lock:
                        try:
                            data = result.as_dict()
                            data["stdout"] = data["stderr"] = ""
                            send_json(self.wfile, "result", {"result": data, "error": None,
                                "files": [{"path": f.path, "mode": f.mode} for f in result.files]})
                            send(self.wfile, "stdout", result.stdout.encode())
                            send(self.wfile, "stderr", result.stderr.encode())
                            for output in result.files:
                                send(self.wfile, "output", output.content)
                            send_json(self.wfile, "complete", {})
                        except OSError:
                            pass
                except BaseException as error:
                    outcome = getattr(error, "outcome", None) or getattr(error, "execution_outcome", None) or result
                    data = outcome.as_dict() if isinstance(outcome, JobResult) else None
                    if data is not None:
                        data["stdout"] = data["stderr"] = ""
                    reply("result", {"result": data, "error": owner._error(error), "files": []})
                    # Error receipts contain original facts, never raw private output.
                    try:
                        with send_lock:
                            send(self.wfile, "stdout", b"")
                            send(self.wfile, "stderr", b"")
                            send_json(self.wfile, "complete", {})
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
                        if session is not None:
                            session.connections.discard(self.request)
                            session.operations.pop(cancel, None)

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = False
            block_on_close = True

        try:
            self._server = Server(str(self.endpoint), Handler)
            os.chmod(self.endpoint, 0o600)
            self._thread = Thread(target=self._server.serve_forever,
                                  kwargs={"poll_interval": 0.1}, daemon=True)
            self._thread.start()
        except BaseException:
            self._temporary.cleanup()
            raise

    @contextmanager
    def session(self, workspace, access):
        if not isinstance(access, FilesystemAccess):
            raise TypeError("Agent command session requires an explicit filesystem scope")
        workspace = Path(workspace).absolute()
        source = require_path_access(access, workspace)
        with pinned(source) as original:
            identity = os.fstat(original)
            if not stat.S_ISDIR(identity.st_mode):
                raise ValueError("command workspace must be a directory")
            descriptor = os.dup(original)
        registration = _Session(workspace, source, descriptor, (identity.st_dev, identity.st_ino), access)
        token = secrets.token_hex(32)
        with self._lock:
            if self._closed:
                os.close(descriptor)
                raise RuntimeError("command host is closed")
            self._sessions[token] = registration
        try:
            yield ({ENDPOINT_ENV: str(self.endpoint), TOKEN_ENV: token},
                   FilesystemAccess((FilesystemMount(self.directory, self.directory, False),)))
        finally:
            self._revoke(token, registration)
            os.close(descriptor)

    def _revoke(self, token, session):
        with self._lock:
            self._sessions.pop(token, None)
            operations = tuple(session.operations.items())
            for cancel, _ in operations:
                if self.capabilities.cancellation:
                    cancel.set()
            for connection in tuple(session.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        for _, done in operations:
            done.wait()

    def _check(self, token, session):
        if self._closed or self._sessions.get(token) is not session:
            raise PermissionError("command session is unavailable")
        with pinned(session.source) as descriptor:
            current = os.fstat(descriptor)
            original = os.fstat(session.descriptor)
            if ((current.st_dev, current.st_ino) != session.identity
                    or (original.st_dev, original.st_ino) != session.identity):
                raise PermissionError("registered command workspace was replaced")

    def _authenticate(self, value):
        if not isinstance(value, dict) or set(value) != {"version", "token", "workspace"} or value["version"] != 1:
            raise ValueError("invalid command session envelope")
        if not isinstance(value["token"], str) or not isinstance(value["workspace"], str):
            raise ValueError("invalid command session identity")
        token = value["token"]
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                raise PermissionError("command session is unavailable")
            if Path(value["workspace"]) != session.workspace:
                raise PermissionError("command workspace differs from its session")
            self._check(token, session)
        return token, session

    def _command(self, session, value, inputs):
        allowed = {"command", "timeout", "hardware", "operation", "artifacts", "num_gpus", "languages",
                   "requirements", "environment", "profile_options", "output_options", "input_files"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("command transport accepts only public inputs")
        if value.get("hardware") not in (None, "", self.hardware):
            raise ValueError("Agent cannot change selected hardware")
        if value.get("requirements"):
            raise ValueError("Agent cannot change Task runtime requirements")
        for name in ("command", "artifacts", "languages", "requirements"):
            if not isinstance(value.get(name, []), list) or any(not isinstance(v, str) for v in value.get(name, [])):
                raise ValueError("command inputs must contain strings")
        environment = value.get("environment", {})
        if (not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                or transport_environment_key(k) for k, v in environment.items())):
            raise ValueError("transport configuration cannot enter workload environment")
        command = CommandSpec(session.workspace, tuple(value.get("command", ())),
            timeout=value.get("timeout", 600), hardware=self.hardware, operation=value.get("operation", "run"),
            artifacts=tuple(value.get("artifacts", ())), environment=environment,
            num_gpus=value.get("num_gpus", 1), languages=tuple(value.get("languages", ())),
            filesystem_access=session.access, input_files=inputs,
            profile_options=ProfileOptions(**value["profile_options"]) if value.get("profile_options") is not None else None,
            output_options=OutputOptions(**value["output_options"]) if value.get("output_options") is not None else None)
        validate_workload_environment(command.environment)
        self.capabilities.require(command)
        for item in inputs:
            frozen = read_input(session.access, session.workspace / item.path)
            if frozen != (item.content, item.mode):
                raise PermissionError("input differs from its authorized snapshot")
        for name in command.artifacts:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("command output must be workspace-relative")
            target = require_path_access(session.access, session.workspace / relative, write=True)
            validate_target(target / ".command-output" if target.is_dir() else target)
        return command

    @staticmethod
    def _result_files(session, command, result):
        for item in result.files:
            relative = Path(item.path)
            if not any(relative == Path(name) or relative.is_relative_to(Path(name)) for name in command.artifacts):
                raise PermissionError("result file is outside requested output paths")
            require_path_access(session.access, session.workspace / relative)

    @staticmethod
    def _error(error):
        from aka.contracts.execution import ExecutionPending, ExecutionUnavailable, UnsupportedExecution
        from aka.contracts.evaluation import EvaluationUnavailable
        from aka.contracts.experiment import ExperimentRecordError
        if isinstance(error, UnsupportedExecution):
            return {"category": "execution", "code": "unsupported"}
        if isinstance(error, ExecutionPending):
            return {"category": "execution", "code": "pending"}
        if isinstance(error, ExecutionUnavailable):
            return {"category": "execution", "code": "unavailable"}
        if isinstance(error, EvaluationUnavailable):
            return {"category": "evaluation", "code": "unavailable"}
        if isinstance(error, ExperimentRecordError):
            return {"category": "record", "code": "unavailable"}
        if isinstance(error, PermissionError):
            return {"category": "request", "code": "unauthorized"}
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return {"category": "request", "code": "invalid_request"}
        return {"category": "experiment", "code": "operation_failed"}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            registrations = tuple(self._sessions.items())
        for token, session in registrations:
            self._revoke(token, session)
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
