"""Campaign-owned HTTP GPU/Wiki service; no Journal or promotion authority."""
from __future__ import annotations

import json
import argparse
import hashlib
import logging
import math
import os
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from supervisor.gateway import (
    EPISODE_EVALUATIONS_PATH, build_parser, configured_queue_wait_grace, validate_evaluation_options,
)
from supervisor.workspace import (
    MAX_FILE_BYTES, MAX_TOTAL_BYTES, InputSizeLimitError, publish, read_input, relative_path, snapshot,
)
from supervisor.measurement_records import JOB_ROOT_ENV, MeasurementStore
from supervisor.measurements import QUERY_KINDS

OWNER_ENV = "ATREX_AKA_RUNTIME_OWNER"
URL_ENV = "ATREX_AKA_RUNTIME_URL"
TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
DISPATCH_RETRY_SECONDS = 5
ROOT = Path(__file__).resolve().parents[1]
_RUNTIMES: dict[str, "SupervisorRuntime"] = {}
_RUNTIME_LOCK = threading.RLock()
AUTHORITY = frozenset({
    "--workspace", "--hardware", "--url", "--gateway-profile", "--ssh", "--ssh-init",
    "--ssh-gpu", "--ssh-runtime-bind", "--health-command", "--runtime-health-command",
    "--timeout", "--max-input-file-mb", "--max-output-file-mb",
})
FORBIDDEN = frozenset({"--preflight", "--check-health"})
WIKI_STORES = {"query_nl": "--store-root", "query_wiki": "--json-store", "query_hardware": "--store"}


class RequestValidationError(ValueError):
    """A repairable request error detected before execution can start."""


class RequestDispatchTimeout(RuntimeError):
    """A queue/deadline failure known to precede executor creation."""


class SessionRevokedError(RuntimeError):
    """Authorization was lost before executor creation."""


@contextmanager
def _validating_request():
    # Only request decoding/preparation belongs here, never execution, output
    # publication, response projection/encoding or temporary-directory cleanup.
    try:
        yield
    except (ValueError, UnicodeError) as error:
        raise RequestValidationError(str(error)) from error


class _RequestParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"{message}; use --kind OPERATION --help for accepted arguments")


def parse_gateway(argv):
    # argparse help must not write to process-global streams from an HTTP thread.
    options = argv[:argv.index("--")] if "--" in argv else argv
    if "--help" in options or "-h" in options:
        return None
    args = build_parser(_RequestParser).parse_args(argv)
    validate_evaluation_options(args)
    return args


def failure(message: str, *, repairable: bool = True, next_action: str | None = None) -> dict:
    return {"ok": False, "repairable": repairable, "error": {
        "message": message[:2000],
        "next_action": next_action or ("Correct the arguments, using full option names, then retry."
                        if repairable else "Report this blocker; do not bypass the Runtime or resubmit blindly."),
    }}


def validated_argv(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 512 or any(
        not isinstance(item, str) or "\x00" in item for item in value
    ):
        raise ValueError("argv must be a NUL-free string array with at most 512 entries")
    return list(value)


def filter_options(argv: list[str], owned: frozenset[str], forbidden: frozenset[str]) -> list[str]:
    """Strip full legacy authority flags; reject prefixes before argparse sees them."""
    result, index = [], 0
    while index < len(argv):
        word = argv[index]
        if word == "--":
            return result + argv[index:]
        option = word.split("=", 1)[0]
        if option.startswith("--") and option not in owned | forbidden and any(
            name.startswith(option) for name in owned | forbidden
        ):
            raise ValueError(f"Abbreviated authority option is not allowed: {option}")
        if option in forbidden:
            raise ValueError(f"Supervisor-owned option is not allowed: {option}")
        if option in owned:
            if "=" not in word:
                index += 1
                if index >= len(argv) or argv[index].startswith("-"):
                    raise ValueError(f"{option} requires a value")
        else:
            result.append(word)
        index += 1
    return result


def scrub_environment(environment: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in environment.items() if
            not key.startswith(("AGATE_", "ATREX_SANDBOX_", "ATREX_WIKI_", "GPU_WIKI_", "ATREX_GPU_WIKI_"))
            and key not in {OWNER_ENV, URL_ENV, TOKEN_ENV, JOB_ROOT_ENV, "ATREX_AKA_MEASUREMENT_REPETITIONS",
                            "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_BENCH_RUNTIME_ROOT"}}


@dataclass(frozen=True)
class RuntimeConfig:
    hardware: str
    timeout: int = 600
    url: str = ""
    profile: str = ""
    ssh: str = ""
    ssh_init: str = ""
    ssh_gpu: int | None = None
    health_command: str = ""
    private_reference_dir: Path | None = None
    atrex_bench_root: Path | None = None
    wiki_profile_root: Path | None = None
    task_id: str = ""
    optimization_mode: str = "leaderboard"
    workspace: Path | None = None
    # None preserves the Gateway execution + remote queue grace + cleanup budget.
    request_timeout: float | None = None
    queue_timeout: float = 60

    def __post_init__(self):
        for name in ("request_timeout", "queue_timeout"):
            value = getattr(self, name)
            if name == "request_timeout" and value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Runtime {name} must be a finite positive number of seconds")

    def command(self, workspace: Path) -> list[str]:
        result = [sys.executable, str(ROOT / "supervisor/gateway.py"),
                  "--workspace", str(workspace), "--hardware", self.hardware,
                  "--timeout", str(self.timeout)]
        for flag, value in (("--url", self.url), ("--gateway-profile", self.profile), ("--ssh", self.ssh)):
            if value:
                result += [flag, value]
        if self.ssh:
            if self.ssh_init:
                result += ["--ssh-init", self.ssh_init]
            if self.ssh_gpu is not None:
                result += ["--ssh-gpu", str(self.ssh_gpu)]
            if self.health_command:
                result += ["--health-command", self.health_command]
        return result


@dataclass
class Capability:
    workspace: Path
    context: dict[str, str]
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set only while holding this capability's request lock.
    deadline: float | None = field(default=None, repr=False)
    request_id: str = ""


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = False


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, result, *, retry_after: int | None = None):
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        if getattr(self, "request_id", None):
            self.send_header("X-Request-ID", self.request_id)
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def do_GET(self):
        self.reply(200 if self.path == "/healthz" else 404,
                   {"status": "ok"} if self.path == "/healthz" else failure("Unknown endpoint"))

    def do_POST(self):
        owner = self.server.owner
        self.request_id = uuid.uuid4().hex
        deadline = time.monotonic() + owner.request_timeout_seconds
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        with owner.lock:
            capability = owner.capabilities.get(token) if not owner.closed else None
        if capability is None:
            self.reply(401, failure("Missing or revoked Session capability", repairable=False))
            return
        if self.path not in {"/v1/gateway/execute", "/v1/wiki/query"}:
            self.reply(404, failure("Unknown endpoint"))
            return
        try:
            with _validating_request():
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported; supply Content-Length")
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= length <= MAX_REQUEST_BYTES:
                    self.reply(413, failure("Request exceeds 2 MiB or has no Content-Length"))
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
            with owner.request_lock(capability, deadline, self.request_id):
                with owner.lock:
                    if owner.closed or owner.capabilities.get(token) is not capability:
                        self.reply(401, failure("Session capability revoked", repairable=False))
                        return
                result = owner.execute(capability, self.path, request)
            self.reply(200, result)
        except RequestValidationError as error:
            self.reply(400, failure(str(error)))
        except RequestDispatchTimeout as error:
            if not owner._live(capability):
                self.reply(401, failure("Session capability revoked", repairable=False))
                return
            result = failure(str(error), next_action=(
                f"No job was submitted by this request. Wait at least {DISPATCH_RETRY_SECONDS} seconds, "
                "then retry the same request unchanged with backoff. Do not retry in a tight loop."))
            result["error"]["code"] = "request_not_started"
            result["retry_after_seconds"] = DISPATCH_RETRY_SECONDS
            logging.getLogger(__name__).warning(
                "Supervisor Runtime request %s timed out before dispatch; safe to retry after backoff",
                self.request_id)
            self.reply(429, result, retry_after=DISPATCH_RETRY_SECONDS)
        except SessionRevokedError:
            self.reply(401, failure("Session capability revoked", repairable=False))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            # Keep the traceback on the operator's stderr (logging.lastResort
            # handles an unconfigured Campaign logger), never in the response.
            # In particular, an exception that quotes the bearer must not leak it.
            diagnostic = traceback.format_exc().replace(token, "[Session capability]")
            logging.getLogger(__name__).error("Supervisor Runtime request %s failed\n%s",
                                              self.request_id, diagnostic)
            self.reply(503, failure("Runtime could not confirm the operation outcome", repairable=False))


class SupervisorRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.environment = {k: v for k, v in os.environ.items() if k not in {OWNER_ENV, URL_ENV, TOKEN_ENV}}
        self.gateway_queue_wait_grace = configured_queue_wait_grace(self.environment)
        repetitions = self.environment.get("ATREX_AKA_MEASUREMENT_REPETITIONS", "1")
        if repetitions not in {"1", "3"}:
            raise ValueError("ATREX_AKA_MEASUREMENT_REPETITIONS must be 1 or 3")
        self.measurement_repetitions = int(repetitions)
        # Freeze the same value for the enclosing deadline and its subprocesses.
        self.environment["ATREX_SANDBOX_QUEUE_WAIT_GRACE"] = str(self.gateway_queue_wait_grace)
        self.owner_id = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.capabilities: dict[str, Capability] = {}
        self.closed = False
        self.slots = threading.BoundedSemaphore(16)
        self.temporary = tempfile.TemporaryDirectory(prefix="aka-supervisor-")
        self.root = Path(self.temporary.name).resolve()
        self.audit_root = None
        if config.workspace:
            workspace = config.workspace.resolve()
            scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
            self.audit_root = workspace.parent / ".atrex-supervisor-runtime" / scope
            from .agent_home import open_private_directory
            with open_private_directory(self.audit_root):
                pass
        self.measurements = MeasurementStore((self.audit_root or self.root) / "measurements")
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.owner = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        with _RUNTIME_LOCK:
            _RUNTIMES[self.owner_id] = self
        try:
            self.thread.start()
        except BaseException:
            with _RUNTIME_LOCK:
                _RUNTIMES.pop(self.owner_id, None)
            self.server.server_close()
            self.temporary.cleanup()
            raise

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def request_timeout_seconds(self) -> float:
        if self.config.request_timeout is not None:
            return self.config.request_timeout
        return self.config.timeout + self.gateway_queue_wait_grace + 120

    def close(self):
        with self.lock:
            self.closed = True
            self.capabilities.clear()
        with _RUNTIME_LOCK:
            _RUNTIMES.pop(self.owner_id, None)
        self.server.shutdown()
        # Request processes observe closed/revoked state and terminate their own
        # group before server_close joins handlers. No orphan request threads.
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def revoke(self, token: str):
        with self.lock:
            self.capabilities.pop(token, None)

    @contextmanager
    def session(self, workspace: Path, environment: dict[str, str]):
        workspace = workspace.resolve(strict=True)
        if not workspace.is_dir() or workspace == ROOT:
            raise ValueError("Runtime Session needs a separate Campaign workspace")
        token = secrets.token_urlsafe(32)
        context = {key: environment[key] for key in (
            "ATREX_TELEMETRY_TRACE", "ATREX_TELEMETRY_ATTEMPT_ID", "ATREX_TELEMETRY_CAMPAIGN_ID",
            "ATREX_TELEMETRY_ITERATION_ID", "ATREX_ENVIRONMENT_STATE_FILE",
        ) if key in environment}
        capability = Capability(workspace, context)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            self.capabilities[token] = capability
        values = scrub_environment(environment)
        values.update({URL_ENV: self.url, TOKEN_ENV: token})
        try:
            yield values
        finally:
            self.revoke(token)

    def _live(self, capability):
        with self.lock:
            return not self.closed and any(value is capability for value in self.capabilities.values())

    def _acquire(self, lock, capability, deadline):
        wait_deadline = min(deadline, time.monotonic() + self.config.queue_timeout)
        while True:
            if not self._live(capability):
                raise SessionRevokedError("Session revoked")
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                raise RequestDispatchTimeout("Supervisor request queue deadline exceeded; no executor started")
            if lock.acquire(timeout=min(0.2, remaining)):
                return

    @contextmanager
    def request_lock(self, capability, deadline, request_id):
        self._acquire(capability.lock, capability, deadline)
        capability.deadline, capability.request_id = deadline, request_id
        try:
            yield
        finally:
            capability.deadline, capability.request_id = None, ""
            capability.lock.release()

    def run_executor(self, command, cwd, environment, capability):
        deadline = capability.deadline or (time.monotonic() + self.request_timeout_seconds)
        self._acquire(self.slots, capability, deadline)
        try:
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                if not self._live(capability):
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
                                capability.request_id or "direct", now - started, max(0, deadline - now))
                            next_notice = now + 60
                        oversized = os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > 64 * 1024 * 1024
                        if now >= deadline or not self._live(capability) or oversized:
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
                            raise RuntimeError("Request interrupted; outcome may be unknown")
                        time.sleep(0.1)
                    output.seek(0)
                    errors.seek(0)
                    if os.fstat(output.fileno()).st_size > 4 * 1024 * 1024:
                        raise RuntimeError("GPU output exceeds the parsing limit; outcome cannot be projected safely")
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
            self.slots.release()

    def execute(self, capability: Capability, route: str, request: dict) -> dict:
        with _validating_request():
            allowed = {"argv"} if route.endswith("execute") else {"argv", "tool"}
            if set(request) != allowed:
                raise ValueError(f"Request fields must be {sorted(allowed)}")
            argv = validated_argv(request["argv"])
        environment = dict(self.environment)
        environment.update(capability.context)
        for key in (OWNER_ENV, URL_ENV, TOKEN_ENV, "ATREX_PRIVATE_REFERENCE_DIR"):
            environment.pop(key, None)
        with tempfile.TemporaryDirectory(dir=self.root, prefix="request-") as temporary:
            staged = Path(temporary)
            with _validating_request():
                if route.endswith("query"):
                    command = self._wiki_command(capability, request["tool"], argv, staged, environment)
                    kind = "wiki"
                else:
                    argv = filter_options(argv, AUTHORITY, FORBIDDEN)
                    parsed = parse_gateway(argv)
                    if parsed is not None and parsed.kind in QUERY_KINDS:
                        from supervisor.measurements import query
                        return query(self.measurements, parsed, capability.workspace)
                    if parsed is not None and parsed.kind != "env":
                        snapshot(capability.workspace, staged)
                    (staged / ".orchestrator_mode.json").write_text(json.dumps({"mode": self.config.optimization_mode}))
                    # The canonical driver is trusted code, not a mutable Agent input.
                    if self.config.atrex_bench_root:
                        from .constants import ATREX_BENCH_HARNESS
                        shutil.copy2(ATREX_BENCH_HARNESS, staged / "test_kernel.py")
                    # Querying candidate files never grants access to other workspaces.
                    # The trusted evaluator checkout is shared, not copied per Session.
                    if self.config.atrex_bench_root:
                        environment["ATREX_BENCH_RUNTIME_ROOT"] = str(self.config.atrex_bench_root)
                    if self.config.private_reference_dir:
                        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(self.config.private_reference_dir)
                    # Keep the legacy report compiler's log alongside private records.
                    (staged / ".atrex_long_horizon").mkdir()
                    (staged / ".atrex_long_horizon/journal.json").write_text("{}")
                    if parsed is not None:
                        for value in ([] if parsed.no_sync else parsed.sync or ["profiles"]):
                            if relative_path(value).parts[0] not in {"profiles", "scratch"}:
                                raise ValueError("--sync must be inside profiles/ or scratch/")
                    command = self.config.command(staged) + argv
                    kind = "gateway"
            if kind == "gateway" and parsed is not None and parsed.kind != "env" and not parsed.dry_run:
                from supervisor.measurements import execute
                # Parse the Supervisor's authority flags too, so the exact target,
                # execution timeout and transport enter the task identity.
                effective = parse_gateway(command[2:])
                return execute(self, capability, staged, effective, argv, environment, command)
            # From this point execution may have submitted a job. Failures must
            # never be classified as repairable argument errors by the handler.
            process = self.run_executor(command, staged, environment, capability)
            self.audit_process(capability, kind, argv, process)
            if kind == "gateway":
                self.publish_evaluation_log(capability.workspace, staged)
                if process.returncode == 0:
                    self.publish_legacy_outputs(capability.workspace, staged, parsed)
            from supervisor.projection import project_response
            return project_response(process, generalized=self.config.private_reference_dir is not None,
                                    wiki=kind == "wiki", private_paths=(str(staged), str(self.root),
                                    str(self.config.private_reference_dir or ""), str(ROOT),
                                    self.config.url, str(self.config.atrex_bench_root or "")))

    def audit_process(self, capability, kind, argv, process, record_id=None):
        if self.audit_root:
            publish(self.audit_root, f"request-{capability.request_id or uuid.uuid4().hex}.json", json.dumps({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": kind, "argv": argv, "exit_code": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr,
                "gateway_record_id": record_id,
                "capture_limits": {"stdout_bytes": 4 * 1024 * 1024, "stderr_bytes": 128 * 1024},
            }, ensure_ascii=False).encode())

    def _wiki_command(self, capability, tool, argv, staged, environment):
        if not isinstance(tool, str) or tool not in WIKI_STORES:
            raise ValueError("Unsupported Wiki tool")
        # No prefix spelling or '--' escape may reinterpret private host files.
        forbidden = frozenset({"--store", "--store-root", "--json-store", "--keep-workspace"})
        argv = filter_options(argv, frozenset(), forbidden)
        for index, item in enumerate(argv):
            option = item.split("=", 1)[0]
            if option.startswith("--") and option != "--file" and "--file".startswith(option):
                raise ValueError("Use the full --file option")
            if option == "--file":
                raw = item.split("=", 1)[1] if "=" in item else (
                    argv[index + 1] if index + 1 < len(argv) else "")
                data = read_input(capability.workspace, raw, limit=128 * 1024)
                target = staged / "query.txt"
                target.write_bytes(data)
                if "=" in item:
                    argv[index] = "--file=" + str(target)
                else:
                    argv[index + 1] = str(target)
        if tool in {"query_nl", "query_wiki"}:
            # Strip user context limits, then enforce a bounded result.
            argv = filter_options(argv, frozenset({"--max-bytes"}), frozenset())
            argv += ["--brief", "--max-bytes", str(128 * 1024)]
        if self.config.wiki_profile_root:
            environment["ATREX_WIKI_PROFILE_ROOT"] = str(self.config.wiki_profile_root)
        environment["ATREX_WIKI_TASK_ID"] = self.config.task_id
        return [sys.executable, str(ROOT / "gpu-wiki/tools" / f"{tool}.py"), *argv]

    def publish_legacy_outputs(self, workspace, staged, args):
        # Profile artifacts remain part of the old Agent workflow. Only these
        # declared output trees can be returned, not source or control files.
        if args is None:
            return
        # Validate the entire union before changing any Agent output. Spool to
        # private disk instead of retaining up to 64 MiB of payloads in memory.
        total, visited = 0, 0
        paths = set()
        with tempfile.TemporaryDirectory(dir=self.root, prefix="outputs-") as temporary:
            ready = Path(temporary)
            for value in ([] if args.no_sync else (args.sync or ["profiles"])):
                path = relative_path(value)
                if path.parts[0] not in {"profiles", "scratch"}:
                    raise ValueError("--sync must be inside profiles/ or scratch/")
                source = staged / path
                if source.is_symlink():
                    raise ValueError("Synchronized output cannot be a symlink")
                files = source.rglob("*") if source.is_dir() else [source]
                for item in files:
                    visited += 1
                    if visited > 4096:
                        raise ValueError("Synchronized output exceeds 4096 entries")
                    if not item.is_file() or item.is_symlink():
                        continue
                    relative = item.relative_to(staged).as_posix()
                    if relative in paths:
                        continue
                    # Only successful bounded reads increase total, so remaining
                    # stays non-negative. Enforce both budgets on every read.
                    remaining = MAX_TOTAL_BYTES - total
                    try:
                        data = read_input(staged, relative, limit=min(MAX_FILE_BYTES, remaining))
                    except InputSizeLimitError as error:
                        if remaining <= MAX_FILE_BYTES:
                            message = (f"Synchronized output exceeds cumulative size limit of {MAX_TOTAL_BYTES} bytes; "
                                       f"{remaining} bytes remain while reading {relative!r}")
                        else:
                            message = (f"Synchronized output file {relative!r} exceeds per-file size limit "
                                       f"of {MAX_FILE_BYTES} bytes; {remaining} bytes remain in the sync budget")
                        raise ValueError(message) from error
                    total += len(data)
                    paths.add(relative)
                    publish(ready, relative, data)
            for relative in sorted(paths):
                publish(workspace, relative, read_input(ready, relative))

    def publish_evaluation_log(self, workspace, staged):
        # Failed correctness evaluations still belong in the legacy evidence
        # log, but their partial profiles/scratch files do not belong in outputs.
        log = staged / EPISODE_EVALUATIONS_PATH
        if log.is_file():
            from orchestrator.session_tail import read_regular_bytes
            data = read_regular_bytes(log, limit=MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Evaluation log exceeds the size limit")
            publish(workspace, EPISODE_EVALUATIONS_PATH,
                    data, append=True)


@contextmanager
def session_environment(workspace: Path, environment: dict[str, str]):
    owner_id = environment.get(OWNER_ENV)
    if not owner_id:
        raise RuntimeError("Campaign Runtime owner is required for a managed Session")
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(owner_id)
    if runtime is None:
        raise RuntimeError("Campaign Runtime is not active in this Supervisor")
    if environment.get("ATREX_AGENT_WORKSPACE_ROLE", "optimizer") != "optimizer":
        yield scrub_environment(environment)
        return
    with runtime.session(workspace, environment) as values:
        yield values
