"""Campaign-owned GPU/Wiki, Journal and sealed Episode handoff service."""
from __future__ import annotations

import json
import argparse
import hashlib
import logging
import math
import os
import re
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
from supervisor.journal import SupervisorJournalService, initialize_journal, load_journal, journal_lock
from supervisor.errors import AgentRequestError, RuntimeStateError
from .constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_HARNESS
from .episode_workspace import EpisodeWorkspace

OWNER_ENV = "ATREX_AKA_RUNTIME_OWNER"
URL_ENV = "ATREX_AKA_RUNTIME_URL"
TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
DISPATCH_RETRY_SECONDS = 5
ROOT = Path(__file__).resolve().parents[1]
_RUNTIMES: dict[str, "SupervisorRuntime"] = {}
_RUNTIME_LOCK = threading.RLock()


def supervisor_campaign_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return workspace.parent / ".atrex-supervisor-runtime" / scope


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


SOL_ACCEPTANCE_INPUTS = frozenset({
    "reference.py",
    "definition.json",
    "workload.jsonl",
    "solution.json",
    "test_kernel.py",
})


def candidate_acceptance_argv(*, atrex_bench: bool) -> list[str]:
    """Return the evaluator-specific, Supervisor-owned acceptance request.

    Native Atrex-Bench has a typed six-case correctness mode. SOL-ExecBench
    instead owns correctness through its full workload evaluator; typed controls
    cannot represent that contract and would prevent the required Dev fallback.
    """
    if atrex_bench:
        return [
            "--kind", "run", "--mode", "correctness_only",
            "--multi-seed", "5", "--no-sync",
        ]
    return ["--kind", "run", "--no-sync"]


def sol_evaluation_config(workspace: Path) -> bytes | None:
    """Read optional SOL policy from V0, never from mutable files or the index."""
    from .workspace_state import v0_baseline_commit

    try:
        revision = v0_baseline_commit(workspace)
        if not revision:
            raise RuntimeStateError("SOL evaluation requires a committed V0 configuration anchor")
        listing = subprocess.run(
            ["git", "ls-tree", "-z", "-l", revision, "--", "config.json"],
            cwd=workspace, capture_output=True, check=True, timeout=30,
        ).stdout
        if not listing:
            return None
        entry = re.fullmatch(rb"100(?:644|755) blob ([0-9a-f]{40}(?:[0-9a-f]{24})?) +([0-9]+)\tconfig\.json\x00", listing)
        if entry is None or int(entry[2]) > MAX_FILE_BYTES:
            raise RuntimeStateError("SOL V0 config.json must be a bounded regular Git blob")
        content = subprocess.run(
            ["git", "cat-file", "blob", entry[1].decode("ascii")],
            cwd=workspace, capture_output=True, check=True, timeout=30,
        ).stdout
        if len(content) != int(entry[2]) or not isinstance(json.loads(content), dict):
            raise RuntimeStateError("SOL V0 config.json must contain a valid JSON object")
        return content
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeStateError(
            "Cannot read the committed SOL evaluation configuration; no job was submitted"
        ) from error


def authoritative_candidate_record(record: dict, source: bytes, *, atrex_bench: bool) -> bool:
    """Validate evaluator identity without trusting the projected result alone."""
    request = record.get("request")
    if not isinstance(request, dict):
        return False
    inputs = request.get("inputs")
    options = request.get("options")
    if not isinstance(inputs, dict) or not isinstance(options, dict):
        return False
    common = (
        record.get("cacheable") is True
        and record.get("operation") == "evaluate"
        and inputs.get("kernel.py") == hashlib.sha256(source).hexdigest()
    )
    if not common:
        return False
    if atrex_bench:
        policy = options.get("evaluation_policy")
        return (
            isinstance(policy, dict)
            and policy.get("mode") in {"full", "correctness_only"}
            and policy.get("num_correctness_cases") == 6
        )
    return (
        SOL_ACCEPTANCE_INPUTS <= inputs.keys()
        and inputs.get("test_kernel.py") == hashlib.sha256(
            read_input(SOL_HARNESS.parent, SOL_HARNESS.name)
        ).hexdigest()
        and options.get("command") == ["python3", "test_kernel.py"]
        and not options.get("evaluation_input_path")
        and not options.get("evaluation_shapes_path")
    )


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
class JournalBinding:
    service: SupervisorJournalService
    lock: threading.Lock = field(default_factory=threading.Lock)
    view: EpisodeWorkspace | None = None


@dataclass
class Capability:
    workspace: Path
    context: dict[str, str]
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set only while holding this capability's request lock.
    deadline: float | None = field(default=None, repr=False)
    request_id: str = ""
    journal: JournalBinding | None = None


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
        if self.path not in {"/v1/gateway/execute", "/v1/wiki/query", "/v1/journal/execute", "/v1/plugins/execute"}:
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
        except AgentRequestError as error:
            self.reply(400, error.response)
        except RuntimeStateError as error:
            self.reply(503, error.response)
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
        from .plugins import PluginRegistry
        self.plugins = PluginRegistry()
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
        self.journals: dict[Path, JournalBinding] = {}
        self.closed = False
        self.slots = threading.BoundedSemaphore(16)
        self.temporary = tempfile.TemporaryDirectory(prefix="aka-supervisor-")
        self.root = Path(self.temporary.name).resolve()
        self.audit_root = None
        if config.workspace:
            workspace = config.workspace.resolve()
            self.audit_root = supervisor_campaign_root(workspace)
            from .agent_home import open_private_directory
            with open_private_directory(self.audit_root):
                pass
        self.measurements = MeasurementStore((self.audit_root or self.root) / "measurements")
        # The lock is private and survives restarts; never read an Agent-supplied lock.
        plugin_lock = (self.audit_root or self.root) / "plugins"
        self.plugins.check_lock(plugin_lock)
        publish(plugin_lock, ".atrex_plugins/lock.json", json.dumps(self.plugins.snapshot()).encode())
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
    def journals_root(self) -> Path:
        """Private Journal collection, sibling to the Campaign's Measurement Store."""
        return (self.audit_root or self.root) / "journals"

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

    def episode_journal_path(self, episode: int) -> Path:
        """Controller-only location, shared by registration and read-only recovery."""
        if type(episode) is not int or episode < 1:
            raise ValueError("Invalid controller Episode number")
        return self.journals_root / "episodes" / f"e{episode:08d}" / "journal.json"

    def read_episode_journal(self, episode: int) -> dict:
        """Read existing private state without registering or creating an Episode.

        Recovery can run before the workspace is rebound to a live capability.
        Terminal validation must still verify identity and finalization.
        """
        return load_journal(self.episode_journal_path(episode))

    def register_episode(self, workspace: Path, *, episode: int, base_commit: str,
                         branch: str, memory_version: int, minimum_experiments: int = 0,
                         supervisor_git: bool = False):
        """Trusted controller binding, never populated from an Agent request or file."""
        workspace = workspace.resolve(strict=True)
        if (type(episode) is not int or episode < 1 or type(memory_version) is not int
                or memory_version < 1 or not isinstance(branch, str) or not branch
                or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", base_commit) is None
                or type(minimum_experiments) is not int or minimum_experiments < 0):
            raise ValueError("Invalid controller Episode identity")
        path = self.episode_journal_path(episode)
        evidence = path.parent
        expected = dict(episode=episode, base_commit=base_commit, episode_branch=branch,
                        memory_version=memory_version)
        with journal_lock(path):
            if path.exists():
                value = load_journal(path)
                if any(value.get(key) != item for key, item in expected.items()):
                    raise RuntimeError("Private Runtime Journal does not match the controller Episode")
            else:
                initialize_journal(path, episode=episode, base_commit=base_commit, branch=branch,
                                   memory_version=memory_version)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            existing = self.journals.get(workspace)
            if existing is not None:
                if existing.service.path != path:
                    raise RuntimeError("Workspace already belongs to another Runtime Journal")
                # Explicit historical minimum-count policies are immutable after binding.
                # A repeated binding must not silently retain a different report gate.
                if existing.service.minimum_experiments != minimum_experiments:
                    raise RuntimeError(
                        "Runtime Journal minimum_experiments changed for an already-bound "
                        f"Episode: bound={existing.service.minimum_experiments}, "
                        f"requested={minimum_experiments}. Resume with the original "
                        "controller Episode trial contract; no binding was changed"
                    )
                if existing.service.supervisor_git != supervisor_git:
                    raise RuntimeError("Cannot change Git ownership during an Episode")
                return existing.view.prepare() if existing.view else workspace
            view = None
            agent_workspace = workspace
            if supervisor_git:
                from .hardware import hardware_vendor
                view = EpisodeWorkspace(workspace, evidence / "agent",
                                        is_ppu=hardware_vendor(self.config.hardware) == "ppu")
                agent_workspace = view.prepare()
            binding = JournalBinding(SupervisorJournalService(
                workspace=agent_workspace, git_workspace=workspace,
                campaign_root=self.journals_root, evidence_root=evidence,
                minimum_experiments=minimum_experiments, supervisor_git=supervisor_git), view=view)
            self.journals[workspace] = binding
            self.journals[agent_workspace] = binding
            return agent_workspace

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
            capability.journal = self.journals.get(workspace)
        values = scrub_environment(environment)
        values.update({URL_ENV: self.url, TOKEN_ENV: token})
        if capability.journal and capability.journal.view:
            from .episode_workspace import EPISODE_WORKSPACE_ENV
            values[EPISODE_WORKSPACE_ENV] = str(capability.journal.view.root)
            from .agent_sandbox import git_directory
            common = git_directory(capability.journal.view.worktree)
            if common is None:
                raise RuntimeError(
                    "Cannot resolve the managed Episode Git common directory; "
                    "refusing to start an Agent Session without private-path protection. "
                    "Repair the controller worktree's Git metadata before retrying"
                )
            values["ATREX_EPISODE_PRIVATE_PATHS"] = json.dumps([
                str(capability.journal.view.worktree), str(self.audit_root or self.root),
                str(common),
            ])
            values["GIT_CEILING_DIRECTORIES"] = str(capability.journal.view.root.parent)
        # Preparation can fail or race with shutdown; grant authority only after
        # every required private path is known, without holding the lock over Git.
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            self.capabilities[token] = capability
        try:
            yield values
        finally:
            self.revoke(token)
            if capability.journal and capability.journal.view:
                with capability.journal.lock:
                    primary = sys.exception()
                    try:
                        capability.journal.view.publish(
                            sealed_source=capability.journal.service.sealed_source(),
                        )
                    except Exception as error:
                        if primary is not None:
                            primary.add_note(f"Episode draft publication failed: {error}")
                        else:
                            raise

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

    def execute_trusted(self, workspace: Path, argv: list[str]) -> dict:
        """Controller-only execution/reuse; no Session, draft publication or HTTP switch."""
        capability = Capability(workspace.resolve(strict=True), {})
        token = secrets.token_urlsafe(32)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            self.capabilities[token] = capability
        try:
            with self.request_lock(capability, time.monotonic() + self.request_timeout_seconds, "acceptance"):
                return self.execute(capability, "/v1/gateway/execute", {"argv": argv}, reuse_completed=True)
        finally:
            self.revoke(token)

    def execute(self, capability: Capability, route: str, request: dict, *,
                reuse_completed=False, reuse_correctness=False, candidate_source: bytes | None = None) -> dict:
        if route == "/v1/plugins/execute":
            return self.execute_plugin(capability, request)
        if route == "/v1/journal/execute":
            return self.execute_journal(capability, request)
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
                        if candidate_source is not None:
                            # Controller-only override: acceptance measures the
                            # report's frozen source, not subsequent draft edits.
                            publish(staged, "kernel.py", candidate_source)
                        if capability.journal and capability.journal.view:
                            from .episode_workspace import PUBLIC_FILES
                            # An Agent draft is never the authority for evaluator inputs.
                            for name in PUBLIC_FILES:
                                try:
                                    content = read_input(capability.journal.view.worktree, name)
                                except FileNotFoundError:
                                    continue
                                publish(staged, name, content)
                            try:
                                source = read_input(staged, "kernel.py")
                            except FileNotFoundError:
                                # Standalone Dev probes need no candidate. Other
                                # operations enforce their inputs in measurement_inputs.
                                pass
                            else:
                                kernel = self.measurements.kernel(source)
                                publish(capability.journal.view.worktree,
                                        ".atrex_long_horizon/policy_review_request.json",
                                        json.dumps({"schema_version": 1, "kernel_id": kernel["kernel_id"]}).encode())
                    (staged / ".orchestrator_mode.json").write_text(json.dumps({"mode": self.config.optimization_mode}))
                    # The canonical driver is trusted code, not a mutable Agent input.
                    if self.config.atrex_bench_root:
                        shutil.copy2(ATREX_BENCH_HARNESS, staged / "test_kernel.py")
                    else:
                        shutil.copy2(SOL_HARNESS, staged / "test_kernel.py")
                        if parsed is not None and parsed.kind != "env" and (staged / "workload.jsonl").is_file():
                            controller_workspace = (capability.journal.service.git_workspace
                                if capability.journal else self.config.workspace or capability.workspace)
                            config = sol_evaluation_config(controller_workspace)
                            if config is None:
                                # An untracked Agent config must not introduce a
                                # policy absent from the pinned baseline.
                                (staged / "config.json").unlink(missing_ok=True)
                            else:
                                publish(staged, "config.json", config)
                    if parsed is not None and parsed.kind != "env":
                        # Compatibility Profile routes execute this entry
                        # through Dev/SSH. Keep it private, overwrite any Agent copy,
                        # and stage it before measurement identity/bundle capture.
                        shutil.copy2(PROFILE_DRIVER, staged / "profile_driver.py")
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
                return execute(self, capability, staged, effective, argv, environment, command,
                               reuse_completed=reuse_completed, reuse_correctness=reuse_correctness)
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

    def _validate_candidate_correctness(self, capability: Capability, source: bytes) -> dict:
        """Reuse or execute the evaluator-specific mandatory correctness gate.

        Atrex-Bench uses its typed six-case gate. SOL-ExecBench executes the
        official evaluator over all workload.jsonl records with their own
        tolerances. Run under the caller's Session/deadline, never a capability
        surviving its revocation.
        """
        from supervisor.measurements import result_from_response

        atrex_bench = self.config.atrex_bench_root is not None
        response = self.execute(
            capability,
            "/v1/gateway/execute",
            {"argv": candidate_acceptance_argv(atrex_bench=atrex_bench)},
            reuse_completed=True,
            reuse_correctness=atrex_bench,
            candidate_source=source,
        )
        if not self._live(capability):
            raise SessionRevokedError("Session revoked before report acceptance")
        try:
            result = result_from_response(response, "evaluate")
            identity = result or next((json.loads(line.split("=", 1)[1])
                for line in reversed(response["stdout"].splitlines())
                if line.startswith("[sandbox] RECORD_JSON=")), {})
            record_id = identity["gateway_record_id"]
            record = self.measurements.read(record_id)
            authoritative = authoritative_candidate_record(
                record, source, atrex_bench=atrex_bench,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeStateError(
                "Supervisor candidate correctness evidence is unavailable; report was not accepted",
                code="candidate_validation_unavailable",
            ) from error
        if not authoritative:
            raise RuntimeStateError(
                "Supervisor candidate correctness did not produce authoritative evidence; report was not accepted",
                code="candidate_validation_unavailable",
            )
        # Some confirmed candidate failures (e.g. compile errors) have no Eval
        # payload, only a cacheable terminal Record. Unknown/infra outcomes are
        # never cacheable and cannot reach this repairable rejection path.
        if (result is None and response["exit_code"] != 0) or (result and result.get("all_pass") is False):
            gate_name = (
                "multi-seed correctness (base case plus five additional seeds)"
                if atrex_bench else
                "SOL-ExecBench full-workload correctness"
            )
            raise AgentRequestError(
                f"The candidate failed Supervisor {gate_name}. "
                "The report was not accepted and no candidate commit was created.",
                code="candidate_correctness_failed", gateway_record_id=record_id,
                next_action=(
                    f"Inspect python3 tools/sandbox.py --kind record-read --record-id {record_id}. "
                    "Repair kernel.py, record a passing full Evaluate and its Experiment, then resubmit "
                    "episode-report with the matching selected_experiment_id. The Supervisor runs its "
                    "acceptance check automatically; do not repeat the failed GPU request unchanged. "
                    "If abandoning the candidate, submit an evidence-backed pivot instead."
                ),
            )
        result_complete = (
            result
            and result.get("all_pass") is True
            and response["exit_code"] == 0
            and not result.get("error")
        )
        if not atrex_bench:
            total = result.get("correctness_total") if result else None
            result_complete = (
                result_complete
                and type(total) is int
                and total > 0
                and result.get("correctness_passed") == total
            )
        if not result_complete:
            raise RuntimeStateError(
                "Supervisor candidate correctness result is inconsistent; report was not accepted"
            )
        evidence = {"gateway_record_id": record_id, "reused": response.get("reused") is True}
        if atrex_bench:
            evidence.update(additional_seeds=5, correctness_cases=6, evaluator="atrex_bench")
        else:
            evidence.update(additional_seeds=0, workloads=total, evaluator="sol_execbench")
        return evidence

    def execute_journal(self, capability: Capability, request: dict) -> dict:
        binding = capability.journal
        if binding is None:
            raise AgentRequestError("Journal tools require a controller-registered Long Horizon Episode", repairable=False,
                                    code="journal_unavailable", next_action="Use this session's Baseline finish procedure; do not supply paths or Episode IDs to create a Journal")
        self._acquire(binding.lock, capability, capability.deadline)
        try:
            if not self._live(capability):
                raise SessionRevokedError()
            try:
                with journal_lock(binding.service.path):
                    value = binding.service.execute(request, validate_candidate=(
                        lambda source: self._validate_candidate_correctness(capability, source)
                    ))
            except (AgentRequestError, RuntimeStateError):
                raise
            except ValueError as error:
                raise AgentRequestError(str(error), next_action=(
                    "Correct the indicated field or lifecycle state, then retry. Read skills/runtime-records/references/journal.md. "
                    "Use list/load-direction or list/load-experiment for IDs, and record-read for measurement evidence. "
                    "A rejected episode-report is not a completed Episode; fix it and submit again.")) from error
            response = {"exit_code": 0, "stdout": json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", "stderr": ""}
            self.audit_process(capability, "journal", [str(request.get("operation"))],
                               subprocess.CompletedProcess([], 0, response["stdout"], ""))
            return response
        finally:
            binding.lock.release()

    def audit_process(self, capability, kind, argv, process, record_id=None):
        if self.audit_root:
            publish(self.audit_root, f"request-{capability.request_id or uuid.uuid4().hex}.json", json.dumps({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": kind, "argv": argv, "exit_code": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr,
                "gateway_record_id": record_id,
                "capture_limits": {"stdout_bytes": 4 * 1024 * 1024, "stderr_bytes": 128 * 1024},
            }, ensure_ascii=False).encode())

    def execute_plugin(self, capability: Capability, request: dict) -> dict:
        from plugin_runtime.schema import validate_schema
        with _validating_request():
            if request == {"action": "list"}:
                return {"exit_code": 0, "stdout": json.dumps(self.plugins.public_catalog()) + "\n", "stderr": ""}
            if set(request) != {"action", "tool", "input"} or request["action"] != "call":
                raise ValueError("Use action=list, or action=call with tool and input")
            if not isinstance(request["tool"], str):
                raise ValueError("tool must be a catalog name")
            tool = self.plugins.validate_call(request["tool"], request["input"])
        if request["tool"] == "gpu-wiki.query":
            # Reuse the existing scoped, bounded Wiki executor and private audit.
            value = request["input"]
            argv = [value["request"]]
            for name in ("max_records", "max_bytes", "exclude"):
                if name in value:
                    argv += ["--" + name.replace("_", "-"), str(value[name])]
            return self.execute(capability, "/v1/wiki/query", {"tool": "query_nl", "argv": argv})
        with tempfile.TemporaryDirectory(dir=self.root, prefix="plugin-") as temporary:
            staged = Path(temporary)
            with _validating_request():
                snapshot(capability.workspace, staged)
            # Do not honor paths, environment or executable declarations from the Agent.
            invocation = dict(request, snapshot=self.plugins.snapshot())
            request_path = staged / ".plugin-request.json"
            request_path.write_text(json.dumps(invocation))
            environment = dict(self.environment)
            environment.update(capability.context)
            # Preserve catalog-wide environment semantics without rehashing every
            # plugin/resource in the child. These are cached operator declarations.
            environment.update(self.plugins.environment(staged))
            if self.config.wiki_profile_root:
                environment["ATREX_WIKI_PROFILE_ROOT"] = str(self.config.wiki_profile_root)
            environment["ATREX_WIKI_TASK_ID"] = self.config.task_id
            process = self.run_executor(
                [sys.executable, str(ROOT / "supervisor/plugin_executor.py"), str(request_path)],
                staged, environment, capability,
            )
            self.audit_process(capability, "plugin", [request["tool"]], process)
            if process.returncode:
                # Plugin errors can contain private resources or evaluator paths.
                raise RuntimeError("Plugin execution failed; inspect private request diagnostics")
            result = json.loads(process.stdout)
            validate_schema(tool["output_schema"], result, "output")
            rendered = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
            if len(rendered.encode()) > 256 * 1024:
                raise RuntimeError("Plugin response exceeds the public result limit")
            return {"exit_code": 0, "stdout": rendered, "stderr": ""}

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
            # Honor narrower context requests without permitting an unbounded result.
            requested_limits = []
            for index, item in enumerate(argv):
                if item == "--max-bytes":
                    if index + 1 >= len(argv):
                        raise ValueError("--max-bytes requires a positive byte limit")
                    requested_limits.append(int(argv[index + 1]))
                elif item.startswith("--max-bytes="):
                    requested_limits.append(int(item.split("=", 1)[1]))
            if any(value <= 0 for value in requested_limits):
                raise ValueError("--max-bytes must be positive")
            argv = filter_options(argv, frozenset({"--max-bytes"}), frozenset())
            argv += ["--brief", "--max-bytes", str(min([128 * 1024, *requested_limits]))]
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
            binding = self.journals.get(workspace)
            destination = binding.service.git_workspace if binding and binding.view else workspace
            publish(destination, EPISODE_EVALUATIONS_PATH,
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
