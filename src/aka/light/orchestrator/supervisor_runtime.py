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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from aka.light.runtime.paths import legacy_resource_root

from aka.light.supervisor.gateway import (
    EPISODE_EVALUATIONS_PATH, build_parser, configured_queue_wait_grace, find_agate, validate_evaluation_options,
)
from aka.light.supervisor.workspace import (
    MAX_FILE_BYTES, MAX_TOTAL_BYTES, InputSizeLimitError, publish, read_input, relative_path, snapshot,
)
from aka.light.supervisor.measurement_records import JOB_ROOT_ENV
from aka.light.supervisor.measurements import QUERY_KINDS
from aka.light.supervisor.journal import SupervisorJournalService, load_journal, journal_lock
from aka.light.supervisor.errors import AgentRequestError, RuntimeStateError, error_response
from aka.light.supervisor.request_feedback import augment_error
from aka.light.supervisor.plugin_output import (
    MAX_OUTPUT_BYTES, check_executor_output_failure, project_plugin_output, raise_output_error,
)
from .constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_HARNESS
from .episode_workspace import EpisodeWorkspace
from aka.light.acceptance.provider import CandidateValidation
from aka.light.runtime.errors import (
    RequestValidationError, RequestDispatchTimeout, GatewayExecutionUnavailable, SessionRevokedError,
)
from aka.light.runtime.paths import supervisor_campaign_root
from aka.light.wiki.config import WikiConfig
from aka.light.wiki.service import WikiService

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





@contextmanager
def _validating_request():
    # Only request decoding/preparation belongs here, never execution, output
    # publication, response projection/encoding or temporary-directory cleanup.
    try:
        yield
    except (AgentRequestError, RuntimeStateError):
        raise
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


def failure(message: str, *, repairable: bool = True, next_action: str | None = None,
            code: str | None = None) -> dict:
    from aka.light.supervisor.errors import UNKNOWN_OUTCOME
    return error_response(message, code=code or ("invalid_request" if repairable else "outcome_unknown"),
                          repairable=repairable, next_action=next_action or (
                              "Correct the arguments, using full option names, then retry."
                              if repairable else UNKNOWN_OUTCOME))


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
            and not key.startswith(("AKA_AGENT_", "AKA_LAUNCH_"))
            and key not in {OWNER_ENV, URL_ENV, TOKEN_ENV, JOB_ROOT_ENV, "ATREX_AKA_MEASUREMENT_REPETITIONS",
                            "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_BENCH_RUNTIME_ROOT",
                            "ATREX_AKA_IDENTITY_KEY", "ATREX_AKA_REPO_ROOT",
                            "ATREX_INTERNAL_RESOURCES_ROOT"}}


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
    measurement_repetitions: int = 1
    wiki_enabled: bool = False
    wiki_options: WikiConfig = field(default_factory=WikiConfig)

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
    parent: Capability | None = field(default=None, repr=False)
    # In-memory controller object, never a token, PID or clock supplied by Agent.
    session_control: object | None = field(default=None, repr=False)


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = False


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, result, *, retry_after: int | None = None, execution_exit_code: int | None = None):
        if status >= 400:
            owner = self.server.owner
            capability = getattr(self, "capability", None)
            binding = capability.journal if capability else None
            result = augment_error(
                result, route=self.path, request=getattr(self, "decoded_request", None),
                research_records=owner.journal_factory.research_records,
                supervisor_git=binding.service.supervisor_git if binding else True,
                parser=build_parser() if self.path == "/v1/gateway/execute" and status == 400 else None,
                catalog=owner.plugins.catalog() if self.path == "/v1/plugins/execute" and status == 400 else (),
            )
            from aka.light.supervisor.projection import credential_values, redact_diagnostics
            result = redact_diagnostics(result,
                private_paths=(str(owner.root), str(owner.audit_root or ""), str(ROOT),
                               str(owner.config.private_reference_dir or ""), owner.config.url,
                               str(owner.config.atrex_bench_root or "")),
                private_values=(*credential_values(owner.environment), getattr(self, "bearer", "")))
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            if execution_exit_code is not None:
                self.send_header("X-Atrex-Execution-Exit-Code", str(execution_exit_code))
            if getattr(self, "request_id", None):
                self.send_header("X-Request-ID", self.request_id)
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # A timed-out/disconnected Agent cannot receive the reply. This also
            # covers replies emitted from an exception handler during shutdown;
            # it must not replace the recorded outcome or start another response.
            return

    def do_GET(self):
        self.reply(200 if self.path == "/healthz" else 404,
                   {"status": "ok"} if self.path == "/healthz" else failure("Unknown endpoint"))

    def do_POST(self):
        owner = self.server.owner
        self.request_id = uuid.uuid4().hex
        deadline = time.monotonic() + owner.request_timeout_seconds
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        self.bearer = token
        with owner.lock:
            capability = owner.capabilities.get(token) if not owner.closed else None
        self.capability = capability
        if capability is None:
            self.reply(401, failure("Missing or revoked Session capability", repairable=False,
                                   code="session_unavailable", next_action="Ask the operator to resume or restart the Session. Do not change credentials or bypass the Runtime."))
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
                self.decoded_request = request
            # A complete authenticated request pauses only its Provider tree.
            # This handler/executor is outside that tree. Release before reply:
            # the stopped CLI cannot drain a large response. Disconnection does
            # not release this scope early or cancel an already submitted job.
            wait = capability.session_control.request() if capability.session_control else nullcontext()
            with wait:
                if owner.is_wiki_request(self.path, request):
                    result = owner.execute(capability, self.path, request)
                else:
                    with owner.request_lock(capability, deadline, self.request_id):
                        with owner.lock:
                            if owner.closed or owner.capabilities.get(token) is not capability:
                                raise SessionRevokedError("Session capability revoked")
                        result = owner.execute(capability, self.path, request)
            execution_exit_code = None
            if self.path == "/v1/gateway/execute" and isinstance(result.get("agent_result"), dict):
                # Never expose the private compatibility streams alongside the
                # canonical result. Initial reads and record-read share this object.
                execution_exit_code = result["exit_code"]
                result = result["agent_result"]
            self.reply(200, result, execution_exit_code=execution_exit_code)
        except AgentRequestError as error:
            self.reply(400, error.response)
        except RuntimeStateError as error:
            self.reply(503, error.response)
        except RequestValidationError as error:
            self.reply(400, failure(str(error)))
        except RequestDispatchTimeout as error:
            if not owner._live(capability):
                self.reply(401, failure("Session capability revoked", repairable=False, code="session_revoked"))
                return
            result = failure(str(error), code="request_not_started", next_action=(
                f"No job was submitted by this request. Wait at least {DISPATCH_RETRY_SECONDS} seconds, "
                "then retry the same request unchanged with backoff. Do not retry in a tight loop."))
            result["retry_after_seconds"] = DISPATCH_RETRY_SECONDS
            logging.getLogger(__name__).warning(
                "Supervisor Runtime request %s timed out before dispatch; safe to retry after backoff",
                self.request_id)
            self.reply(429, result, retry_after=DISPATCH_RETRY_SECONDS)
        except SessionRevokedError:
            self.reply(401, failure("Session capability revoked", repairable=False, code="session_revoked"))
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
    def __init__(self, config: RuntimeConfig, *, workspace_provider=None, services=None,
                 agent_resources=None):
        self.config = config
        from aka.light.runtime.services import builtin_services
        services = services if services is not None else builtin_services()
        self.executor, self.acceptance = services.executor, services.acceptance
        self.journal_factory = services.journal
        self.agent_resources = agent_resources
        if agent_resources is not None:
            from .core_resources import CoreResourceProjection
            self.plugins = CoreResourceProjection(agent_resources)
        else:
            from .plugins import PluginRegistry
            self.plugins = PluginRegistry(enabled=config.wiki_enabled)
        self.workspace_provider = workspace_provider
        self.environment = {k: v for k, v in os.environ.items() if k not in {OWNER_ENV, URL_ENV, TOKEN_ENV}}
        self.gateway_queue_wait_grace = configured_queue_wait_grace(self.environment)
        repetitions = str(config.measurement_repetitions)
        if repetitions not in {"1", "3"}:
            raise ValueError("measurement_repetitions must be 1 or 3")
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
        self.experiments = services.experiments.open((self.audit_root or self.root) / "measurements")
        self.measurements = self.experiments.store
        # The lock is private and survives restarts; never read an Agent-supplied lock.
        plugin_lock = (self.audit_root or self.root) / "plugins"
        if self.agent_resources is not None:
            self.plugins.check_lock()
        else:
            self.plugins.check_lock(plugin_lock)
        plugin_snapshot = self.plugins.snapshot()
        publish(plugin_lock, ".atrex_plugins/lock.json", json.dumps(plugin_snapshot).encode())
        self.wiki = WikiService((self.audit_root or self.root) / "wiki", config.wiki_options,
                                plugin_snapshot)
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
            capabilities = list(self.capabilities.values())
            self.capabilities.clear()
        for capability in capabilities:
            if capability.session_control:
                capability.session_control.kill()
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
            capability = self.capabilities.pop(token, None)
        if capability and capability.session_control:
            capability.session_control.kill()

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
                         branch: str, memory_version: int,
                         supervisor_git: bool = False):
        """Trusted controller binding, never populated from an Agent request or file."""
        workspace = workspace.resolve(strict=True)
        if (type(episode) is not int or episode < 1 or type(memory_version) is not int
                or memory_version < 1 or not isinstance(branch, str) or not branch
                or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", base_commit) is None):
            raise ValueError("Invalid controller Episode identity")
        path = self.episode_journal_path(episode)
        evidence = path.parent
        self.journal_factory.initialize(path, episode=episode, base_commit=base_commit,
                                        branch=branch, memory_version=memory_version)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            existing = self.journals.get(workspace)
            if existing is not None:
                if existing.service.path != path:
                    raise RuntimeError("Workspace already belongs to another Runtime Journal")
                if existing.service.supervisor_git != supervisor_git:
                    raise RuntimeError("Cannot change Git ownership during an Episode")
                return existing.view.prepare() if existing.view else workspace
            view = None
            agent_workspace = workspace
            if supervisor_git:
                from .hardware import hardware_vendor
                from aka.light.workspace.history import EpisodeHistory
                history = (EpisodeHistory(
                    episode=episode,
                    episodes_root=self.config.workspace.resolve() / ".atrex_long_horizon/episodes",
                ) if self.config.workspace else None)
                workspace_options = {
                    "is_ppu": hardware_vendor(self.config.hardware) == "ppu",
                    "research_records": self.journal_factory.research_records,
                    "plugin_registry": self.plugins,
                    "adaptive_root": (self.audit_root or self.root) / "adaptive-state",
                    "episode": episode,
                    "history": history,
                }
                if self.agent_resources is not None:
                    workspace_options["agent_resources"] = self.agent_resources
                view = (
                    self.workspace_provider.open_draft
                    if self.workspace_provider else EpisodeWorkspace
                )(workspace, evidence / "agent", **workspace_options)
                agent_workspace = view.prepare()
            journal_options = {
                "workspace": agent_workspace,
                "git_workspace": workspace,
                "campaign_root": self.journals_root,
                "evidence_root": evidence,
                "supervisor_git": supervisor_git,
            }
            if self.agent_resources is not None:
                journal_options["agent_resources"] = self.agent_resources
            binding = JournalBinding(self.journal_factory.open(**journal_options), view=view)
            self.journals[workspace] = binding
            self.journals[agent_workspace] = binding
            return agent_workspace

    @contextmanager
    def session(self, workspace: Path, environment: dict[str, str], *, session_control=None):
        workspace = workspace.resolve(strict=True)
        if not workspace.is_dir() or workspace == ROOT:
            raise ValueError("Runtime Session needs a separate Campaign workspace")
        token = secrets.token_urlsafe(32)
        context = {key: environment[key] for key in (
            "ATREX_TELEMETRY_TRACE", "ATREX_TELEMETRY_ATTEMPT_ID", "ATREX_TELEMETRY_CAMPAIGN_ID",
            "ATREX_TELEMETRY_ITERATION_ID", "ATREX_ENVIRONMENT_STATE_FILE",
        ) if key in environment}
        capability = Capability(workspace, context, session_control=session_control)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            capability.journal = self.journals.get(workspace)
        values = scrub_environment(environment)
        values.update({URL_ENV: self.url, TOKEN_ENV: token})
        if capability.journal and capability.journal.view:
            from .episode_workspace import EPISODE_WORKSPACE_ENV
            from aka.light.workspace.adaptive import PROTECTED_PATHS_ENV
            with capability.journal.lock:
                values[PROTECTED_PATHS_ENV] = json.dumps(capability.journal.view.prepare_session())
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
            return (not self.closed and any(value is capability for value in self.capabilities.values())
                    and (capability.parent is None or self._live(capability.parent)))

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

    def run_executor(self, command, cwd, environment, capability, *, deadline=None, request_id=None):
        from aka.light.execution.process import ExecutionControl
        deadline = deadline or capability.deadline or (time.monotonic() + self.request_timeout_seconds)
        return self.executor.run_gateway(command, cwd, environment, ExecutionControl(
            deadline, request_id or capability.request_id, lambda: self._live(capability),
            lambda: self._acquire(self.slots, capability, deadline), self.slots.release,
        ))

    def is_wiki_request(self, route, request):
        if self.agent_resources is not None:
            return False
        if route == "/v1/wiki/query":
            return True
        if route == "/v1/plugins/execute" and request.get("action") == "call":
            return any(request.get("tool") == f"{plugin.id}.{name}" and tool.get("runtime_tool") == "wiki-query"
                       for plugin in self.plugins.plugins for name, tool in plugin.tools.items())
        return False

    def execute_trusted(self, workspace: Path, argv: list[str], *, execution_timeout: int | None = None,
                        parent=None, reference_snapshot: Path | None = None) -> dict:
        """Controller-only execution/reuse; no Session, draft publication or HTTP switch."""
        capability = Capability(workspace.resolve(strict=True), {}, parent=parent)
        token = secrets.token_urlsafe(32)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            self.capabilities[token] = capability
        try:
            deadline = time.monotonic() + self.request_timeout_seconds
            if parent is not None and parent.deadline is not None:
                deadline = min(deadline, parent.deadline)
            with self.request_lock(capability, deadline, "acceptance"):
                return self.execute(capability, "/v1/gateway/execute", {"argv": argv}, reuse_completed=True,
                                    execution_timeout=execution_timeout, reference_snapshot=reference_snapshot)
        finally:
            self.revoke(token)

    def execute(self, capability: Capability, route: str, request: dict, *,
                reuse_completed=False, reuse_correctness=False, candidate_source: bytes | None = None,
                execution_timeout: int | None = None, reference_snapshot: Path | None = None) -> dict:
        if self.agent_resources is not None and route == "/v1/wiki/query":
            return self._execute_core_wiki_compat(capability, request)
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
                        return self.experiments.query(parsed, capability.workspace)
                    if parsed is not None and not self.config.ssh:
                        # Invalid operator overrides are known configuration
                        # failures, not Agent errors or unknown GPU outcomes.
                        # Reject before staging, task reservation or execution.
                        find_agate(environment)
                    if parsed is not None and parsed.kind != "env":
                        from aka.light.workspace.sources import POLICY, policy_at, unpack, snapshot as source_snapshot, install as install_source
                        authority = (capability.journal.view.worktree if capability.journal and capability.journal.view
                                     else self.config.workspace or capability.workspace)
                        source_policy = policy_at(authority, committed=True)
                        if candidate_source is not None:
                            if unpack(candidate_source)[0] != source_policy:
                                raise ValueError("Sealed candidate has a different source contract")
                            frozen_source = candidate_source
                        elif source_policy is not None:
                            frozen_source = source_snapshot(capability.workspace, policy=source_policy)
                        else:
                            if (capability.workspace / POLICY).exists() or (capability.workspace / POLICY).is_symlink():
                                raise ValueError("This single-file task cannot introduce a source-tree policy")
                            frozen_source = None
                        snapshot(capability.workspace, staged,
                                 custom_tools=bool(capability.journal and capability.journal.view))
                        if frozen_source is not None:
                            # The staged policy must come from controller state,
                            # never an editable copy in the Agent draft.
                            (staged / POLICY).unlink(missing_ok=True)
                            install_source(staged, frozen_source, replace=False)
                            if source_policy is not None:
                                # Remove obsolete editable files copied from the draft.
                                install_source(staged, frozen_source)
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
                                from aka.light.workspace.sources import snapshot as source_snapshot
                                source = source_snapshot(staged)
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
                        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(reference_snapshot or self.config.private_reference_dir)
                    # Keep the legacy report compiler's log alongside private records.
                    (staged / ".atrex_long_horizon").mkdir()
                    (staged / ".atrex_long_horizon/journal.json").write_text("{}")
                    if parsed is not None:
                        for value in ([] if parsed.no_sync else parsed.sync or ["profiles"]):
                            if relative_path(value).parts[0] not in {"profiles", "scratch"}:
                                raise ValueError("--sync must be inside profiles/ or scratch/")
                    command = self.config.command(staged) + argv
                    if execution_timeout is not None:
                        if type(execution_timeout) is not int or not 1 <= execution_timeout <= 600:
                            raise ValueError("Trusted execution timeout must be 1–600 seconds")
                        command[command.index("--timeout") + 1] = str(execution_timeout)
                    kind = "gateway"
            if kind == "gateway" and parsed is not None and parsed.kind != "env" and not parsed.dry_run:
                from aka.light.experiment.provider import MeasurementEffects
                # Parse the Supervisor's authority flags too, so the exact target,
                # execution timeout and transport enter the task identity.
                effective = parse_gateway(command[2:])
                effects = MeasurementEffects(
                    mode=self.config.optimization_mode, repetitions=self.measurement_repetitions,
                    private_reference_dir=self.config.private_reference_dir,
                    private_paths=(str(self.root), str(ROOT), str(self.config.private_reference_dir or ""),
                                   self.config.url, str(self.config.atrex_bench_root or "")),
                    run=lambda command, cwd, env: self.run_executor(command, cwd, env, capability),
                    audit=lambda argv, process, record: self.audit_process(capability, "gateway", argv, process, record),
                    publish_log=lambda staged: self.publish_evaluation_log(capability.workspace, staged),
                    publish_outputs=lambda staged, args: self.publish_legacy_outputs(capability.workspace, staged, args),
                )
                return self.experiments.execute(staged, effective, argv, environment, command, effects=effects,
                                               reuse_completed=reuse_completed, reuse_correctness=reuse_correctness)
            # From this point execution may have submitted a job. Failures must
            # never be classified as repairable argument errors by the handler.
            if kind == "wiki":
                return self.wiki.execute(self, capability, command, staged, environment, argv)
            process = self.run_executor(command, staged, environment, capability)
            self.audit_process(capability, kind, argv, process)
            if kind == "gateway":
                self.publish_evaluation_log(capability.workspace, staged)
                if process.returncode == 0:
                    self.publish_legacy_outputs(capability.workspace, staged, parsed)
            from aka.light.supervisor.projection import credential_values, project_response
            response = project_response(process, generalized=self.config.private_reference_dir is not None,
                                    operation=parsed.kind if kind == "gateway" and parsed is not None else None,
                                    private_values=credential_values(environment),
                                    wiki=kind == "wiki", private_paths=(str(staged), str(self.root),
                                    str(self.config.private_reference_dir or ""), str(ROOT),
                                    self.config.url, str(self.config.atrex_bench_root or "")))
            if kind == "gateway" and parsed is not None and parsed.kind == "env":
                from aka.light.evaluation.public_result import seal_response
                return seal_response(response, {"operation": "env"})
            return response

    def _validate_candidate_correctness(self, capability: Capability, source: bytes) -> dict:
        validator = getattr(self, "numerical_validator", None)

        def ensure_live():
            if not self._live(capability):
                raise SessionRevokedError("Session revoked during report acceptance")

        context = CandidateValidation(
            atrex_bench=self.config.atrex_bench_root is not None,
            production=self.config.optimization_mode == "production",
            measure=lambda argv, frozen, correctness: self.execute(
                capability, "/v1/gateway/execute", {"argv": argv},
                reuse_completed=True, reuse_correctness=correctness, candidate_source=frozen),
            read_record=self.measurements.read, ensure_live=ensure_live,
            research_records=self.journal_factory.research_records,
            supplemental=(lambda frozen: validator(
                capability.journal.service.git_workspace, frozen, parent=capability
            )) if validator else None,
        )
        return self.acceptance.validate_candidate(context, source)

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
                evidence_hint = (
                    "Use list/load-direction or list/load-experiment for IDs, and record-read for measurement evidence. "
                    if self.journal_factory.research_records else
                    "Use record-read to check gateway_record_ids; candidate_ready must cite a passing full Evaluate for current kernel.py. "
                )
                raise AgentRequestError(str(error), next_action=(
                    "Correct the indicated field or lifecycle state, then retry. Read skills/runtime-records/references/journal.md. "
                    + evidence_hint +
                    "A rejected episode-report is not a completed Episode; fix it and submit again.")) from error
            response = {"exit_code": 0, "stdout": json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", "stderr": ""}
            self.audit_process(capability, "journal", [str(request.get("operation"))],
                               subprocess.CompletedProcess([], 0, response["stdout"], ""))
            return response
        finally:
            binding.lock.release()

    def audit_process(self, capability, kind, argv, process, record_id=None):
        if self.audit_root:
            audited_argv = list(argv)
            if kind == "gateway":
                audited_argv = []
                redact_next = False
                for value in argv:
                    if redact_next:
                        audited_argv.append("<redacted>")
                        redact_next = False
                        continue
                    option = value.split("=", 1)[0]
                    if option in {"--env", "--ssh-init", "--url"}:
                        audited_argv.append(option + "=<redacted>" if "=" in value else option)
                        redact_next = "=" not in value
                    else:
                        audited_argv.append(value)
            publish(self.audit_root, f"request-{capability.request_id or uuid.uuid4().hex}.json", json.dumps({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": kind, "argv": audited_argv, "exit_code": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr,
                "gateway_record_id": record_id,
                "capture_limits": {"stdout_bytes": 4 * 1024 * 1024, "stderr_bytes": 128 * 1024},
            }, ensure_ascii=False).encode())

    def execute_plugin(self, capability: Capability, request: dict) -> dict:
        with _validating_request():
            if request == {"action": "list"}:
                return {"exit_code": 0, "stdout": json.dumps(self.plugins.public_catalog()) + "\n", "stderr": ""}
            if set(request) != {"action", "tool", "input"} or request["action"] != "call":
                raise ValueError("Use action=list, or action=call with tool and input")
            if not isinstance(request["tool"], str):
                raise ValueError("tool must be a catalog name")
            tool = self.plugins.validate_call(request["tool"], request["input"])
        if self.agent_resources is not None:
            result = self.plugins.call_result(
                request["tool"], request["input"], capability.workspace,
                cwd=capability.workspace,
                executor=lambda entrypoint, value, context, timeout, **kwargs: self._run_core_entry(
                    capability, request["tool"], entrypoint, value, context, timeout,
                    cwd=kwargs.get("cwd") or capability.workspace,
                    env_extra={
                        **(context.get("environment", {}) if isinstance(context, dict) else {}),
                        **(kwargs.get("env_extra") or {}),
                    },
                    python_paths=kwargs.get("python_paths", ()),
                ),
            )
            receipt = result.get("receipt", {})
            status = receipt.get("status") if isinstance(receipt, dict) else None
            exit_code = 0 if status == "ok" else (75 if status in {"tool_cancelled", "tool_timeout"} else 1)
            return {"exit_code": exit_code,
                    "stdout": json.dumps(result, ensure_ascii=False) + "\n", "stderr": ""}
        if tool.get("runtime_tool") == "wiki-query":
            # Reuse the existing scoped, bounded Wiki executor and private audit.
            value = request["input"]
            argv = [value["request"]]
            for name in ("max_records", "max_bytes", "exclude"):
                if name in value:
                    argv += ["--" + name.replace("_", "-"), str(value[name])]
            response = self.execute(capability, "/v1/wiki/query", {"tool": "query_nl", "argv": argv})
            # The schema describes successful tool output, not executor failures.
            # Validate after dispatch (outside _validating_request): drift is a
            # Supervisor failure, not an Agent-repairable input error.
            if response["exit_code"] == 0:
                # The Wiki projection may already have truncated an oversized
                # stdout. Classify the size before trying to decode that prefix.
                if len(response["stdout"].encode()) > MAX_OUTPUT_BYTES:
                    raise_output_error(request["tool"], "result_too_large")
                project_plugin_output(request["tool"], tool["output_schema"], response["stdout"])
            return response
        with tempfile.TemporaryDirectory(dir=self.root, prefix="plugin-") as temporary:
            staged = Path(temporary)
            with _validating_request():
                snapshot(capability.workspace, staged,
                         custom_tools=bool(capability.journal and capability.journal.view))
            # Do not honor paths, environment or executable declarations from the Agent.
            invocation = dict(request, snapshot=self.plugins.snapshot())
            request_path = staged / ".plugin-request.json"
            request_path.write_text(json.dumps(invocation))
            environment = scrub_environment(self.environment)
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
                check_executor_output_failure(request["tool"], process.returncode, process.stdout)
                # Plugin errors can contain private resources or evaluator paths.
                raise RuntimeError("Plugin execution failed; inspect private request diagnostics")
            rendered = project_plugin_output(request["tool"], tool["output_schema"], process.stdout)
            return {"exit_code": 0, "stdout": rendered, "stderr": ""}

    def _run_core_entry(self, capability, tool_name, entrypoint, value, context, timeout, *,
                        env_extra=None, cwd=None, python_paths=()):
        """Run one Core tool stage inside Light's cancellable process group."""
        from aka.resources import ResourcesError

        call_id = context.get("call_id", "") if isinstance(context, dict) else ""
        stage = "extract" if isinstance(value, dict) and "result" in value else "tool"
        now = time.monotonic()
        light_deadline = capability.deadline or (now + self.request_timeout_seconds)
        deadline = min(light_deadline, now + float(timeout))
        if not self._live(capability):
            raise ResourcesError("tool_cancelled", "Light Session was cancelled")
        if deadline <= now:
            raise ResourcesError("tool_timeout", "Core tool exceeded its execution deadline")

        with tempfile.TemporaryDirectory(dir=self.root, prefix="core-tool-") as temporary:
            staged = Path(temporary)
            request_path = staged / "request.json"
            paths = [str(path) for path in python_paths if isinstance(path, (str, Path)) and str(path)]
            publish(staged, "request.json", json.dumps({
                "entrypoint": entrypoint, "input": value, "context": context,
                "python_paths": list(dict.fromkeys(paths)),
            }, ensure_ascii=False, allow_nan=False).encode())
            environment = scrub_environment(self.environment)
            environment.update(capability.context)
            if env_extra:
                environment.update(env_extra)
            # Ignore cwd, PYTHONPATH, user site configuration and sitecustomize
            # from the Agent workspace. Only the binding's frozen launch paths
            # and trusted host paths are inserted by the private shim.
            environment.pop("PYTHONPATH", None)
            environment["PYTHONIOENCODING"] = "utf-8"
            runner = (
                "import json,runpy,sys; "
                "request=json.load(open(sys.argv[1], encoding='utf-8')); "
                "sys.path[:0]=request['python_paths']; "
                "sys.stdin=open(sys.argv[1], encoding='utf-8'); "
                "runpy.run_module('aka.resources.runner', run_name='__main__')"
            )
            try:
                process = self.run_executor(
                    [sys.executable, "-I", "-c", runner, str(request_path)],
                    Path(cwd or capability.workspace), environment, capability,
                    deadline=deadline,
                    request_id=f"{call_id}:{stage}" if call_id else f"{tool_name}:{stage}",
                )
            except SessionRevokedError as error:
                raise ResourcesError("tool_cancelled", "Light Session was cancelled") from error
            except RequestDispatchTimeout as error:
                if not self._live(capability):
                    raise ResourcesError("tool_cancelled", "Light Session was cancelled") from error
                raise ResourcesError("tool_timeout", "Core tool did not start before its deadline") from error
            except GatewayExecutionUnavailable as error:
                if not self._live(capability):
                    raise ResourcesError("tool_cancelled", "Light Session was cancelled") from error
                if time.monotonic() >= deadline:
                    raise ResourcesError("tool_timeout", "Core tool exceeded its execution deadline") from error
                raise ResourcesError("tool_failed", "Core tool execution was interrupted") from error
            except Exception as error:
                raise ResourcesError("tool_failed", "Core tool execution failed") from error
            self.audit_process(capability, "core-tool", [tool_name, entrypoint, stage], process)
            if process.returncode:
                raise ResourcesError("tool_failed", "Core tool entrypoint failed")
            try:
                result = json.loads(process.stdout)
            except (TypeError, ValueError) as error:
                raise ResourcesError("invalid_output", "Core tool entrypoint returned invalid JSON") from error
            if not self._live(capability):
                raise ResourcesError("tool_cancelled", "Light Session was cancelled")
            return result, ""

    def _execute_core_wiki_compat(self, capability, request):
        """Preserve query_nl's raw CLI projection without opening legacy stores."""
        with _validating_request():
            if set(request) != {"tool", "argv"}:
                raise ValueError("Wiki request must contain tool and argv")
            if request.get("tool") != "query_nl":
                raise AgentRequestError(
                    "This legacy Wiki command is not available in the selected Core profile.",
                    code="tool_not_found", repairable=False,
                    next_action="Use python3 tools/plugin.py list and call an enabled Core Wiki tool.",
                )
            argv = validated_argv(request["argv"])
            values: dict[str, object] = {}
            query_parts = []
            query_file = None
            index = 0
            while index < len(argv):
                item = argv[index]
                if item == "--":
                    raise ValueError("Wiki requests do not accept a '--' option terminator")
                option, equals, inline = item.partition("=")
                key = {"--max-records": "max_records", "--max-bytes": "max_bytes",
                       "--exclude": "exclude"}.get(option)
                if key is not None:
                    if equals:
                        raw = inline
                    else:
                        index += 1
                        raw = argv[index] if index < len(argv) else ""
                    if not raw:
                        raise ValueError(f"{option} requires a value")
                    values[key] = int(raw) if key in {"max_records", "max_bytes"} else raw
                elif option == "--file":
                    if equals:
                        raw = inline
                    else:
                        index += 1
                        raw = argv[index] if index < len(argv) else ""
                    if not raw or query_file is not None:
                        raise ValueError("--file requires one workspace-relative file")
                    query_file = read_input(capability.workspace, raw, limit=128 * 1024).decode("utf-8")
                elif option == "--brief" and not equals:
                    pass
                elif item.startswith("-"):
                    raise ValueError(f"Unsupported option for Core Wiki compatibility: {option}")
                else:
                    query_parts.append(item)
                index += 1
            if query_file is not None and query_parts:
                raise ValueError("Use either query text or --file, not both")
            query = query_file if query_file is not None else " ".join(query_parts)
            if not query:
                raise ValueError("Wiki query text is required")
            values["request"] = query

        tool_name = "gpu-wiki.query"
        if not any(row.get("name") == tool_name for row in self.plugins.catalog()):
            raise AgentRequestError(
                "The selected Core profile does not enable the public GPU Wiki.",
                code="tool_not_found", repairable=False,
                next_action="Use python3 tools/plugin.py list and call an enabled Core Wiki tool.",
            )
        self.plugins.validate_call(tool_name, values)
        response = self.execute_plugin(capability, {"action": "call", "tool": tool_name, "input": values})
        result = json.loads(response["stdout"])
        receipt = result.get("receipt", {})
        if receipt.get("status") != "ok":
            diagnostics = result.get("diagnostics", [])
            message = diagnostics[0].get("message", "Core Wiki query failed") if diagnostics else "Core Wiki query failed"
            status = receipt.get("status")
            return {"exit_code": 75 if status in {"tool_cancelled", "tool_timeout"} else 1,
                    "stdout": "", "stderr": str(message)[:2000]}
        output = json.dumps(result.get("data"), ensure_ascii=False) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            return {"exit_code": 1, "stdout": "", "stderr": "Core Wiki result exceeds the Light output limit"}
        return {"exit_code": 0, "stdout": output, "stderr": ""}

    def _wiki_command(self, capability, tool, argv, staged, environment):
        if not self.config.wiki_enabled:
            raise AgentRequestError("GPU Wiki is disabled in the selected Workflow configuration",
                                    code="wiki_disabled", repairable=False,
                                    next_action="Continue with the enabled tools. Only the operator can enable Wiki; do not retry or access its private store directly.")
        if not isinstance(tool, str) or tool not in WIKI_STORES:
            raise ValueError("Unsupported Wiki tool")
        if "--" in argv:
            raise ValueError("Wiki requests do not accept a '--' option terminator")
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
        if tool == "query_nl":
            # No Agent-controlled deadline or bridge model/provider override.
            argv = filter_options(argv, frozenset({"--timeout", "--agent-cli"}), frozenset())
            argv += ["--timeout", str(self.config.wiki_options.bridge_timeout)]
            if self.config.wiki_options.model or self.config.wiki_options.settings_file:
                environment["ATREX_WIKI_BRIDGE_CLI"] = "claude"
        if self.config.wiki_profile_root:
            environment["ATREX_WIKI_PROFILE_ROOT"] = str(self.config.wiki_profile_root)
        environment["ATREX_WIKI_TASK_ID"] = self.config.task_id
        return [sys.executable, str(legacy_resource_root() / "gpu-wiki/tools" / f"{tool}.py"), *argv]

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
            from aka.light.orchestrator.session_tail import read_regular_bytes
            data = read_regular_bytes(log, limit=MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Evaluation log exceeds the size limit")
            binding = self.journals.get(workspace)
            destination = binding.service.git_workspace if binding and binding.view else workspace
            publish(destination, EPISODE_EVALUATIONS_PATH,
                    data, append=True)


@contextmanager
def session_environment(workspace: Path, environment: dict[str, str], *, session_control=None):
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
    with runtime.session(workspace, environment, session_control=session_control) as values:
        yield values
