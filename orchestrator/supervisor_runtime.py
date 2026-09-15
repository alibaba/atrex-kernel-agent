"""Supervisor-owned HTTP Runtime for untrusted AKA Agent sessions.

The service does not replace AKA's campaign controller.  It owns privileged
Gateway/Wiki effects and the authoritative Direction/Experiment Journal.  A
short-lived bearer capability binds every request to one exact Agent workspace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from long_horizon.promotion_audit import AUDIT_FILENAME_RE
from supervisor.errors import (
    UNKNOWN_OUTCOME,
    AgentRequestError,
    RuntimeStateError,
    error_response,
)

from .agent_assets import DEFAULT_AGENT_SKILLS, resolve_agent_skills
from .agent_workspace import AgentWorkspace
from .constants import ATREX_BENCH_RUNTIME_ENV
from .sandbox_launch import SandboxLaunch

if TYPE_CHECKING:
    from .session_capture import SessionCapture

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_GATEWAY_RESULT_BYTES = 512 * 1024
MAX_DEV_STDOUT_BYTES = 64 * 1024
MAX_WIKI_RESULT_BYTES = 256 * 1024
MAX_WIKI_PAYLOAD_BYTES = 128 * 1024
MAX_STDERR_BYTES = 32 * 1024
RUNTIME_URL_ENV = "ATREX_AKA_RUNTIME_URL"
RUNTIME_TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
WIKI_PROFILE_ROOT_ENV = "ATREX_WIKI_PROFILE_ROOT"
SUPERVISOR_EVIDENCE_ROOT_ENV = "ATREX_AKA_SUPERVISOR_EVIDENCE_ROOT"
SUPERVISOR_HISTORY_ROOT_ENV = "ATREX_AKA_SUPERVISOR_HISTORY_ROOT"
_RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        ATREX_BENCH_RUNTIME_ENV,
        RUNTIME_URL_ENV,
        RUNTIME_TOKEN_ENV,
        SUPERVISOR_EVIDENCE_ROOT_ENV,
        SUPERVISOR_HISTORY_ROOT_ENV,
        "ATREX_AKA_MODE_STATE_FILE",
        "ATREX_AKA_REUSE_GATEWAY_RESULTS",
        "ATREX_AKA_INTERNAL_MEASUREMENT",
        "ATREX_AKA_COMPARISON_RUN_TIMEOUT",
    }
)
_REQUEST_CONTEXT_ENVIRONMENT_KEYS = frozenset(
    {
        "ATREX_ENVIRONMENT_STATE_FILE",
        "ATREX_TELEMETRY_ATTEMPT_ID",
        "ATREX_TELEMETRY_CAMPAIGN_ID",
        "ATREX_TELEMETRY_ITERATION_ID",
        "ATREX_TELEMETRY_TRACE",
        "ATREX_WIKI_TASK_ID",
    }
)
_SENSITIVE_AGENT_PREFIXES = (
    "AGATE_",
    "ATREX_GPU_WIKI_",
    "ATREX_WIKI_",
    "GPU_WIKI_",
)
_SENSITIVE_AGENT_KEYS = frozenset(
    {
        "ATREX_GATEWAY_CAPABILITY",
        "ATREX_PRIVATE_REFERENCE_DIR",
        "ATREX_WIKI_CAPABILITY",
        "ATREX_WIKI_BEARER_TOKEN",
        "GPU_WIKI_BEARER_TOKEN",
    }
)


@dataclass(frozen=True, slots=True)
class SupervisorRuntimeConfig:
    """Campaign-owned endpoint and Agent isolation policy."""

    repository_root: Path
    hardware: str
    sandbox_timeout: int
    sandbox_url: str = ""
    sandbox_profile: str = ""
    sandbox_ssh: str = ""
    sandbox_ssh_init: str = ""
    sandbox_ssh_gpu: int | None = None
    sandbox_health_command: str = ""
    sandbox_ssh_runtime_binds: tuple[str, ...] = ()
    private_reference_dir: Path | None = None
    atrex_bench_root: Path | None = None
    agent_sandbox: str = "auto"
    bwrap_executable: str = "bwrap"
    agent_skills: tuple[str, ...] = DEFAULT_AGENT_SKILLS
    agent_reference_projects: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_skills", resolve_agent_skills(self.agent_skills))
        if self.agent_sandbox not in {"auto", "bwrap", "none"}:
            raise ValueError("agent_sandbox must be auto, bwrap, or none")
        if not self.hardware:
            raise ValueError("Supervisor Runtime requires a hardware target")
        if sum(bool(value) for value in (
            self.sandbox_url,
            self.sandbox_profile,
            self.sandbox_ssh,
        )) > 1:
            raise ValueError("Supervisor Runtime sandbox endpoints are mutually exclusive")


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    token: str
    workspace: Path
    campaign_root: Path
    backend: str
    issued_at: float
    evidence_root: Path
    request_environment: tuple[tuple[str, str], ...] = ()
    worktree: Path | None = None

    @property
    def wiki_profile_root(self) -> Path:
        # evidence lives at <campaign>/workspaces/<scope>/evidence. Capture
        # authority from this issued scope, not a later Agent-edited Git path.
        return self.evidence_root.parents[2] / "wiki-profile"


@dataclass(frozen=True, slots=True)
class RuntimeSessionLease:
    """One process invocation's Runtime authority and optional bwrap command."""

    runtime: SupervisorRuntime
    token: str
    launch: SandboxLaunch
    capture: SessionCapture

    @property
    def command(self) -> tuple[str, ...]:
        return tuple(self.launch.command)

    @property
    def environment(self) -> dict[str, str]:
        return self.launch.environment

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return self.launch.pass_fds

    def close_launch_fds(self) -> None:
        self.launch.close()

    def close(self) -> None:
        self.runtime.revoke(self.token)


class _RuntimeHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, owner: SupervisorRuntime) -> None:
        self.owner = owner
        super().__init__(("127.0.0.1", 0), _RuntimeRequestHandler)


class _RuntimeRequestHandler(BaseHTTPRequestHandler):
    server: _RuntimeHttpServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: int, value: Mapping[str, object]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, code: str, message: str, **details: object) -> None:
        self._json(
            status,
            error_response(message, code=code, **details),
        )

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
            return
        self._json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path not in {
            "/v1/gateway/execute",
            "/v1/wiki/query",
            "/v1/journal/execute",
        }:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
            return
        raw_length = self.headers.get("content-length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "content-length must be an integer",
            )
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"request body exceeds {MAX_REQUEST_BYTES} bytes",
            )
            return
        authorization = self.headers.get("authorization", "")
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        capability = self.server.owner.authorize(token)
        if capability is None:
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "invalid_capability",
                "the scoped Runtime capability is missing, expired, or revoked",
                repairable=False,
                next_action=(
                    "This session cannot use Runtime tools. Report the blocker so the operator "
                    "can restore the session; do not change credentials or bypass the Runtime."
                ),
            )
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be an object")
            if self.path == "/v1/gateway/execute":
                response = self.server.owner.execute_gateway(capability, request)
            elif self.path == "/v1/journal/execute":
                response = self.server.owner.execute_journal(capability, request)
            else:
                response = self.server.owner.execute_wiki(capability, request)
        except AgentRequestError as exc:
            self._json(HTTPStatus.BAD_REQUEST, exc.response)
            return
        except RuntimeStateError as exc:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, exc.response)
            return
        except ValueError as exc:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(exc),
            )
            return
        except Exception:  # trusted boundary: do not disclose a traceback
            error_id = f"rterr_{uuid.uuid4().hex}"
            logging.getLogger(__name__).exception("Runtime request failed: %s", error_id)
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "runtime_failure",
                "Supervisor Runtime could not return a confirmed outcome.",
                repairable=False,
                next_action=UNKNOWN_OUTCOME,
                error_id=error_id,
            )
            return
        self._json(HTTPStatus.OK, response)


def _bounded_output(value: str, limit: int) -> tuple[str, int]:
    """Keep useful output from both ends and report omitted UTF-8 bytes."""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value, 0
    marker = b"\n... <Supervisor Runtime omitted middle output> ...\n"
    budget = max(0, limit - len(marker))
    head_size = budget // 3
    tail_size = budget - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    kept = head + marker.decode() + tail
    return kept, len(encoded) - len(kept.encode("utf-8"))


def _command_result(
    process: subprocess.CompletedProcess[str],
    *,
    stdout_limit: int,
) -> dict[str, object]:
    stdout, omitted_stdout = _bounded_output(process.stdout or "", stdout_limit)
    stderr, omitted_stderr = _bounded_output(process.stderr or "", MAX_STDERR_BYTES)
    result: dict[str, object] = {
        "exit_code": int(process.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }
    if omitted_stdout or omitted_stderr:
        result["truncated"] = {
            "stdout_bytes_omitted": omitted_stdout,
            "stderr_bytes_omitted": omitted_stderr,
        }
    return result


def _option_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _has_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in argv)


def _with_wiki_projection_defaults(tool: str, argv: list[str]) -> list[str]:
    """Enforce a compact public Wiki view without changing query semantics."""
    projected = list(argv)
    if tool not in {"query_nl", "query_wiki"}:
        return projected
    if not _has_option(projected, "--brief"):
        projected.append("--brief")
    max_bytes = _option_value(projected, "--max-bytes")
    if max_bytes is None:
        projected.extend(["--max-bytes", str(MAX_WIKI_PAYLOAD_BYTES)])
    else:
        try:
            requested = int(max_bytes)
        except ValueError:
            return projected
        if requested > MAX_WIKI_PAYLOAD_BYTES:
            for index, value in enumerate(projected):
                if value.startswith("--max-bytes="):
                    projected[index] = f"--max-bytes={MAX_WIKI_PAYLOAD_BYTES}"
                    break
                if value == "--max-bytes" and index + 1 < len(projected):
                    projected[index + 1] = str(MAX_WIKI_PAYLOAD_BYTES)
                    break
    return projected


def _gateway_output_limit(argv: list[str]) -> int:
    """Give arbitrary Agent-owned Dev output a tighter context bound."""
    kind = (_option_value(argv, "--kind") or "auto").casefold()
    if kind == "dev":
        return MAX_DEV_STDOUT_BYTES
    if kind in {
        "run",
        "profile",
        "check",
        "disassemble",
        "env",
        "record-read",
    }:
        return MAX_GATEWAY_RESULT_BYTES
    command = argv[argv.index("--") + 1 :] if "--" in argv else []
    names = {Path(value).name for value in command}
    if "test_kernel.py" in names or names.intersection(
        {"profile_nvidia.sh", "profile_kernel.sh"}
    ):
        return MAX_GATEWAY_RESULT_BYTES
    return MAX_DEV_STDOUT_BYTES


def _project_dry_run_stdout(stdout: str) -> str:
    """Remove Supervisor authority and host paths from gateway dry-run output."""
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    if not isinstance(raw, dict):
        return stdout
    visible = {
        key: raw[key]
        for key in (
            "hardware",
            "kind",
            "num_gpus",
            "fallback_kind",
            "candidate_bytes",
            "shape_count",
            "shape_batch_count",
            "mode",
            "options",
            "sync",
        )
        if key in raw
    }
    return json.dumps(visible, ensure_ascii=False, separators=(",", ":")) + "\n"


def _validated_argv(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("argv must be a non-empty string array")
    if any(not isinstance(item, str) or "\x00" in item for item in value):
        raise ValueError("argv must contain only NUL-free strings")
    if len(value) > 512:
        raise ValueError("argv contains too many entries")
    return list(value)


_CANONICAL_GATEWAY_VALUE_OPTIONS = frozenset(
    {
        "--workspace",
        "--hardware",
        "--url",
        "--gateway-profile",
        "--ssh",
        "--ssh-init",
        "--ssh-gpu",
        "--ssh-runtime-bind",
        "--health-command",
        "--timeout",
    }
)


def _reject_abbreviated_options(argv: list[str], checked_options: frozenset[str] | set[str]) -> None:
    """Fail before dispatch if a prefix could bypass a textual policy check.

    Downstream parsers also disable abbreviations. The command after ``--`` is
    opaque Dev input, not Supervisor options.
    """
    for value in argv:
        if value == "--":
            break
        option = value.split("=", 1)[0]
        if not option.startswith("--") or option in checked_options:
            continue
        if any(known.startswith(option) for known in checked_options):
            raise AgentRequestError(
                f"Abbreviated option {option!r} is not allowed.",
                code="invalid_arguments",
                next_action=(
                    "Use complete option names for allowed inputs; remove Supervisor-owned "
                    "endpoint, workspace, hardware, timeout, and Wiki store overrides. "
                    "For a Wiki query file, use --file with a workspace-relative path."
                ),
            )


def _without_endpoint_overrides(argv: list[str]) -> list[str]:
    """Remove Agent-supplied authority fields before applying Campaign policy."""
    _reject_abbreviated_options(argv, _CANONICAL_GATEWAY_VALUE_OPTIONS)
    output: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--":
            output.extend(argv[index:])
            break
        option = value.split("=", 1)[0]
        if option in _CANONICAL_GATEWAY_VALUE_OPTIONS:
            if "=" not in value:
                index += 1
                if index >= len(argv) or argv[index].startswith("--"):
                    raise ValueError(f"{option} requires a value")
            index += 1
            continue
        output.append(value)
        index += 1
    return output


def _translate_visible_path(value: str, workspace: Path) -> str:
    visible = "/home/agent/workspace"
    if value == visible:
        return str(workspace)
    if value.startswith(visible + "/"):
        return str(workspace / value[len(visible) + 1 :])
    return value


class SupervisorRuntime:
    """One Campaign-scoped trusted service and its Session capabilities."""

    def __init__(
        self,
        config: SupervisorRuntimeConfig,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._environment = dict(os.environ if environment is None else environment)
        for key in _RUNTIME_ENVIRONMENT_KEYS:
            self._environment.pop(key, None)
        self._environment.pop(WIKI_PROFILE_ROOT_ENV, None)
        self._lock = threading.RLock()
        # The global lock protects maps only. Slow Journal/Git/publication work
        # is serialized per private workspace, including tokens sharing a scope.
        self._journal_locks: dict[Path, threading.RLock] = {}
        self._audit_locks: dict[Path, threading.RLock] = {}
        self._capabilities: dict[str, RuntimeCapability] = {}
        self._launches: dict[str, SandboxLaunch] = {}
        self._workspace_views: dict[str, AgentWorkspace] = {}
        self._server: _RuntimeHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="aka-supervisor-runtime-")
        self.state_root = Path(self._temporary.name)
        self.atrex_bench_runtime = self._materialize_atrex_bench_runtime(
            self.config.atrex_bench_root
        )
        self._request_slots = threading.BoundedSemaphore(16)

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Supervisor Runtime is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> SupervisorRuntime:
        if self._server is not None:
            return self
        self._server = _RuntimeHttpServer(self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aka-supervisor-runtime",
            daemon=True,
        )
        self._thread.start()
        print(f"[orchestrator] Supervisor Runtime listening at {self.url}", flush=True)
        return self

    def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            self._capabilities.clear()
            for launch in self._launches.values():
                launch.close()
            self._launches.clear()
        self._temporary.cleanup()

    def __enter__(self) -> SupervisorRuntime:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def authorize(self, token: str) -> RuntimeCapability | None:
        if not token:
            return None
        with self._lock:
            return self._capabilities.get(token)

    def revoke(self, token: str) -> None:
        with self._lock:
            capability = self._capabilities.pop(token, None)
            launch = self._launches.pop(token, None)
            view = self._workspace_views.pop(token, None)
        if launch is not None:
            launch.close()
        if view is not None and capability is not None:
            with self._scope_lock(capability):
                view.publish()

    def _scope_lock(self, capability: RuntimeCapability, *, audit: bool = False):
        key = capability.evidence_root.resolve()
        with self._lock:
            locks = self._audit_locks if audit else self._journal_locks
            return locks.setdefault(key, threading.RLock())

    def prepare_session(
        self,
        command: list[str],
        workspace: Path,
        environment: Mapping[str, str],
    ) -> RuntimeSessionLease:
        """Issue authority, hide external credentials, and optionally apply bwrap."""
        workspace = workspace.resolve()
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("Agent workspace must be a real directory")
        campaign_root, scope_root = self._private_scope_for(workspace)
        evidence_root = scope_root / "evidence"
        gateway_records = evidence_root / "gateway-records"
        if gateway_records.is_symlink():
            raise ValueError("Gateway record directory cannot be a symlink")
        gateway_records.mkdir(parents=True, mode=0o700, exist_ok=True)
        response_records = evidence_root / "runtime-responses"
        if response_records.is_symlink():
            raise ValueError("Runtime response directory cannot be a symlink")
        response_records.mkdir(parents=True, mode=0o700, exist_ok=True)
        for name in ("evaluations.jsonl", "runtime-requests.jsonl"):
            path = evidence_root / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Runtime record path is unsafe: {path}")
            path.touch(mode=0o600, exist_ok=True)
        # Provider state shares the invisible per-worktree scope but remains
        # separate from evaluator evidence.
        provider_homes = scope_root / "provider-sessions"
        if provider_homes.is_symlink():
            raise ValueError("Provider Session directory cannot be a symlink")
        provider_homes.mkdir(parents=True, mode=0o700, exist_ok=True)
        backend = str(environment.get("ATREX_AGENT_CLI") or "")
        token = secrets.token_urlsafe(32)
        from .agent_sandbox import (
            _enabled,
            _prepare_provider_home,
            _projected_backends,
            _session_key,
            wrap_agent_command,
        )
        from .session_capture import SessionCapture

        view = None
        provider_home = None
        if _enabled(self.config.agent_sandbox, self.config.bwrap_executable):
            from .agent_workspace import WORKSPACE_ROLE_ENV

            view = AgentWorkspace(
                workspace, scope_root,
                role=str(environment.get(WORKSPACE_ROLE_ENV) or "optimizer"),
            )
            agent_workspace = view.prepare()
            provider_home = _prepare_provider_home(
                provider_homes, _session_key(command, backend, environment),
                _projected_backends(backend, environment),
                Path(environment.get("HOME") or str(Path.home())).resolve(),
            )
        else:
            agent_workspace = workspace
        request_environment = tuple(
            (key, str(environment[key]))
            for key in sorted(_REQUEST_CONTEXT_ENVIRONMENT_KEYS)
            if environment.get(key)
        )
        capability = RuntimeCapability(
            token,
            agent_workspace,
            campaign_root,
            backend,
            time.time(),
            evidence_root,
            request_environment,
            workspace,
        )
        with self._lock:
            self._capabilities[token] = capability
            if view is not None:
                self._workspace_views[token] = view

        exact_environment = {
            str(key): str(value)
            for key, value in environment.items()
            if not key.startswith(_SENSITIVE_AGENT_PREFIXES)
            and key not in _SENSITIVE_AGENT_KEYS
            and key not in _RUNTIME_ENVIRONMENT_KEYS
        }
        exact_environment[RUNTIME_URL_ENV] = self.url
        exact_environment[RUNTIME_TOKEN_ENV] = token
        exact_environment["ATREX_WORKSPACE"] = str(workspace)
        try:
            launch = wrap_agent_command(
                command,
                workspace=workspace,
                environment=exact_environment,
                repository_root=self.config.repository_root,
                provider_homes=provider_homes,
                hidden_host_paths=(campaign_root, scope_root.parent.parent),
                mode=self.config.agent_sandbox,
                bwrap_executable=self.config.bwrap_executable,
                agent_skills=self.config.agent_skills,
                agent_reference_projects=self.config.agent_reference_projects,
                agent_workspace=agent_workspace,
                provider_home=provider_home,
                read_only_paths=view.read_only_paths if view is not None else None,
            )
            with self._lock:
                self._launches[token] = launch
            capture = SessionCapture(
                scope_root / "sessions", backend=backend, command=command,
                provider_home=provider_home,
                context=dict(request_environment),
                native_environment=dict(environment),
            )
            print(f"[orchestrator] Session capture: {capture.root}", flush=True)
        except BaseException:
            self.revoke(token)
            raise
        return RuntimeSessionLease(self, token, launch, capture)

    def execute_journal(
        self,
        capability: RuntimeCapability,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        """Apply one synchronous mutation or query to the private Runtime Journal."""
        from supervisor.journal import SupervisorJournalService

        service = SupervisorJournalService(
            workspace=capability.workspace,
            campaign_root=capability.campaign_root,
            evidence_root=capability.evidence_root,
            git_workspace=capability.worktree,
        )
        with self._scope_lock(capability):
            try:
                # A request may have queued behind another mutation before its
                # lease was revoked. Do not apply that stale queued mutation.
                if self.authorize(capability.token) != capability:
                    raise RuntimeStateError("Session capability has been revoked.", code="invalid_capability")
                if request.get("operation") == "episode_report":
                    with self._lock:
                        view = self._workspace_views.get(capability.token)
                    if view is not None:
                        view.publish()
                result = service.execute(request)
                returncode = 0
            except (AgentRequestError, RuntimeStateError) as exc:
                result = exc.response
                returncode = 2 if result["repairable"] else 75
            except ValueError as exc:
                operation = request.get("operation")
                hint = {
                    "direction_load": (
                        "Use list-directions --output-path scratch/directions.json, "
                        "then load a returned Direction ID."
                    ),
                    "experiment_load": (
                        "Use list-experiments --output-path scratch/experiments.json, "
                        "then load a returned Experiment ID."
                    ),
                    "directions_list": (
                        "Choose a writable, workspace-relative --output-path such as "
                        "scratch/directions.json, not a symlink or directory."
                    ),
                    "experiments_list": (
                        "Choose a writable, workspace-relative --output-path such as "
                        "scratch/experiments.json, not a symlink or directory."
                    ),
                    "direction_update": (
                        "Correct the request or Direction lifecycle as indicated, then resubmit "
                        "update-direction. Use load-direction for associated_experiment_ids. "
                        "Every closure requires explicit supporting_experiment_ids and "
                        "hypothesis_status=unresolved|supported|refuted; lifecycle is not a verdict."
                    ),
                    "experiment_record": (
                        "Correct the indicated field or referenced state, then resubmit "
                        "record-experiment. Use record-read for Gateway IDs and load-direction "
                        "for lifecycle state. Cite at least one real Kernel-bound Gateway Record. "
                        "Late evidence may attach to a closed Direction without restarting it."
                    ),
                    "episode_report": (
                        "Correct the report or indicated Journal state, then call episode-report "
                        "again. Read the cited Gateway records to verify that the selected "
                        "Experiment measures the exact current kernel.py."
                    ),
                }.get(
                    operation if isinstance(operation, str) else None,
                    "Use python3 tools/sandbox.py --help to select a supported operation.",
                )
                result = error_response(str(exc), next_action=hint)
                returncode = 2
            except Exception:
                error_id = f"rterr_{uuid.uuid4().hex}"
                logging.getLogger(__name__).exception("Journal request failed: %s", error_id)
                result = error_response(
                    "Journal operation outcome could not be confirmed.",
                    code="runtime_failure", repairable=False, next_action=UNKNOWN_OUTCOME,
                    error_id=error_id,
                )
                returncode = 75
        stdout = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        process = subprocess.CompletedProcess(
            args=["journal", str(request.get("operation") or "")],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )
        response_record_id = self._record_process_response(
            capability, "journal", process
        )
        self._audit(
            capability,
            "journal",
            returncode,
            [str(request.get("operation") or "")],
            response_record_id=response_record_id,
        )
        return _command_result(process, stdout_limit=MAX_GATEWAY_RESULT_BYTES)

    @staticmethod
    def _private_scope_for(workspace: Path) -> tuple[Path, Path]:
        """Return the Campaign root and invisible per-worktree Runtime scope."""
        campaign_root = workspace
        git = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if git.returncode == 0 and git.stdout.strip():
            raw_common = Path(git.stdout.strip())
            common = (
                (workspace / raw_common).resolve()
                if not raw_common.is_absolute()
                else raw_common.resolve()
            )
            if common.is_dir() and common.name == ".git":
                campaign_root = common.parent
        scope_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in workspace.name
        )[:80] or "workspace"
        scope_digest = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
        campaign_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in campaign_root.name
        )[:80] or "campaign"
        campaign_digest = hashlib.sha256(str(campaign_root).encode()).hexdigest()[:12]
        private_campaign_root = (
            campaign_root.parent
            / ".atrex-supervisor-runtime"
            / f"{campaign_name}-{campaign_digest}"
        )
        scope_root = (
            private_campaign_root
            / "workspaces"
            / f"{scope_name}-{scope_digest}"
        )
        return campaign_root, scope_root

    def evidence_root(self, workspace: Path) -> Path:
        """Locate supervisor-owned evidence without exposing it to the Agent."""
        _, scope_root = self._private_scope_for(workspace.resolve())
        return scope_root / "evidence"

    def _materialize_atrex_bench_runtime(self, source: Path | None) -> Path | None:
        """Create one private evaluator copy shared by every Agent Session."""
        if source is None:
            return None
        source = source.expanduser().resolve()
        evaluator = source / "scripts" / "run_eval.py"
        package = source / "src" / "atrex_bench"
        if not evaluator.is_file() or not package.is_dir():
            raise FileNotFoundError(
                "invalid private Atrex-Bench runtime root: expected "
                f"{evaluator} and {package}"
            )
        destination = self.state_root / "evaluator-runtime" / "atrex-bench"
        (destination / "scripts").mkdir(parents=True, mode=0o700)
        shutil.copy2(evaluator, destination / "scripts" / "run_eval.py")
        shutil.copytree(
            package,
            destination / "src" / "atrex_bench",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        return destination

    def execute_gateway(
        self,
        capability: RuntimeCapability,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        argv = _validated_argv(request.get("argv"))
        argv = [_translate_visible_path(item, capability.workspace) for item in argv]
        argv = _without_endpoint_overrides(argv)
        canonical = [
            "--workspace",
            str(capability.workspace),
            "--hardware",
            self.config.hardware,
            "--timeout",
            str(self.config.sandbox_timeout),
        ]
        if self.config.sandbox_url:
            canonical += ["--url", self.config.sandbox_url]
        elif self.config.sandbox_profile:
            canonical += ["--gateway-profile", self.config.sandbox_profile]
        elif self.config.sandbox_ssh:
            canonical += ["--ssh", self.config.sandbox_ssh]
            if self.config.sandbox_ssh_init:
                canonical += ["--ssh-init", self.config.sandbox_ssh_init]
            if self.config.sandbox_ssh_gpu is not None:
                canonical += ["--ssh-gpu", str(self.config.sandbox_ssh_gpu)]
            if self.config.sandbox_health_command:
                canonical += ["--health-command", self.config.sandbox_health_command]
            for bind in self.config.sandbox_ssh_runtime_binds:
                canonical += ["--ssh-runtime-bind", bind]
        command = [
            sys.executable,
            str(self.config.repository_root / "supervisor" / "gateway.py"),
            *canonical,
            *argv,
        ]
        environment = dict(self._environment)
        environment.update(capability.request_environment)
        environment.pop("ATREX_AKA_REUSE_GATEWAY_RESULTS", None)
        environment.pop("ATREX_AKA_INTERNAL_MEASUREMENT", None)
        environment.pop("ATREX_AKA_COMPARISON_RUN_TIMEOUT", None)
        from .optimization_policy import MODE_STATE_ENV

        environment[MODE_STATE_ENV] = str(capability.evidence_root.parents[2] / "optimization-policy.json")
        environment[SUPERVISOR_EVIDENCE_ROOT_ENV] = str(capability.evidence_root)
        environment[SUPERVISOR_HISTORY_ROOT_ENV] = str(
            capability.campaign_root / ".atrex_long_horizon" / "episodes"
        )
        if self.atrex_bench_runtime is not None:
            environment[ATREX_BENCH_RUNTIME_ENV] = str(self.atrex_bench_runtime)
        if self.config.private_reference_dir is not None:
            environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(
                self.config.private_reference_dir
            )
        with self._request_slots:
            process = subprocess.run(
                command,
                cwd=capability.workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        response_record_id = self._record_process_response(
            capability, "gateway", process
        )
        process = self._project_process_for_agent(
            capability,
            process,
            dry_run=_has_option(argv, "--dry-run"),
        )
        self._audit(
            capability,
            "gateway",
            process.returncode,
            argv,
            response_record_id=response_record_id,
        )
        return _command_result(
            process,
            stdout_limit=_gateway_output_limit(argv),
        )

    def execute_wiki(
        self,
        capability: RuntimeCapability,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        tool = request.get("tool")
        if tool not in {"query_nl", "query_wiki", "query_hardware"}:
            raise ValueError("unsupported Wiki tool")
        argv = _validated_argv(request.get("argv"))
        argv = [_translate_visible_path(item, capability.workspace) for item in argv]
        forbidden = {
            "query_nl": {"--store-root"},
            "query_wiki": {"--json-store"},
            "query_hardware": {"--store"},
        }[str(tool)]
        _reject_abbreviated_options(
            argv, forbidden | {"--file", "--keep-workspace", "--max-bytes", "--brief"},
        )
        if any(item.split("=", 1)[0] in forbidden for item in argv):
            raise ValueError("Agent cannot override the Supervisor-owned Wiki store")
        if _has_option(argv, "--keep-workspace"):
            raise ValueError("Agent cannot retain a Supervisor-owned Wiki workspace")
        for index, item in enumerate(argv):
            if item == "--file":
                if index + 1 >= len(argv):
                    raise ValueError("Wiki --file requires a value")
                raw_source = argv[index + 1]
            elif item.startswith("--file="):
                raw_source = item.split("=", 1)[1]
            else:
                continue
            source = Path(raw_source)
            if source.is_absolute() or ".." in source.parts:
                raise ValueError("Wiki --file must be relative to the Agent workspace")
            resolved = (capability.workspace / source).resolve()
            if not resolved.is_relative_to(capability.workspace):
                raise ValueError("Wiki --file must remain inside the Agent workspace")
            relative = resolved.relative_to(capability.workspace)
            if relative.parent == Path("memory") and AUDIT_FILENAME_RE.fullmatch(relative.name):
                raise ValueError("Promotion audits are Supervisor-only; put the Wiki query in scratch/")
        argv = _with_wiki_projection_defaults(str(tool), argv)
        script = self.config.repository_root / "gpu-wiki" / "tools" / f"{tool}.py"
        environment = dict(self._environment)
        environment.update(capability.request_environment)
        profile_root = capability.wiki_profile_root
        if profile_root.is_symlink():
            raise ValueError("Wiki profile directory cannot be a symlink")
        profile_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        environment[WIKI_PROFILE_ROOT_ENV] = str(profile_root)
        with self._request_slots:
            process = subprocess.run(
                [sys.executable, str(script), *argv],
                cwd=capability.workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=1800,
            )
        response_record_id = self._record_process_response(
            capability, f"wiki:{tool}", process
        )
        process = self._project_process_for_agent(capability, process)
        self._audit(
            capability,
            f"wiki:{tool}",
            process.returncode,
            argv,
            response_record_id=response_record_id,
        )
        return _command_result(
            process,
            stdout_limit=MAX_WIKI_RESULT_BYTES,
        )

    def _project_process_for_agent(
        self,
        capability: RuntimeCapability,
        process: subprocess.CompletedProcess[str],
        *,
        dry_run: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Hide Supervisor-only locations while preserving actionable CLI text."""
        replacements = {
            str(capability.wiki_profile_root): "<private-wiki-profile>",
            str(capability.workspace): "/home/agent/workspace",
            str(capability.evidence_root): "<private-runtime-evidence>",
            str(self.state_root): "<private-runtime-state>",
            str(self.config.repository_root): "<supervisor-source>",
        }
        if self.atrex_bench_runtime is not None:
            replacements[str(self.atrex_bench_runtime)] = "<private-evaluator>"
        if self.config.private_reference_dir is not None:
            replacements[str(self.config.private_reference_dir)] = "<private-reference>"
        if getattr(capability, "worktree", None) is not None:
            replacements[str(capability.worktree)] = "/home/agent/workspace"
        if self.config.sandbox_url:
            replacements[self.config.sandbox_url] = "<configured-gateway>"

        def sanitize(value: str | None) -> str:
            output = value or ""
            for private, visible in sorted(
                replacements.items(), key=lambda item: len(item[0]), reverse=True
            ):
                if private:
                    output = output.replace(private, visible)
            return output

        stdout = sanitize(process.stdout)
        if dry_run:
            stdout = _project_dry_run_stdout(stdout)
        return subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=sanitize(process.stderr),
        )

    def _record_process_response(
        self,
        capability: RuntimeCapability,
        operation: str,
        process: subprocess.CompletedProcess[str],
    ) -> str:
        """Persist exact command channels before creating the Agent projection."""
        record_id = f"rtresp_{uuid.uuid4().hex}"
        root = capability.evidence_root / "runtime-responses"
        if root.is_symlink():
            raise RuntimeError("Runtime response directory became unsafe")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = root / f"{record_id}.json"
        payload = {
            "response_record_id": record_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "exit_code": int(process.returncode),
            "stdout": process.stdout or "",
            "stderr": process.stderr or "",
        }
        temporary = root / f".{record_id}.tmp"
        # UUID-named response files do not share mutable state.
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
        return record_id

    def _audit(
        self,
        capability: RuntimeCapability,
        operation: str,
        exit_code: int,
        argv: list[str],
        *,
        response_record_id: str | None = None,
    ) -> None:
        capability.evidence_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = capability.evidence_root / "runtime-requests.jsonl"
        row = {
            "request_id": f"rtreq_{uuid.uuid4().hex}",
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "exit_code": int(exit_code),
            "response_record_id": response_record_id,
            "request_digest": hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest(),
        }
        with self._scope_lock(capability, audit=True), path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

_ACTIVE_LOCK = threading.RLock()
_ACTIVE_RUNTIME: SupervisorRuntime | None = None


@contextmanager
def activate_supervisor_runtime(runtime: SupervisorRuntime) -> Iterator[None]:
    """Expose one Campaign Runtime to central Agent process launching."""
    global _ACTIVE_RUNTIME
    with _ACTIVE_LOCK:
        if _ACTIVE_RUNTIME is not None:
            raise RuntimeError("another Supervisor Runtime is already active")
        _ACTIVE_RUNTIME = runtime
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            if _ACTIVE_RUNTIME is runtime:
                _ACTIVE_RUNTIME = None


def active_supervisor_runtime() -> SupervisorRuntime | None:
    with _ACTIVE_LOCK:
        return _ACTIVE_RUNTIME


def supervisor_evidence_root(workspace: Path) -> Path:
    """Return the authoritative evidence root for an active Campaign."""
    runtime = active_supervisor_runtime()
    if runtime is None:
        return workspace / ".atrex_long_horizon"
    return runtime.evidence_root(workspace)


def supervisor_journal_path(workspace: Path) -> Path:
    """Return the private Journal path for one Episode workspace."""
    return supervisor_evidence_root(workspace) / "journal.json"


def supervisor_campaign_root(workspace: Path) -> Path:
    """Locate persistent Campaign evidence, including after the service closes."""
    _, scope_root = SupervisorRuntime._private_scope_for(workspace.resolve())
    return scope_root.parent.parent


__all__ = [
    "RUNTIME_TOKEN_ENV",
    "RUNTIME_URL_ENV",
    "SUPERVISOR_EVIDENCE_ROOT_ENV",
    "SUPERVISOR_HISTORY_ROOT_ENV",
    "WIKI_PROFILE_ROOT_ENV",
    "RuntimeSessionLease",
    "SupervisorRuntime",
    "SupervisorRuntimeConfig",
    "activate_supervisor_runtime",
    "active_supervisor_runtime",
    "supervisor_campaign_root",
    "supervisor_evidence_root",
    "supervisor_journal_path",
]
