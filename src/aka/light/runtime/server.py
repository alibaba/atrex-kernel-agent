"""Campaign-owned GPU/Wiki, Journal, research, and commit-review service."""
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
from collections.abc import Mapping
from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from aka.contracts.evaluation import DEFAULT_EVALUATION_STEP_TIMEOUT, derived_evaluation_timeout
from aka.light.evaluation.policy import SHAPE_BATCH_SIZE
from aka.light.evaluation.shapes import shape_request_timeout
from aka.light.evaluation.public_result import compact_receipt

from aka.light.evaluation.gateway import (
    COMMAND_WORKSPACE_MANIFEST,
    EPISODE_EVALUATIONS_PATH,
    build_parser,
    configured_queue_wait_grace,
    requested_sync_paths,
    validate_evaluation_options,
)
from aka.light.workspace.files import (
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    InputSizeLimitError,
    publish,
    read_input,
    relative_path,
    snapshot,
)
from aka.light.experiment.store import (
    ABBA_BATCH_ROOT_ENV,
    BATCH_ROOT_ENV,
    JOB_ROOT_ENV,
    RUNTIME_REQUEST_ID_ENV,
    TaskRunning,
    private_write,
)
from aka.light.experiment.records import task_conflict_response
from aka.light.workflow.journal_service import SupervisorJournalService
from aka.light.runtime.request_errors import AgentRequestError, RuntimeStateError, error_response
from aka.light.runtime.request_feedback import augment_error
from aka.light.runtime.plugin_output import (
    MAX_OUTPUT_BYTES,
    check_executor_output_failure,
    project_plugin_output,
    raise_output_error,
)
from aka.light.workflow.constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_HARNESS
from aka.light.workspace.episode import EpisodeWorkspace
from aka.light.experiment.frontier import kernel_full_evaluation_evidence
from aka.light.runtime.errors import (
    RequestValidationError, RequestDispatchTimeout, GatewayExecutionUnavailable, SessionRevokedError,
    ResultPublicationError,
)
from aka.light.execution.retry import InfrastructureUnavailable
from aka.light.runtime.paths import (
    supervisor_campaign_root,
    supervisor_research_catalog_root,
    supervisor_submission_root,
)
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
    "--timeout", "--shape-batch-size", "--gpu-clock-mhz",
    "--max-input-file-mb", "--max-output-file-mb",
})
FORBIDDEN = frozenset({
    "--preflight",
    "--check-health",
    # Agent ABBA identifies its baseline by Git commit. The Runtime owns the
    # temporary source export and its private path.
    "--baseline-path",
    # Managed Sessions query the Campaign's assigned target. Letting an Agent
    # choose another product or force a shared refresh would make the returned
    # facts disagree with the Kernel execution target.
    "--env-gpu",
    "--env-capabilities",
    "--env-force",
})
WIKI_STORES = {"query_nl": "--store-root", "query_wiki": "--json-store", "query_hardware": "--store"}
WIKI_RUNTIME_TOOLS = frozenset({"wiki-query", "wiki"})


def _journal_evidence_hint(modules: tuple[str, ...] | list[str]) -> str:
    enabled = set(modules)
    locations = ["PROGRESS.md"]
    if "directions" in enabled:
        locations.append("research-library/directions/<direction_id>/vN/direction.json")
    if "experiments" in enabled:
        locations.append("research-library/experiments/<experiment_id>/vN/experiment.json")
    locations.append("records/<gateway_record_id>.json")
    if enabled:
        return "Read " + ", ".join(locations) + " for visible IDs and evidence. "
    return (
        "Inspect PROGRESS.md and records/<gateway_record_id>.json to check citations; "
        "submit-kernel requires a passing full Evaluate for the exact commit. "
    )


def sol_evaluation_config(workspace: Path) -> bytes | None:
    """Read optional SOL policy from V0, never from mutable files or the index."""
    from aka.light.workspace.state import v0_baseline_commit

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
    if args.kind == "command" and args.command in ([], ["--"]):
        raise ValueError(
            "--kind command requires argv after --; use --kind command -- <command>"
        )
    validate_evaluation_options(args)
    return args


def managed_gateway_help(argv: list[str]) -> dict | None:
    """Return the Agent-visible managed contract instead of low-level CLI help.

    The underlying parser also owns Supervisor-only endpoint, target, timeout and
    evaluator controls.  Showing that raw help would invite requests that this
    HTTP boundary intentionally rejects, so ``--kind OPERATION --help`` exposes
    the same filtered schema attached to validation errors.
    """
    options = argv[:argv.index("--")] if "--" in argv else argv
    if "--help" not in options and "-h" not in options:
        return None
    kind = None
    for index, word in enumerate(options):
        if word == "--kind" and index + 1 < len(options):
            kind = options[index + 1]
        elif word.startswith("--kind="):
            kind = word.split("=", 1)[1]
    from aka.light.runtime.request_feedback import gateway_schema
    schema = gateway_schema(kind, build_parser(_RequestParser))
    return {
        "exit_code": 0,
        "stdout": json.dumps(
            {
                "managed_runtime_request_schema": schema,
                "instruction": (
                    "Use only these Agent-controlled options. The Supervisor supplies target, "
                    "endpoint, credentials, evaluator policy, batching and request deadlines."
                ),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        "stderr": "",
    }


def managed_gateway_argv(
    argv: list[str], parsed: argparse.Namespace | None, hardware: str
) -> list[str]:
    """Inject Supervisor-owned target selection for managed environment reads."""
    if parsed is None or parsed.kind != "env":
        return argv
    if not isinstance(hardware, str) or not hardware.strip():
        raise ValueError("Managed env query requires the Campaign hardware identity")
    return [*argv, "--env-gpu", hardware.strip(), "--env-capabilities"]


def agent_gateway_response(result: dict) -> tuple[dict, int | None]:
    """Select the HTTP response contract for one completed Gateway call.

    Measurements have a durable Gateway Record, so their terminal response is
    a compact receipt plus the executor exit classification header. Env is a
    read-only Agate capability query with no Gateway Record; return its already
    redacted executor streams unchanged so capability values are not discarded
    by the measurement-summary whitelist.
    """
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        return result, None
    if agent_result.get("operation") == "env":
        response = dict(result)
        response.pop("agent_result", None)
        return response, None
    return compact_receipt(agent_result), result["exit_code"]


def materialized_baseline_argv(argv: list[str], baseline_path: str) -> list[str]:
    """Replace the Agent's commit selector with Runtime's private source path."""
    result: list[str] = []
    index = 0
    inserted = False
    while index < len(argv):
        word = argv[index]
        if word == "--":
            if not inserted:
                result.extend(("--baseline-path", baseline_path))
                inserted = True
            result.extend(argv[index:])
            break
        if word == "--baseline-commit":
            if index + 1 >= len(argv):
                raise ValueError("--baseline-commit requires a value")
            index += 2
            continue
        if word.startswith("--baseline-commit="):
            index += 1
            continue
        result.append(word)
        index += 1
    if not inserted:
        result.extend(("--baseline-path", baseline_path))
    return result


def validate_gateway_transport(
    parsed,
    environment,
    *,
    ssh,
    url,
    profile,
) -> None:
    """Validate the SDK transport before staging or reserving a GPU task."""
    if ssh:
        return
    from aka.light.evaluation.source_transport import validate_gateway_sdk

    validate_gateway_sdk(
        parsed,
        environment,
        authority_url=url,
        authority_profile=profile,
    )


def failure(message: str, *, repairable: bool = True, next_action: str | None = None,
            code: str | None = None, **details) -> dict:
    from aka.light.runtime.request_errors import UNKNOWN_OUTCOME
    return error_response(message, code=code or ("invalid_request" if repairable else "outcome_unknown"),
                          repairable=repairable, next_action=next_action or (
                              "Correct the arguments, using full option names, then retry."
                              if repairable else UNKNOWN_OUTCOME), **details)


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
    from aka.light.execution.gateway_jobs import TERMINAL_VALIDATION_ENV
    return {key: value for key, value in environment.items() if
            not key.startswith(("AGATE_", "ATREX_SANDBOX_", "ATREX_WIKI_", "GPU_WIKI_", "ATREX_GPU_WIKI_"))
            and key not in {OWNER_ENV, URL_ENV, TOKEN_ENV, "ATREX_AKA_RUNTIME_CONNECTION_FILE", JOB_ROOT_ENV, BATCH_ROOT_ENV,
                            ABBA_BATCH_ROOT_ENV, RUNTIME_REQUEST_ID_ENV,
                            "ATREX_AKA_MEASUREMENT_REPETITIONS",
                            "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_BENCH_RUNTIME_ROOT",
                            "ATREX_AKA_IDENTITY_KEY", TERMINAL_VALIDATION_ENV}}


@dataclass(frozen=True)
class RuntimeConfig:
    hardware: str
    timeout: int = derived_evaluation_timeout()
    evaluation_step_timeout: int = DEFAULT_EVALUATION_STEP_TIMEOUT
    shape_batch_size: int = SHAPE_BATCH_SIZE
    gpu_clock_mhz: int | None = None
    url: str = ""
    profile: str = ""
    ssh: str = ""
    ssh_init: str = ""
    ssh_gpu: int | None = None
    health_command: str = ""
    # Native evaluator inputs remain Supervisor-owned even when exact Shapes
    # are public.  This is independent of ``private_reference_dir``, which
    # denotes the generalized hidden-Shape policy.
    evaluator_reference_dir: Path | None = None
    private_reference_dir: Path | None = None
    atrex_bench_root: Path | None = None
    wiki_profile_root: Path | None = None
    task_id: str = ""
    optimization_mode: str = "fast"
    workspace_policy: str = ""
    workspace: Path | None = None
    # Pooled trajectories share the Campaign-owned Measurement Store. Journal,
    # Research, Session, and Runtime-audit state lives in lineage namespaces
    # within the same Campaign-private tree, not in trajectory-owned stores.
    # Serial campaigns leave this unset and use the same Campaign layout.
    measurement_root: Path | None = None
    # None preserves the Gateway execution + remote queue grace + cleanup budget.
    request_timeout: float | None = None
    queue_timeout: float = 60
    measurement_repetitions: int = 1
    pause_agent_during_requests: bool = True
    research_synchronization: str = "independent"
    wiki_enabled: bool = True
    wiki_options: WikiConfig = field(default_factory=WikiConfig)
    # Selected aka.contracts.wiki.WikiProvider; None selects the built-in local store.
    wiki_provider: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.workspace_policy, str):
            raise TypeError("Runtime workspace_policy must be text")
        if type(self.pause_agent_during_requests) is not bool:
            raise TypeError("Runtime pause_agent_during_requests must be a boolean")
        if self.research_synchronization not in {
            "independent", "synchronous", "asynchronous"
        }:
            raise ValueError(
                "Runtime research_synchronization must be independent, synchronous, or asynchronous"
            )
        if self.measurement_root is not None and not isinstance(
            self.measurement_root, Path
        ):
            raise TypeError("Runtime measurement_root must be a pathlib.Path")
        for name in ("evaluator_reference_dir", "private_reference_dir"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"Runtime {name} must be a pathlib.Path")
        if type(self.evaluation_step_timeout) is not int or self.evaluation_step_timeout <= 0:
            raise ValueError("Runtime evaluation_step_timeout must be a positive integer")
        if type(self.shape_batch_size) is not int or self.shape_batch_size <= 0:
            raise ValueError("Runtime shape_batch_size must be a positive integer")
        if self.gpu_clock_mhz is not None and (
            type(self.gpu_clock_mhz) is not int or self.gpu_clock_mhz <= 0
        ):
            raise ValueError("Runtime gpu_clock_mhz must be a positive integer")
        for name in ("request_timeout", "queue_timeout"):
            value = getattr(self, name)
            if name == "request_timeout" and value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Runtime {name} must be a finite positive number of seconds")

    def command(self, workspace: Path) -> list[str]:
        result = [sys.executable, str(ROOT / "evaluation/gateway.py"),
                  "--workspace", str(workspace), "--hardware", self.hardware,
                  "--timeout", str(self.timeout),
                  "--evaluation-step-timeout", str(self.evaluation_step_timeout),
                  "--shape-batch-size", str(self.shape_batch_size)]
        if self.gpu_clock_mhz is not None:
            result += ["--gpu-clock-mhz", str(self.gpu_clock_mhz)]
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
    require_gluon: bool = False
    baseline_kernel_commit_id: str | None = None


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
        owner = self.server.owner
        if status >= 400:
            result = augment_error(
                result, route=self.path, request=getattr(self, "decoded_request", None),
                research_records=owner.journal_factory.research_records,
                directions_enabled=owner.journal_factory.directions_enabled,
                experiments_enabled=owner.journal_factory.experiments_enabled,
                parser=build_parser() if self.path == "/v1/gateway/execute" and status == 400 else None,
                catalog=owner.plugins.catalog() if self.path == "/v1/plugins/execute" and status == 400 else (),
            )
            from aka.light.evaluation.projection import credential_values, redact_diagnostics
            result = redact_diagnostics(result,
                private_paths=(str(owner.root), str(owner.audit_root or ""), str(ROOT),
                               str(getattr(owner.config, "evaluator_reference_dir", None) or ""),
                               str(owner.config.private_reference_dir or ""), owner.config.url,
                               str(owner.config.atrex_bench_root or "")),
                private_values=(*credential_values(owner.environment), getattr(self, "bearer", "")))
        if getattr(self, "audit_active", False):
            try:
                owner.audit_request_finished(
                    self.request_id,
                    http_status=status,
                    result=result,
                    execution_exit_code=execution_exit_code,
                    duration_seconds=max(
                        0.0,
                        time.monotonic()
                        - getattr(self, "audit_started_monotonic", time.monotonic()),
                    ),
                )
            except (OSError, ValueError, TypeError):
                logging.getLogger(__name__).exception(
                    "Failed to finalize Runtime request audit %s", self.request_id
                )
            self.audit_active = False
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
        self.audit_active = False
        self.audit_started_monotonic = time.monotonic()
        # Gateway jobs own their remote deadlines/retry policy. Keeping the
        # local HTTP request alive prevents the Agent from escaping a still
        # running Runtime call and starting unrelated work.
        deadline = (
            float("inf")
            if self.path == "/v1/gateway/execute"
            else time.monotonic() + owner.request_timeout_seconds
        )
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
            owner.audit_request_started(self.request_id, route=self.path)
        except (OSError, ValueError, TypeError):
            logging.getLogger(__name__).exception(
                "Failed to start Runtime request audit %s", self.request_id
            )
        else:
            self.audit_active = True
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
                if self.audit_active:
                    try:
                        owner.audit_request_details(
                            self.request_id, route=self.path, request=request
                        )
                    except (OSError, ValueError, TypeError):
                        logging.getLogger(__name__).exception(
                            "Failed to bind Runtime request audit %s", self.request_id
                        )
                if (
                    self.path == "/v1/journal/execute"
                    and request.get("operation") == "kernel_submit"
                ):
                    deadline = float("inf")
            # Suspension is optional. When enabled, stop only this Session's
            # Provider tree; the handler/executor remains outside that tree.
            # Release before reply because a stopped CLI cannot drain a large
            # response. Disconnection never releases the scope early or cancels
            # an already submitted job.
            wait = (
                capability.session_control.request()
                if capability.session_control and owner.config.pause_agent_during_requests
                else nullcontext()
            )
            with wait:
                if owner.is_wiki_request(self.path, request):
                    result = owner.execute(capability, self.path, request)
                else:
                    identity, prepared = None, False

                    def check_running():
                        # A repeated call must not spend the Session queue
                        # timeout before reaching the nonblocking task lock.
                        # Use the full staged identity, not argv alone: the
                        # Agent may have edited sources since the first call.
                        nonlocal identity, prepared
                        if not prepared:
                            identity = owner.execute(capability, self.path, request,
                                                     prepare_identity_only=True)
                            prepared = True
                        if identity is not None:
                            owner.measurements.check_running(identity)

                    with owner.request_lock(
                        capability, deadline, self.request_id,
                        before_wait=check_running if self.path == "/v1/gateway/execute" else None,
                    ):
                        with owner.lock:
                            if owner.closed or owner.capabilities.get(token) is not capability:
                                raise SessionRevokedError("Session capability revoked")
                        result = owner.execute(capability, self.path, request)
            execution_exit_code = None
            if self.path == "/v1/gateway/execute" and isinstance(result.get("agent_result"), dict):
                result, execution_exit_code = agent_gateway_response(result)
            self.reply(200, result, execution_exit_code=execution_exit_code)
        except TaskRunning as error:
            self.reply(200, task_conflict_response(error))
        except AgentRequestError as error:
            self.reply(400, error.response)
        except RuntimeStateError as error:
            self.reply(503, error.response)
        except InfrastructureUnavailable as error:
            # The operation is known to be incomplete due to infrastructure;
            # it is not an unknown handler outcome or a Kernel verdict.
            self.reply(
                503,
                failure(
                    str(error),
                    repairable=False,
                    code="infrastructure_unavailable",
                    reason="infrastructure_unavailable",
                    error_class="infra",
                    outcome="incomplete",
                    request_id=self.request_id,
                    retry_after_seconds=DISPATCH_RETRY_SECONDS,
                    next_action=(
                        f"Keep the candidate and request unchanged. Wait at least "
                        f"{DISPATCH_RETRY_SECONDS} seconds, then retry the identical request; "
                        "the Runtime will reconcile and reuse completed private Gateway evidence. "
                        "Do not reinterpret this as a Kernel failure."
                    ),
                ),
                retry_after=DISPATCH_RETRY_SECONDS,
            )
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
        except GatewayExecutionUnavailable as error:
            next_action = (
                "The Gateway process started, so this request may already have created or completed "
                "an Agate job. Keep the candidate and request unchanged. Inspect available Kernel/Record "
                "history first; if no terminal Record is visible, report this request_id to the operator "
                "for Job reconciliation. Do not loop or submit a modified replacement blindly."
            )
            logging.getLogger(__name__).warning(
                "Supervisor Runtime request %s ended with unknown Gateway outcome: %s",
                self.request_id, error.code,
            )
            self.reply(503, failure(
                str(error),
                repairable=False,
                code=error.code,
                next_action=next_action,
                reason=error.reason,
                outcome="unknown",
                phase="gateway_execution",
                gateway_started=True,
                cancellation_requested=error.cancellation_requested,
                cancellation_confirmed=False,
                request_id=self.request_id,
                **error.details,
            ))
        except ResultPublicationError as error:
            record_id = error.gateway_record_id
            record_instruction = (
                f"The GPU operation completed and its result is saved as {record_id}. Read "
                f"`records/{record_id}.json`; do not "
                "resubmit the completed request."
                if record_id else
                "The GPU operation completed, but its saved Record ID could not be returned. "
                "Report this request_id to the operator for Record reconciliation; do not resubmit blindly."
            )
            next_action = (
                record_instruction
                + " Requested output files were not synchronized. If their contents are still needed, "
                  "make the remote output bounded and request only the required file in a new Dev probe."
            )
            recovery = [{"instruction": next_action}]
            if record_id:
                recovery.insert(0, {
                    "path": f"records/{record_id}.json",
                    "instruction": "Read the projected completed measurement; this does not rerun the GPU job.",
                })
            issue_path = error.details.get("output_path", "$")
            self.reply(503, failure(
                str(error),
                repairable=False,
                code=error.code,
                next_action=next_action,
                issues=[{"path": issue_path, "code": error.code, "message": str(error)[:2000]}],
                recovery=recovery,
                outcome="completed",
                phase=error.phase,
                execution_completed=True,
                result_recorded=record_id is not None,
                gateway_record_id=record_id,
                request_id=self.request_id,
                **error.details,
            ))
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
    def __init__(self, config: RuntimeConfig, *, workspace_provider=None, services=None):
        self.config = config
        from aka.light.evaluation.agate_native import audit_summary
        logging.getLogger(__name__).warning(
            "Agate native option audit: %s",
            json.dumps(audit_summary(), sort_keys=True),
        )
        from aka.light.runtime.services import builtin_services
        services = services if services is not None else builtin_services()
        self.executor, self.acceptance = services.executor, services.acceptance
        self.journal_factory = services.journal
        from aka.light.runtime.catalog import PluginRegistry
        self.wiki_provider = None
        if config.wiki_enabled:
            self.wiki_provider = config.wiki_provider
            if self.wiki_provider is None:
                from aka.light.wiki.local import LocalWikiProvider
                self.wiki_provider = LocalWikiProvider()
        self.plugins = PluginRegistry(enabled=config.wiki_enabled, provider=self.wiki_provider)
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
        # In-memory Episode bindings route Runtime requests to the persistent
        # Agent repository.  They are not durable Journals; authoritative
        # state lives in ResearchStore, SubmissionStore and MeasurementStore.
        self.journals: dict[Path, JournalBinding] = {}
        self.closed = False
        self.slots = threading.BoundedSemaphore(16)
        self.temporary = tempfile.TemporaryDirectory(prefix="aka-supervisor-")
        self.root = Path(self.temporary.name).resolve()
        self.audit_root = None
        if config.workspace:
            workspace = config.workspace.resolve()
            self.audit_root = supervisor_campaign_root(workspace)
            from aka.light.agent.home import open_private_directory
            with open_private_directory(self.audit_root):
                pass
            repository = ROOT.parents[2]
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            from aka.light.runtime.paths import (
                record_campaign_runtime_configuration,
            )

            record_campaign_runtime_configuration(
                workspace,
                {
                    "schema_version": 1,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "hardware": config.hardware,
                    "optimization_mode": config.optimization_mode,
                    "gateway_profile": config.profile or None,
                    "evaluation_timeout_seconds": config.timeout,
                    "evaluation_step_timeout_seconds": config.evaluation_step_timeout,
                    "shape_batch_size": config.shape_batch_size,
                    "gpu_clock_mhz": config.gpu_clock_mhz,
                    "measurement_repetitions": config.measurement_repetitions,
                    "request_timeout_seconds": config.request_timeout,
                    "queue_timeout_seconds": config.queue_timeout,
                    "pause_agent_during_requests": config.pause_agent_during_requests,
                    "wiki_enabled": config.wiki_enabled,
                    "transport": "ssh" if config.ssh else "agate_sdk",
                    "workspace_policy_sha256": hashlib.sha256(
                        config.workspace_policy.encode()
                    ).hexdigest(),
                    "source_checkout": {
                        "git_commit_id": revision.stdout.strip()
                        if revision.returncode == 0
                        else None,
                        "tracked_files_dirty": bool(dirty.stdout.strip())
                        if dirty.returncode == 0
                        else None,
                    },
                },
            )
        from aka.light.runtime.paths import supervisor_measurement_root
        self.measurement_root = (
            config.measurement_root.expanduser().resolve()
            if config.measurement_root is not None
            else (
                supervisor_measurement_root(config.workspace)
                if config.workspace is not None
                else self.root / "measurements"
            )
        )
        from aka.light.agent.home import open_private_directory
        with open_private_directory(self.measurement_root):
            pass
        self.experiments = services.experiments.open(self.measurement_root)
        self.measurements = self.experiments.store
        from aka.light.research import ResearchStore
        self.research = ResearchStore((self.audit_root or self.root) / "research")
        # The lock is private and survives restarts; never read an Agent-supplied lock.
        plugin_lock = (self.audit_root or self.root) / "plugins"
        self.plugins.check_lock(plugin_lock)
        publish(plugin_lock, ".atrex_plugins/lock.json", json.dumps(self.plugins.snapshot()).encode())
        self.wiki = WikiService((self.audit_root or self.root) / "wiki", config.wiki_options,
                                self.plugins.snapshot(), provider=self.wiki_provider)
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
        # Physical-job recovery gets one complete second wait window.  Size the
        # default Supervisor watchdog for both windows, including the
        # sequential Performance allowance for every Shape in one batch.
        # Explicit operator request timeouts remain authoritative.
        batch_timeout = shape_request_timeout(
            self.config.timeout,
            self.config.shape_batch_size,
            performance_timeout=self.config.evaluation_step_timeout,
        )
        return 2 * (batch_timeout + self.gateway_queue_wait_grace) + 120

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

    def _write_lineage_state(
        self,
        *,
        episode: int,
        branch: str,
        base_commit: str,
        base_kernel_commit_id: str | None,
        agent_head_commit_id: str | None,
        tool_modules: tuple[str, ...],
        segment_status: str,
        origin: Mapping[str, object] | None,
    ) -> None:
        """Checkpoint the current audit segment without an Episode Journal."""
        root = self.audit_root or self.root
        path = root / "state.json"
        trajectory = (origin or {}).get("trajectory", 1)
        if type(trajectory) is not int or trajectory <= 0:
            trajectory = 1
        from aka.light.workflow.search import merge_lineage_state

        merge_lineage_state(
            path,
            trajectory,
            current_episode=episode,
            episode_branch=branch,
            base_commit=base_commit,
            base_kernel_commit_id=base_kernel_commit_id,
            agent_head_commit_id=agent_head_commit_id,
            tool_modules=list(tool_modules),
            segment_status=segment_status,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )


    def baseline_kernel_commit(self, workspace: Path) -> str:
        """Return this Episode's canonical baseline in Agent-visible Git."""
        workspace = workspace.resolve(strict=True)
        with self.lock:
            binding = self.journals.get(workspace)
            baseline = (
                binding.baseline_kernel_commit_id if binding is not None else None
            )
        if not baseline:
            raise RuntimeError(
                "Episode baseline is not registered in the Agent-visible Git repository"
            )
        if binding.view is None or not binding.view.agent_git.commit_reachable(baseline):
            raise RuntimeError(
                "Registered Episode baseline is not reachable from Agent-visible Git"
            )
        return baseline

    def register_episode(self, workspace: Path, *, episode: int, base_commit: str,
                         branch: str, memory_version: int,
                         baseline_kernel_commit_id: str | None = None,
                         managed_agent_workspace: bool = False,
                         require_gluon: bool = False,
                         session_capture_root: Path | None = None,
                         origin: Mapping[str, object] | None = None):
        """Bind one audit segment to the trusted persistent Agent repository."""
        workspace = workspace.resolve(strict=True)
        if (type(episode) is not int or episode < 0 or type(memory_version) is not int
                or memory_version < 1 or not isinstance(branch, str) or not branch
                or type(require_gluon) is not bool
                or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", base_commit) is None
                or (
                    baseline_kernel_commit_id is not None
                    and re.fullmatch(
                        r"[0-9a-f]{40}(?:[0-9a-f]{24})?",
                        baseline_kernel_commit_id,
                    ) is None
                )):
            raise ValueError("Invalid persistent Agent Episode identity")
        lineage_root = self.audit_root or self.root
        trajectory = (origin or {}).get("trajectory", 1)
        if type(trajectory) is not int or trajectory <= 0:
            trajectory = 1
        lineage = f"T{trajectory}"
        submissions_root = supervisor_submission_root(workspace)
        evidence = (
            supervisor_research_catalog_root(workspace)
            / "reviews"
            / lineage
            / f"e{episode:08d}"
        )
        # Import pre-centralization submit audit once.  Historical Journal files
        # remain inert source evidence and are never recreated or updated.
        from aka.light.workflow.submissions import migrate_legacy_journals

        migrate_legacy_journals(
            lineage_root / "journals",
            submissions_root,
            lineage=lineage,
        )
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            existing = self.journals.get(workspace)
            if existing is not None:
                if existing.service.episode == episode:
                    if (existing.view is not None) != managed_agent_workspace:
                        raise RuntimeError("Cannot change Agent workspace ownership during an Episode")
                    if existing.require_gluon != require_gluon:
                        raise RuntimeError("Cannot change the final DSL phase during an Episode")
                    return existing.view.prepare() if existing.view else workspace
                if any(
                    capability.workspace == workspace
                    for capability in self.capabilities.values()
                ):
                    raise RuntimeError(
                        "Cannot advance the persistent Agent workspace while its prior "
                        "Episode Session is still active"
                    )
                self.journals.pop(workspace, None)
            view = None
            agent_workspace = workspace
            registered_baseline_kernel_commit_id = None
            if managed_agent_workspace:
                from aka.light.evaluation.hardware import hardware_vendor
                persistent_root = (self.audit_root or self.root) / "agent-workspace"
                view = (self.workspace_provider.open_draft if self.workspace_provider else EpisodeWorkspace)(workspace, evidence / "agent",
                                        is_ppu=hardware_vendor(self.config.hardware) == "ppu",
                                        research_records=self.journal_factory.research_records,
                                        research_modules=self.journal_factory.tool_modules,
                                        plugin_registry=self.plugins,
                                        workspace_policy=self.config.workspace_policy,
                                        persistent_root=persistent_root)
                agent_workspace = view.prepare()
                from aka.light.workspace.sources import from_git
                incumbent_source = from_git(workspace, base_commit)
                registered_baseline_kernel_commit_id = baseline_kernel_commit_id
                if registered_baseline_kernel_commit_id is not None:
                    if (
                        not view.agent_git.commit_reachable(
                            registered_baseline_kernel_commit_id
                        )
                        or from_git(
                            view.agent_git.workspace,
                            registered_baseline_kernel_commit_id,
                        )
                        != incumbent_source
                    ):
                        raise RuntimeError(
                            "Registered Agent baseline is not reachable or does not match "
                            "the Episode starting commit"
                        )
                else:
                    # ``base_commit`` is already an Agent-owned identity. Do
                    # not substitute an older commit with identical source in
                    # the direct Light layout. The fallback is retained only
                    # for legacy callers that still pass a separate source
                    # repository while migrating an older Campaign.
                    if view.agent_git.commit_reachable(base_commit):
                        registered_baseline_kernel_commit_id = base_commit
                    elif workspace != view.root:
                        registered_baseline_kernel_commit_id = (
                            view.agent_git.find_kernel_commit(incumbent_source)
                        )
                    if registered_baseline_kernel_commit_id is None:
                        raise RuntimeError(
                            "Episode starting commit is not reachable from Agent Git"
                        )
                self.measurements.kernel(
                    incumbent_source, registered_baseline_kernel_commit_id
                )
                # Runtime never creates ranking/promotion refs in Agent Git.
                # The Agent owns HEAD, branches, tags and commits. Only peer
                # history import/recovery may add Runtime-managed refs.
            else:
                # Legacy single-repository mode uses the same public/private
                # identity, but keeps the mapping explicit for projections.
                registered_baseline_kernel_commit_id = base_commit
            binding = JournalBinding(self.journal_factory.open(
                workspace=agent_workspace, git_workspace=workspace,
                campaign_root=lineage_root / "research", evidence_root=evidence,
                measurement_root=self.measurement_root,
                submissions_root=submissions_root,
                episode=episode,
                base_commit=base_commit,
                base_kernel_id=registered_baseline_kernel_commit_id,
                episode_branch=branch,
                tool_modules=self.journal_factory.tool_modules,
                research_root=lineage_root / "research",
                research_synchronization=self.config.research_synchronization,
                session_capture_root=session_capture_root,
                origin=origin,
                claim_reviewer=getattr(self, "research_claim_reviewer", None),
            ), view=view, require_gluon=require_gluon,
                baseline_kernel_commit_id=registered_baseline_kernel_commit_id)
            # Rebuild normal-file research projections at every Episode bind,
            # including recovery and Epoch-shared history import.
            binding.service.publish_research_files()
            with self.lock:
                previous = self.journals.get(agent_workspace)
                if previous is not None and previous.service.episode != episode:
                    if any(
                        capability.workspace == agent_workspace
                        for capability in self.capabilities.values()
                    ):
                        raise RuntimeError(
                            "Cannot rebind the persistent Agent workspace while its prior "
                            "Episode Session is still active"
                        )
                    self.journals.pop(agent_workspace, None)
                self.journals[workspace] = binding
                self.journals[agent_workspace] = binding
            self._write_lineage_state(
                episode=episode,
                branch=branch,
                base_commit=base_commit,
                base_kernel_commit_id=registered_baseline_kernel_commit_id,
                agent_head_commit_id=(
                    view.agent_git.head() if view is not None else base_commit
                ),
                tool_modules=tuple(self.journal_factory.tool_modules),
                segment_status="active",
                origin=origin,
            )
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
            from aka.light.workspace.episode import EPISODE_WORKSPACE_ENV
            from aka.light.workspace.managed_assets import PROTECTED_PATHS_ENV
            with capability.journal.lock:
                values[PROTECTED_PATHS_ENV] = json.dumps(capability.journal.view.prepare_session())
            values[EPISODE_WORKSPACE_ENV] = str(capability.journal.view.root)
            from aka.light.agent.sandbox import git_directory
            common = git_directory(capability.journal.view.source_workspace)
            if common is None:
                raise RuntimeError(
                    "Cannot resolve the managed Episode Git common directory; "
                    "refusing to start an Agent Session without private-path protection. "
                    "Repair the persistent Agent repository before retrying"
                )
            private_paths = [self.audit_root or self.root]
            if not common.is_relative_to(capability.journal.view.root):
                private_paths.append(common)
            values["ATREX_EPISODE_PRIVATE_PATHS"] = json.dumps(
                [str(path) for path in private_paths]
            )
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
                        capability.journal.view.publish()
                        service = capability.journal.service
                        self._write_lineage_state(
                            episode=service.episode,
                            branch=service.episode_branch,
                            base_commit=service.base_commit,
                            base_kernel_commit_id=service.base_kernel_id,
                            agent_head_commit_id=(
                                capability.journal.view.agent_git.head()
                            ),
                            tool_modules=service.tool_modules,
                            segment_status="idle",
                            origin=service.origin,
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

    def _acquire(self, lock, capability, deadline, *, before_wait=None):
        wait_deadline = min(deadline, time.monotonic() + self.config.queue_timeout)
        while True:
            if not self._live(capability):
                raise SessionRevokedError("Session revoked")
            if before_wait is not None and lock.locked():
                before_wait()
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                raise RequestDispatchTimeout("Supervisor request queue deadline exceeded; no executor started")
            if lock.acquire(timeout=min(0.2, remaining)):
                return

    @contextmanager
    def request_lock(self, capability, deadline, request_id, *, before_wait=None):
        self._acquire(capability.lock, capability, deadline, before_wait=before_wait)
        capability.deadline, capability.request_id = deadline, request_id
        try:
            yield
        finally:
            capability.deadline, capability.request_id = None, ""
            capability.lock.release()

    def run_executor(self, command, cwd, environment, capability):
        from aka.light.execution.process import ExecutionControl
        deadline = capability.deadline or (time.monotonic() + self.request_timeout_seconds)
        return self.executor.run_gateway(command, cwd, environment, ExecutionControl(
            deadline, capability.request_id, lambda: self._live(capability),
            lambda: self._acquire(self.slots, capability, deadline), self.slots.release,
        ))

    def is_wiki_request(self, route, request):
        if route == "/v1/wiki/query":
            return True
        if route == "/v1/plugins/execute" and request.get("action") == "call":
            return any(request.get("tool") == f"{plugin.id}.{name}" and tool.get("runtime_tool") in WIKI_RUNTIME_TOOLS
                       for plugin in self.plugins.plugins for name, tool in plugin.tools.items())
        return False

    def execute_trusted(self, workspace: Path, argv: list[str], *, execution_timeout: int | None = None,
                        parent=None, reference_snapshot: Path | None = None,
                        terminal_validation: bool = False,
                        kernel_ids: Mapping[str, str] | None = None,
                        journal_workspace: Path | None = None) -> dict:
        """Supervisor-only execution/reuse; no Session or HTTP transport switch.

        ``workspace`` is the exact staged source tree executed by the request.  A
        trusted Supervisor request may separately name the registered Episode
        workspace that owns Agent Git and its Journal. Keeping those authorities
        separate lets Runtime resolve reachable comparison commits without copying
        them into a second Git repository.
        """
        execution_workspace = workspace.resolve(strict=True)
        journal_workspace = (
            journal_workspace.resolve(strict=True)
            if journal_workspace is not None
            else None
        )
        capability = Capability(execution_workspace, {}, parent=parent)
        token = secrets.token_urlsafe(32)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            if journal_workspace is not None:
                binding = self.journals.get(journal_workspace)
                if binding is None or binding.view is None:
                    raise RuntimeError(
                        "Trusted execution Journal workspace is not registered with Agent Git"
                    )
                capability.journal = binding
            self.capabilities[token] = capability
        try:
            deadline = (
                float("inf")
                if terminal_validation
                else time.monotonic() + self.request_timeout_seconds
            )
            if parent is not None and parent.deadline is not None:
                deadline = min(deadline, parent.deadline)
            with self.request_lock(capability, deadline, "acceptance"):
                return self.execute(capability, "/v1/gateway/execute", {"argv": argv}, reuse_completed=True,
                                    execution_timeout=execution_timeout,
                                    reference_snapshot=reference_snapshot,
                                    terminal_validation=terminal_validation,
                                    trusted_kernel_ids=kernel_ids)
        finally:
            self.revoke(token)

    def execute(self, capability: Capability, route: str, request: dict, *,
                reuse_completed=False, reuse_correctness=False, candidate_source: bytes | None = None,
                execution_timeout: int | None = None, reference_snapshot: Path | None = None,
                prepare_identity_only: bool = False,
                terminal_validation: bool = False,
                trusted_kernel_ids: Mapping[str, str] | None = None) -> dict | None:
        if prepare_identity_only and route != "/v1/gateway/execute":
            return None
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
        from aka.light.execution.gateway_jobs import TERMINAL_VALIDATION_ENV
        environment.pop(TERMINAL_VALIDATION_ENV, None)
        if terminal_validation:
            environment[TERMINAL_VALIDATION_ENV] = "1"
        for key in (OWNER_ENV, URL_ENV, TOKEN_ENV, "ATREX_PRIVATE_REFERENCE_DIR"):
            environment.pop(key, None)
        kernel_ids: dict[str, str] = dict(trusted_kernel_ids or {})
        with tempfile.TemporaryDirectory(dir=self.root, prefix="request-") as temporary:
            staged = Path(temporary)
            with _validating_request():
                if route.endswith("query"):
                    command = self._wiki_command(capability, request["tool"], argv, staged, environment)
                    kind = "wiki"
                else:
                    argv = filter_options(argv, AUTHORITY, FORBIDDEN)
                    managed_help = managed_gateway_help(argv)
                    if managed_help is not None:
                        return None if prepare_identity_only else managed_help
                    parsed = parse_gateway(argv)
                    argv = managed_gateway_argv(argv, parsed, self.config.hardware)
                    if parsed is not None and parsed.kind == "env":
                        parsed = parse_gateway(argv)
                    if prepare_identity_only and (
                        parsed is None or parsed.kind == "env" or parsed.dry_run
                    ):
                        return None
                    if parsed is not None:
                        # Invalid operator overrides are known configuration
                        # failures, not Agent errors or unknown GPU outcomes.
                        # Reject before staging, task reservation or execution.
                        validate_gateway_transport(
                            parsed,
                            environment,
                            ssh=self.config.ssh,
                            url=self.config.url,
                            profile=self.config.profile,
                        )
                    if parsed is not None and parsed.kind != "env":
                        from aka.light.workspace.sources import freeze_for_execution, stage_frozen, layout_at
                        authority = (capability.journal.view.source_workspace if capability.journal and capability.journal.view
                                     else self.config.workspace or capability.workspace)
                        frozen_source = freeze_for_execution(authority, capability.workspace, candidate_source)
                        snapshot(
                            capability.workspace,
                            staged,
                            custom_tools=(
                                parsed.kind == "command"
                                or bool(capability.journal and capability.journal.view)
                            ),
                        )
                        if frozen_source is not None:
                            stage_frozen(staged, frozen_source)
                        if parsed.baseline_commit:
                            if not capability.journal or not capability.journal.view:
                                raise AgentRequestError(
                                    "ABBA baseline commits require a registered Episode Git workspace.",
                                    code="baseline_commit_unavailable",
                                    next_action=(
                                        "Run ABBA from an active optimization Episode and pass a full "
                                        "commit shown by local Git or a peer ref such as atrex/T2."
                                    ),
                                )
                            baseline_commit = parsed.baseline_commit
                            from aka.light.experiment.identifiers import KERNEL_RECORD_ID_RE
                            git = capability.journal.view.agent_git
                            if (
                                KERNEL_RECORD_ID_RE.fullmatch(baseline_commit) is None
                                or not git.commit_reachable(baseline_commit)
                            ):
                                raise AgentRequestError(
                                    "ABBA baseline commit is not reachable from local or shared Git refs.",
                                    code="baseline_commit_unavailable",
                                    baseline_kernel_commit_id=baseline_commit,
                                    next_action=(
                                        "Use a complete commit from git log, the current lineage, or "
                                        "a peer remote-tracking ref such as atrex/T2."
                                    ),
                                )
                            from aka.light.workspace.sources import Artifact, from_git
                            try:
                                baseline = Artifact(from_git(git.workspace, baseline_commit))
                                baseline_path = ".atrex_long_horizon/abba-baseline"
                                baseline.export(staged / baseline_path)
                            except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
                                raise AgentRequestError(
                                    "ABBA baseline commit does not contain a valid Kernel source tree.",
                                    code="baseline_commit_invalid",
                                    baseline_kernel_commit_id=baseline_commit,
                                    next_action=(
                                        "Inspect the commit with Git and choose one containing the complete "
                                        "task-owned Kernel source."
                                    ),
                                ) from error
                            kernel_ids["baseline"] = baseline_commit
                            argv = materialized_baseline_argv(argv, baseline_path)
                            parsed = parse_gateway(argv)
                        # Freeze the exact Agent-visible files before trusted
                        # evaluator inputs and Runtime control files are added.
                        # --kind command consumes this allowlist instead of
                        # traversing the later, privileged staging directory.
                        if parsed.kind == "command":
                            command_workspace = sorted(
                                path.relative_to(staged).as_posix()
                                for path in staged.rglob("*")
                                if path.is_file() and not path.is_symlink()
                            )
                            (staged / COMMAND_WORKSPACE_MANIFEST).write_text(
                                json.dumps(command_workspace), encoding="utf-8"
                            )
                        if capability.journal and capability.journal.view:
                            from aka.light.workspace.episode import PUBLIC_FILES
                            # An Agent draft is never the authority for evaluator inputs.
                            for name in PUBLIC_FILES:
                                if layout_at(staged).owns(name):
                                    continue
                                try:
                                    content = read_input(capability.journal.view.source_workspace, name)
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
                                kernel_id = kernel_ids.get("candidate")
                                if kernel_id is None:
                                    from aka.light.workspace.agent_git import (
                                        AgentGit,
                                        DirtyKernelWorktree,
                                    )
                                    git = (
                                        capability.journal.view.agent_git
                                        if capability.journal and capability.journal.view
                                        else AgentGit(capability.workspace)
                                    )
                                    try:
                                        kernel_id = git.kernel_commit(source)
                                    except DirtyKernelWorktree as error:
                                        raise AgentRequestError(
                                            str(error),
                                            code="dirty_kernel_worktree",
                                            changed_paths=list(error.paths),
                                            next_action=(
                                                "Review git diff, then commit the complete Kernel source "
                                                "tree. Retry the same Runtime request only after git status "
                                                "is clean; the HEAD commit is kernel_commit_id."
                                            ),
                                        ) from error
                                else:
                                    from aka.light.experiment.identifiers import KERNEL_RECORD_ID_RE
                                    if KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None:
                                        raise RuntimeStateError(
                                            "Trusted Kernel identity is not a complete Git commit ID"
                                        )
                                kernel_ids["candidate"] = kernel_id
                                if not prepare_identity_only:
                                    self.measurements.kernel(source, kernel_id)
                                    # The immutable Kernel archive is the
                                    # authoritative review input. Do not write
                                    # Supervisor request metadata into Agent Git.
                    (staged / ".orchestrator_mode.json").write_text(json.dumps({
                        "mode": self.config.optimization_mode,
                        "evaluator_shape_visibility": (
                            "hidden" if self.config.private_reference_dir else "exposed"
                        ),
                    }))
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
                    evaluator_reference_dir = (
                        reference_snapshot
                        or self.config.evaluator_reference_dir
                        or self.config.private_reference_dir
                    )
                    if evaluator_reference_dir:
                        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(
                            evaluator_reference_dir
                        )
                    # Keep the legacy report compiler's log alongside private records.
                    (staged / ".atrex_long_horizon").mkdir(exist_ok=True)
                    (staged / ".atrex_long_horizon/journal.json").write_text("{}")
                    if parsed is not None:
                        for value in requested_sync_paths(parsed):
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
                if "candidate" not in kernel_ids:
                    # Standalone capabilities do not own an EpisodeWorkspace/AgentGit
                    # view, but a previously registered exact source still has a
                    # stable commit identity. Resolve it before the nonblocking
                    # running-task probe so its request hash matches execution.
                    from aka.light.workspace.sources import snapshot as source_snapshot
                    try:
                        kernel_ids["candidate"] = self.measurements.find_kernel_id(
                            source_snapshot(staged)
                        )
                    except (FileNotFoundError, ValueError):
                        pass
                if effective.baseline_path and "baseline" not in kernel_ids:
                    raise RuntimeStateError(
                        "Runtime materialized an ABBA baseline without preserving its Git identity"
                    )
                if prepare_identity_only:
                    return self.experiments.task_identity(
                        staged, effective, environment, mode=self.config.optimization_mode,
                        repetitions=self.measurement_repetitions,
                        kernel_ids=kernel_ids,
                    )
                effects = MeasurementEffects(
                    mode=self.config.optimization_mode, repetitions=self.measurement_repetitions,
                    evaluator_reference_dir=self.config.evaluator_reference_dir,
                    private_reference_dir=self.config.private_reference_dir,
                    private_paths=(str(self.root), str(ROOT),
                                   str(self.config.evaluator_reference_dir or ""),
                                   str(self.config.private_reference_dir or ""),
                                   self.config.url, str(self.config.atrex_bench_root or "")),
                    run=lambda command, cwd, env: self.run_executor(command, cwd, env, capability),
                    audit=lambda argv, process, record: self.audit_process(capability, "gateway", argv, process, record),
                    publish_log=lambda staged: self.publish_evaluation_log(capability.workspace, staged),
                    publish_outputs=lambda staged, args: self.publish_legacy_outputs(capability.workspace, staged, args),
                    deadline=capability.deadline, request_id=capability.request_id,
                    kernel_ids=kernel_ids,
                )
                result = self.experiments.execute(
                    staged,
                    effective,
                    argv,
                    environment,
                    command,
                    effects=effects,
                    reuse_completed=reuse_completed,
                    reuse_correctness=reuse_correctness,
                )
                if capability.journal is not None:
                    reused_record_id = None
                    try:
                        agent_result = result.get("agent_result")
                        if isinstance(agent_result, Mapping):
                            capability.journal.service.observe_measurement(agent_result)
                        record_id = (
                            agent_result.get("gateway_record_id")
                            if isinstance(agent_result, Mapping)
                            else None
                        )
                        if result.get("reused") is True and isinstance(record_id, str):
                            reused_record_id = record_id
                        if (
                            isinstance(record_id, str)
                            and result.get("reused") is not True
                            and isinstance(agent_result, Mapping)
                        ):
                            record = self.measurements.read(record_id)
                            capability.journal.service.record_passing_full_evaluate_checkpoint(
                                record
                            )
                        capability.journal.service.publish_research_files(
                            synchronize=False,
                            direction_ids=frozenset(),
                            experiment_ids=frozenset(),
                            record_ids=(
                                frozenset({record_id})
                                if isinstance(record_id, str)
                                else frozenset()
                            ),
                        )
                        if isinstance(agent_result, dict):
                            hint = capability.journal.service.experiment_recording_hint(
                                agent_result
                            )
                            if hint is not None:
                                agent_result["research_next_action"] = hint
                    except (OSError, ValueError, RuntimeStateError) as error:
                        if isinstance(reused_record_id, str):
                            raise RuntimeStateError(
                                "The exact cached Gateway Record was found, but its "
                                "lineage projection could not be published.",
                                code="cached_record_projection_failed",
                                next_action=(
                                    "Retry the same Runtime request after Supervisor "
                                    "storage recovers. The cached result will be reused; "
                                    "do not change the Kernel or submit a replacement job."
                                ),
                            ) from error
                        # The immutable measurement has already completed. A
                        # projection failure must not turn it into an unknown
                        # GPU outcome; registration or the next mutation heals
                        # PROGRESS.md from authoritative stores.
                        logging.getLogger(__name__).exception(
                            "Failed to refresh PROGRESS.md after measurement"
                        )
                return result
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
            from aka.light.evaluation.projection import credential_values, project_response
            response = project_response(process, generalized=self.config.private_reference_dir is not None,
                                    operation=parsed.kind if kind == "gateway" and parsed is not None else None,
                                    private_values=credential_values(environment),
                                    wiki=kind == "wiki", private_paths=(str(staged), str(self.root),
                                    str(self.config.evaluator_reference_dir or ""),
                                    str(self.config.private_reference_dir or ""), str(ROOT),
                                    self.config.url, str(self.config.atrex_bench_root or "")))
            if kind == "gateway" and parsed is not None and parsed.kind == "env":
                from aka.light.evaluation.public_result import seal_response
                return seal_response(response, {"operation": "env"})
            return response

    def execute_journal(self, capability: Capability, request: dict) -> dict:
        binding = capability.journal
        if binding is None:
            raise AgentRequestError("Research/report mutation tools require a controller-registered Long Horizon Episode", repairable=False,
                                    code="journal_unavailable", next_action="Use this session's Baseline finish procedure; do not supply paths or Episode IDs to create private Runtime state")
        self._acquire(binding.lock, capability, capability.deadline)
        try:
            if not self._live(capability):
                raise SessionRevokedError()
            try:
                agent_git = None
                if request.get("operation") == "kernel_submit":
                    from aka.light.workspace.agent_git import AgentGit
                    agent_git = (
                        binding.view.agent_git
                        if binding.view is not None
                        else AgentGit(binding.service.workspace)
                    )
                value = binding.service.execute(
                    request,
                    review_kernel=(
                        lambda commit_id, submission_id: self._review_submitted_kernel(
                            capability,
                            binding,
                            agent_git,
                            commit_id,
                            submission_id,
                        )
                    ) if agent_git is not None and request.get("operation") == "kernel_submit" else None,
                )
            except (AgentRequestError, RuntimeStateError):
                raise
            except ValueError as error:
                evidence_hint = _journal_evidence_hint(
                    self.journal_factory.tool_modules
                )
                raise AgentRequestError(str(error), next_action=(
                    "Correct the indicated field or lifecycle state, then retry. Read skills/runtime-records/references/journal.md. "
                    + evidence_hint)) from error
            response = {"exit_code": 0, "stdout": json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", "stderr": ""}
            self.audit_process(capability, "journal", [str(request.get("operation"))],
                               subprocess.CompletedProcess([], 0, response["stdout"], ""))
            return response
        finally:
            binding.lock.release()

    def _review_submitted_kernel(
        self,
        capability,
        binding,
        agent_git,
        commit_id: str,
        submission_id: str | None = None,
    ) -> dict:
        """Review one exact Agent commit and its saved full-Evaluate evidence."""
        from aka.light.workspace.sources import from_git

        if not agent_git.commit_reachable(commit_id):
            raise AgentRequestError(
                "kernel_commit_id is not reachable from current Agent or imported peer Git refs",
                code="kernel_commit_unreachable",
                kernel_commit_id=commit_id,
                next_action="Use a full commit ID shown by local git log or an imported peer ref.",
            )
        try:
            source = from_git(agent_git.workspace, commit_id)
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            raise AgentRequestError(
                "kernel_commit_id does not contain one complete candidate source tree",
                code="kernel_source_invalid",
                kernel_commit_id=commit_id,
                next_action="Commit a complete evaluator-compatible Kernel source tree and submit that commit.",
            ) from error

        policy_validator = getattr(self, "production_policy_validator", None)
        production_applicable = bool(getattr(self, "production_review_required", False))
        violations: list[str] = []
        implementation_review: dict[str, object] = {
            "availability": "not_applicable",
        }
        if production_applicable and policy_validator is not None:
            policy_result = policy_validator(
                binding.service.git_workspace,
                source,
                parent=capability,
                require_gluon=binding.require_gluon,
                submission_id=submission_id,
            )
            if isinstance(policy_result, Mapping):
                raw_violations = policy_result.get("violations", [])
                violations = (
                    list(raw_violations)
                    if isinstance(raw_violations, (list, tuple))
                    else ["production policy validator returned invalid violations"]
                )
                raw_review = policy_result.get("implementation_review")
                implementation_review = (
                    dict(raw_review)
                    if isinstance(raw_review, Mapping)
                    else {"availability": "unavailable"}
                )
            else:
                violations = list(policy_result)
                implementation_review = {"availability": "unavailable"}
        review = {"source_contract": "pass"}
        if production_applicable:
            from aka.light.acceptance.policy_modes import review_fields

            verdict = "fail" if violations else "pass"
            review.update({
                name: verdict
                for name in review_fields(
                    self.config.optimization_mode,
                    specify_dsl=bool(
                        getattr(self, "submission_specify_dsl", True)
                    ),
                )
            })
        evaluation_evidence = kernel_full_evaluation_evidence(
            self.measurements,
            commit_id,
        )
        return {
            "accepted": (
                not violations
                and evaluation_evidence.get("status") == "passed"
            ),
            "review": review,
            "implementation_review": implementation_review,
            "diagnostics": violations,
            "evaluation_evidence": evaluation_evidence,
        }

    def audit_process(self, capability, kind, argv, process, record_id=None):
        if self.audit_root:
            request_id = capability.request_id or uuid.uuid4().hex
            path = self._request_audit_path(request_id)
            existing = self._read_request_audit(path)
            now = datetime.now(timezone.utc).isoformat()
            lifecycle_owned = bool(existing)
            existing.update({
                "schema_version": 2,
                "request_id": request_id,
                "created_at": existing.get("started_at", existing.get("created_at", now)),
                "started_at": existing.get("started_at", existing.get("created_at", now)),
                "updated_at": now,
                "operation": kind,
                "argv": self._audited_argv(kind, argv),
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "gateway_record_id": record_id or existing.get("gateway_record_id"),
                "capture_limits": {
                    "stdout_bytes": 4 * 1024 * 1024,
                    "stderr_bytes": 128 * 1024,
                },
            })
            if not lifecycle_owned:
                existing.update({
                    "state": "succeeded" if process.returncode == 0 else "failed",
                    "finished_at": now,
                })
            self._publish_request_audit(path, existing)

    @staticmethod
    def _audited_argv(kind: str, argv) -> list[str]:
        audited = [str(value) for value in argv]
        if kind != "gateway":
            return audited
        result = []
        redact_next = False
        for value in audited:
            if redact_next:
                result.append("<redacted>")
                redact_next = False
                continue
            option = value.split("=", 1)[0]
            if option in {"--env", "--ssh-init", "--url"}:
                result.append(option + "=<redacted>" if "=" in value else option)
                redact_next = "=" not in value
            else:
                result.append(value)
        return result

    @staticmethod
    def _read_request_audit(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _publish_request_audit(path: Path, value: dict) -> None:
        publish(
            path.parent,
            path.name,
            (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode(),
        )

    def _request_audit_path(self, request_id: str, route: str | None = None) -> Path:
        if not self.audit_root:
            return self.root / "runtime-audit" / "requests" / f"request-{request_id}.json"
        runtime_audit = self.audit_root / "runtime-audit"
        if route == "/v1/plugins/execute":
            return runtime_audit / "plugin-calls" / f"request-{request_id}.json"
        if route is None:
            for collection in ("requests", "plugin-calls"):
                candidate = runtime_audit / collection / f"request-{request_id}.json"
                if candidate.is_file():
                    return candidate
        return runtime_audit / "requests" / f"request-{request_id}.json"

    def audit_request_started(self, request_id: str, *, route: str) -> None:
        if not self.audit_root:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._publish_request_audit(
            self._request_audit_path(request_id, route),
            {
                "schema_version": 2,
                "request_id": request_id,
                "route": route,
                "state": "running",
                "created_at": now,
                "started_at": now,
                "updated_at": now,
            },
        )

    def audit_request_details(self, request_id: str, *, route: str, request: dict) -> None:
        if not self.audit_root:
            return
        path = self._request_audit_path(request_id, route)
        value = self._read_request_audit(path)
        operation = {
            "/v1/gateway/execute": "gateway",
            "/v1/wiki/query": "wiki",
            "/v1/journal/execute": "journal",
            "/v1/plugins/execute": "plugin",
        }.get(route, "unknown")
        argv = request.get("argv") if isinstance(request.get("argv"), list) else []
        if operation == "gateway":
            audited_argv = self._audited_argv(operation, argv)
        else:
            audited_argv = []
        tool = None
        if operation == "gateway":
            for index, item in enumerate(audited_argv):
                if item == "--kind" and index + 1 < len(audited_argv):
                    tool = audited_argv[index + 1]
                    break
                if item.startswith("--kind="):
                    tool = item.split("=", 1)[1]
                    break
        elif operation == "wiki":
            tool = request.get("tool") or "wiki-query"
        elif operation == "journal":
            tool = request.get("operation")
        elif operation == "plugin":
            tool = request.get("tool")
        value.update({
            "route": route,
            "operation": operation,
            "tool": tool or operation,
            "argv": audited_argv,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._publish_request_audit(path, value)

    def audit_request_finished(
        self,
        request_id: str,
        *,
        http_status: int,
        result: object,
        execution_exit_code: int | None,
        duration_seconds: float,
    ) -> None:
        if not self.audit_root:
            return
        path = self._request_audit_path(request_id)
        value = self._read_request_audit(path)
        now = datetime.now(timezone.utc).isoformat()
        response = result if isinstance(result, dict) else {}
        response_status = str(response.get("status") or response.get("outcome") or "").casefold()
        pending = response_status in {
            "accepted", "pending", "queued", "running", "submitted", "waiting",
        }
        failed = http_status >= 400 or (
            type(execution_exit_code) is int and execution_exit_code != 0
        )
        error = {
            key: response.get(key)
            for key in ("code", "reason", "error_class", "message", "outcome", "phase")
            if response.get(key) is not None
        }
        gateway_record_id = response.get("gateway_record_id")
        if not isinstance(gateway_record_id, str):
            details = response.get("details")
            gateway_record_id = (
                details.get("gateway_record_id") if isinstance(details, dict) else None
            )
        value.update({
            "state": "failed" if failed else "pending" if pending else "succeeded",
            "updated_at": now,
            "finished_at": now,
            "duration_seconds": duration_seconds,
            "http_status": http_status,
            "execution_exit_code": execution_exit_code,
            "error": error or None,
        })
        if isinstance(gateway_record_id, str) and gateway_record_id:
            value["gateway_record_id"] = gateway_record_id
        self._publish_request_audit(path, value)

    def execute_plugin(self, capability: Capability, request: dict) -> dict:
        with _validating_request():
            if request == {"action": "list"}:
                return {"exit_code": 0, "stdout": json.dumps(self.plugins.public_catalog()) + "\n", "stderr": ""}
            if set(request) != {"action", "tool", "input"} or request["action"] != "call":
                raise ValueError("Use action=list, or action=call with tool and input")
            if not isinstance(request["tool"], str):
                raise ValueError("tool must be a catalog name")
            tool = self.plugins.validate_call(request["tool"], request["input"])
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
        if tool.get("runtime_tool") == "wiki":
            # Providers receive the same bounded Supervisor environment as subprocess
            # plugins, plus the explicitly selected Wiki model route. GPU, store and
            # private-reference authority never crosses this seam.
            environment = scrub_environment(self.environment)
            environment.update(capability.context)
            for name in ("ATREX_ENVIRONMENT_RESTART_HANDOFF_ID", "ATREX_ENVIRONMENT_RESTART_LOCK_FD",
                         "ATREX_ENVIRONMENT_STATE_FILE"):
                environment.pop(name, None)
            if self.config.wiki_profile_root:
                environment["ATREX_WIKI_PROFILE_ROOT"] = str(self.config.wiki_profile_root)
            environment["ATREX_WIKI_TASK_ID"] = self.config.task_id
            from aka.light.evaluation.projection import credential_values, redact_diagnostics
            from aka.light.wiki.worker import provider_environment
            try:
                bridge_values = provider_environment(self.config.wiki_options, environment)
            except ValueError as error:
                raise RuntimeStateError(
                    "Wiki provider settings are invalid or the credential endpoint does not match.",
                    code="wiki_configuration_invalid",
                ) from error
            environment.update(bridge_values)

            with tempfile.TemporaryDirectory(dir=self.root, prefix="wiki-") as temporary:
                private_paths = (
                    str(temporary), str(self.root), str(self.audit_root or ""),
                    str(self.config.evaluator_reference_dir or ""),
                    str(self.config.private_reference_dir or ""),
                    str(self.config.wiki_profile_root or ""),
                    self.config.wiki_options.settings_file, self.config.wiki_options.base_url,
                )
                private_values = credential_values(environment)

                def redact(value):
                    return redact_diagnostics(
                        value, private_paths=private_paths, private_values=private_values,
                    )

                def project(result):
                    # Redact and validate before the result can become reusable evidence.
                    # Output drift is a provider failure, never an Agent-repairable input error.
                    result = redact(result)
                    return json.loads(project_plugin_output(
                        request["tool"], tool["output_schema"],
                        json.dumps(result, ensure_ascii=False, allow_nan=False),
                    ))

                result = self.wiki.invoke(
                    self, capability, request["tool"].partition(".")[2],
                    request["input"], Path(temporary), environment, project=project,
                    redact_error=redact,
                )
            return {"exit_code": 0, "stdout": json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
                    "stderr": ""}
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
                [sys.executable, str(ROOT / "runtime/plugin_executor.py"), str(request_path)],
                staged, environment, capability,
            )
            self.audit_process(capability, "plugin", [request["tool"]], process)
            if process.returncode:
                check_executor_output_failure(request["tool"], process.returncode, process.stdout)
                # Plugin errors can contain private resources or evaluator paths.
                raise RuntimeError("Plugin execution failed; inspect private request diagnostics")
            rendered = project_plugin_output(request["tool"], tool["output_schema"], process.stdout)
            return {"exit_code": 0, "stdout": rendered, "stderr": ""}

    def _wiki_command(self, capability, tool, argv, staged, environment):
        if not self.config.wiki_enabled:
            raise AgentRequestError("GPU Wiki is disabled in the selected Workflow configuration",
                                    code="wiki_disabled", repairable=False,
                                    next_action="Continue with the enabled tools. Only the operator can enable Wiki; do not retry or access its private store directly.")
        cli_root = self.wiki_provider.cli_root
        if cli_root is None:
            raise AgentRequestError("The selected Wiki provider has no command-line Wiki tools",
                                    code="wiki_operation_unsupported", repairable=False,
                                    next_action="List the provider's Wiki operations with `python3 tools/plugin.py list` "
                                                "and call them with `python3 tools/plugin.py call`.")
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
                try:
                    data = read_input(capability.workspace, raw, limit=128 * 1024)
                except FileNotFoundError as error:
                    from aka.light.runtime.request_errors import field_error
                    raise field_error(
                        "--file",
                        f"Wiki query input file does not exist: {raw}",
                        code="file_not_found",
                        requested_path=raw,
                        next_action=(
                            "Create the named file inside the workspace or pass the query text "
                            "directly, then retry the same Wiki operation."
                        ),
                    ) from error
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
        return [sys.executable, str(Path(cli_root) / f"{tool}.py"), *argv]

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
            for value in requested_sync_paths(args):
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
            from aka.light.agent.session_tail import read_regular_bytes
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
