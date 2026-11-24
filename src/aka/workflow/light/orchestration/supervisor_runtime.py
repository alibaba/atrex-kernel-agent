"""Campaign Runtime over shared Agent, Execution, Evaluation and Resources owners."""
from __future__ import annotations

from aka.contracts.evaluation import derived_evaluation_timeout
from aka.contracts.agent import AgentToolSession

import json
import argparse
import hashlib
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from contextlib import ExitStack, contextmanager
from collections.abc import Mapping
from aka.contracts.evaluation import DEFAULT_CANDIDATE_TIMEOUT as DEFAULT_EVALUATION_STEP_TIMEOUT, DEFAULT_SHAPE_BATCH_SIZE as SHAPE_BATCH_SIZE
from dataclasses import dataclass, field
from pathlib import Path

from aka.workflow.light.host.workspace import (
    MAX_FILE_BYTES, MAX_TOTAL_BYTES, InputSizeLimitError, publish, read_input, relative_path, snapshot,
)
from aka.workflow.light.host.journal import SupervisorJournalService
from aka.workflow.light.host.errors import AgentRequestError, RuntimeStateError, error_response
from aka.workflow.light.host.request_feedback import augment_error
from aka.workflow.light.host.plugin_output import (
    MAX_OUTPUT_BYTES, raise_output_error,
)
from .constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_HARNESS
from .episode_workspace import EpisodeWorkspace
from aka.experiment.frontier import kernel_full_evaluation_evidence
from aka.experiment.measurement_store import MeasurementWaitTimeout
from aka.workflow.light.support.errors import (
    RequestValidationError, RequestDispatchTimeout, SessionRevokedError,
)
from aka.workflow.light.support.paths import (supervisor_campaign_root, supervisor_measurement_root,
    supervisor_submission_root, supervisor_research_catalog_root)
from aka.resources.wiki.config import WikiConfig
from aka.workflow.light.host.wiki import WikiService, WikiUnavailable

OWNER_ENV = "ATREX_AKA_RUNTIME_OWNER"
URL_ENV = "ATREX_AKA_RUNTIME_URL"
TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"

MAX_REQUEST_BYTES = 2 * 1024 * 1024
DISPATCH_RETRY_SECONDS = 5
from importlib.resources import files

ROOT = Path(str(files("aka.workflow.light"))).resolve()


AUTHORITY = frozenset({
    "--workspace", "--hardware", "--url", "--gateway-profile", "--ssh", "--ssh-init",
    "--ssh-gpu", "--ssh-runtime-bind", "--health-command", "--runtime-health-command",
    "--timeout", "--max-input-file-mb", "--max-output-file-mb",
})
FORBIDDEN = frozenset({"--preflight", "--check-health"})
WIKI_STORES = {"query_nl": "--store-root", "query_wiki": "--json-store", "query_hardware": "--store"}


def normalize_wiki_argv(tool: str, argv: list[str]) -> list[str]:
    """Accept wrapper syntax before validating Supervisor-owned options."""
    argv = list(argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    # A quoted option string is a common wrapper mistake. Never split prose or
    # individually supplied option values (product names and query text may contain spaces).
    if tool in {"query_hardware", "query_wiki"} and len(argv) == 1 and argv[0].startswith("-"):
        argv = shlex.split(argv[0])
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if "--" in argv:
        raise ValueError("Use '--' only before the Wiki arguments, not between them")
    if tool == "query_hardware":
        if not argv:
            return ["--list", "products"]
        if not argv[0].startswith("-"):
            argv = ["--product", *argv]
    return argv or ["--help"]


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


def failure(message: str, *, repairable: bool = True, next_action: str | None = None,
            code: str | None = None) -> dict:
    from aka.workflow.light.host.errors import UNKNOWN_OUTCOME
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
            and key not in {OWNER_ENV, URL_ENV, TOKEN_ENV, "ATREX_AKA_MEASUREMENT_JOB_ROOT", "ATREX_AKA_MEASUREMENT_REPETITIONS",
                            "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_BENCH_RUNTIME_ROOT",
                            "ATREX_AKA_IDENTITY_KEY"}}


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
    wiki_reference_paths: tuple[tuple[str, str], ...] = ()
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
    pause_agent_during_requests: bool = False
    research_synchronization: str = "independent"
    agate_tools: tuple[str, ...] = ("env", "evaluate", "command")
    wiki_enabled: bool = True
    wiki_options: WikiConfig = field(default_factory=WikiConfig)
    # Selected aka.contracts.wiki.WikiProvider; None selects the built-in local store.
    wiki_provider: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        from aka.contracts.agate_tools import normalize_agate_tools
        object.__setattr__(self, "agate_tools", normalize_agate_tools(self.agate_tools))
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
    # Request-local: handlers derive their own capability from the Session grant.
    deadline: float | None = field(default=None, repr=False)
    request_id: str = ""
    journal: JournalBinding | None = None
    parent: Capability | None = field(default=None, repr=False)
    # Public owner grant; process ownership stays in the shared Agent layer.
    agent_session: AgentToolSession | None = field(default=None, repr=False)
    cancel_event: object | None = field(default=None, repr=False)
    filesystem_access: object | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _ToolGrant:
    environment: object = field(repr=False)
    filesystem_access: object


class _OperationCancellation:
    def __init__(self, owner, capability):
        self.owner, self.capability = owner, capability

    def is_set(self):
        return (not self.owner._live(self.capability) or
            self.capability.deadline is not None and time.monotonic() >= self.capability.deadline)


def agent_gateway_response(result: dict) -> tuple[dict, int | None]:
    """Return the full canonical result and its executor exit classification.

    Gateway observations, including Env, are archived by the Runtime and carry
    a Record path. The operation handler redacts the final public response.
    """
    from aka.evaluation.tool_results import compact_receipt
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        return result, None
    return compact_receipt(agent_result), result["exit_code"]

def _runtime_operation(route: str, request: dict) -> str:
    if route == "/v1/gateway/execute":
        argv = request.get("argv", [])
        if not isinstance(argv, list):
            return "gateway"
        words = argv[:argv.index("--")] if "--" in argv else argv
        kind = "auto"
        for index, word in enumerate(words):
            if word == "--kind" and index + 1 < len(words):
                kind = words[index + 1]
            elif isinstance(word, str) and word.startswith("--kind="):
                kind = word.split("=", 1)[1]
        if not isinstance(kind, str):
            return "gateway"
        if kind in {"auto", "evaluate"}:
            if any(isinstance(word, str) and word.split("=", 1)[0] == "--baseline-commit" for word in words):
                return "same_allocation_abba"
            return "evaluate"
        return kind if kind in {"profile", "dev", "command", "check", "disassemble", "env"} else "gateway"
    return {
        "/v1/journal/execute": str(request.get("operation") or "journal"),
        "/v1/wiki/query": str(request.get("tool") or "wiki"),
        "/v1/plugins/execute": str(request.get("tool") or request.get("action") or "plugin"),
    }.get(route, "runtime")

def agent_runtime_response(route: str, request: dict, result: dict) -> tuple[dict, int | None]:
    """Expose one canonical receipt contract for every Agent Runtime operation."""
    if route == "/v1/gateway/execute" and isinstance(result.get("agent_result"), dict):
        return agent_gateway_response(result)
    if (
        isinstance(result.get("operation"), str)
        and result.get("status") in {"completed", "failed", "cancelled"}
        and isinstance(result.get("result"), dict)
    ):
        return result, None
    if type(result.get("exit_code")) is not int:
        return result, None
    operation = _runtime_operation(route, request)
    exit_code = result["exit_code"]
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    detail: dict[str, object] = {}
    if isinstance(stdout, str) and stdout:
        try:
            value = json.loads(stdout)
        except (ValueError, TypeError):
            detail["stdout"] = stdout
        else:
            detail.update(value if isinstance(value, dict) else {"value": value})
    if isinstance(stderr, str) and stderr:
        detail["stderr"] = stderr
    if exit_code:
        detail["exit_code"] = exit_code
        if "error" not in detail:
            detail["error"] = {
                "code": "runtime_operation_failed",
                "message": stderr.strip() if isinstance(stderr, str) and stderr.strip()
                else f"Runtime operation exited with code {exit_code}",
            }
    return {
        "operation": operation,
        "status": "completed" if exit_code == 0 else "failed",
        "result": detail,
    }, exit_code


class _OperationHandler:
    """Workflow operation adapter; the selected Agent owns transport authority."""
    def __init__(self, owner, capability, route):
        self.owner, self.capability, self.route = owner, capability, route

    def handle(self, payload, context):
        from functools import partial
        from aka.core.service_audit import capture_service_exchanges
        request_id = uuid.uuid4().hex
        with capture_service_exchanges(partial(self.owner._record_service_exchange, request_id)):
            return self._handle(payload, context, request_id)

    def _handle(self, payload, context, request_id):
        from dataclasses import replace
        owner = self.owner
        capability = replace(self.capability, parent=self.capability,
                             cancel_event=context.cancel_event, filesystem_access=context.filesystem_access)
        request, status, exit_code = None, 200, None
        started = time.monotonic()
        capability.request_id = request_id
        owner.audit_request_started(request_id, route=self.route)
        try:
            if context.workspace != capability.workspace:
                raise SessionRevokedError("Tool workspace differs from its binding")
            with _validating_request():
                if len(payload) > MAX_REQUEST_BYTES:
                    raise ValueError("Request exceeds 2 MiB")
                request = json.loads(payload)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
            owner.audit_request_details(request_id, route=self.route, request=request)
            # Claim and final-source Reviews have no active-time deadline.
            deadline = (float("inf") if self.route == "/v1/journal/execute"
                        else time.monotonic() + owner.request_timeout_seconds)
            if context.deadline is not None:
                deadline = min(deadline, context.deadline)
            with owner.request_context(capability, deadline, request_id):
                result = owner.execute(capability, self.route, request)
            result, exit_code = agent_runtime_response(self.route, request, result)
        except AgentRequestError as error:
            status, result = 400, error.response
        except RuntimeStateError as error:
            status, result = 503, error.response
        except RequestValidationError as error:
            status, result = 400, failure(str(error))
        except RequestDispatchTimeout as error:
            status, result = 429, failure(str(error), code="request_not_started",
                next_action="No job was submitted. Retry the same request after bounded backoff.")
            result["retry_after_seconds"] = DISPATCH_RETRY_SECONDS
        except MeasurementWaitTimeout as error:
            status, result = 504, failure(str(error), code="measurement_wait_timeout",
                next_action="The original request may still finish. Retrying the same request will join it or read its result.")
        except SessionRevokedError:
            status, result = 401, failure("Tool binding revoked", repairable=False, code="session_revoked")
        except Exception as error:
            status, result = 503, failure(f"{type(error).__name__}: {error}", repairable=False, code="runtime_execution_error")
            logging.getLogger(__name__).exception("Workflow tool operation failed")
        if status >= 400:
            result = augment_error(result, route=self.route, request=request,
                research_records=owner.journal_factory.research_records,
                directions_enabled=owner.journal_factory.directions_enabled,
                experiments_enabled=owner.journal_factory.experiments_enabled,
                enabled_kinds=owner.config.agate_tools,
                parser=None,
                catalog=owner.plugins.catalog() if self.route == "/v1/plugins/execute" and status == 400 else ())
            result = {"operation": _runtime_operation(self.route, request or {}),
                      "status": "failed", "result": {"error": result}}
        if self.route == "/v1/gateway/execute" and isinstance(request, dict) and not result.get("gateway_record_id"):
            result = owner.archive_gateway_response(capability, request, result, status=status, exit_code=exit_code)
        # An upstream Agate error can be carried by a successful Runtime HTTP
        # exchange. Apply the same public boundary regardless of HTTP status;
        # private execution/measurement archives retain the original body.
        from aka.workflow.light.host.projection import credential_values, redact_diagnostics
        result = redact_diagnostics(result,
            private_paths=(str(owner.root), str(owner.audit_root or ""), str(ROOT),
                str(owner.config.private_reference_dir or ""), owner.config.url,
                str(owner.config.atrex_bench_root or "")),
            private_values=credential_values(owner.environment))
        owner.audit_request_finished(request_id, http_status=status, result=result,
            execution_exit_code=exit_code, duration_seconds=time.monotonic() - started)
        return json.dumps({"status": status, "result": result,
                           "execution_exit_code": exit_code}, ensure_ascii=False,
                          allow_nan=False).encode()


class SupervisorRuntime:
    def __init__(self, config: RuntimeConfig, *, workspace_provider=None, services=None):
        self.config = config
        if services is None or workspace_provider is None:
            raise TypeError("Workflow operations require explicitly selected shared modules")
        from aka.execution.gateway.native_options import audit_summary
        logging.getLogger(__name__).warning("Agate native option audit: %s", json.dumps(audit_summary(), sort_keys=True))
        self.agent, self.executor = services.agent, services.executor
        self.journal_factory = services.journal
        self.plugins = services.resources
        self.workspace_provider = workspace_provider
        self.evaluation_input, self.evaluation_configuration = services.evaluation_input, services.configuration
        self.evaluation_identity = getattr(services, "evaluation_identity", None)
        self.environment = dict(services.configuration.environment)
        from aka.execution.configuration import queue_wait_grace
        self.gateway_queue_wait_grace = 0 if config.ssh else queue_wait_grace(self.environment)
        if self.gateway_queue_wait_grace is not None:
            self.environment["ATREX_SANDBOX_QUEUE_WAIT_GRACE"] = str(self.gateway_queue_wait_grace)
        self.measurement_repetitions = config.measurement_repetitions
        if self.measurement_repetitions not in {1, 3}:
            raise ValueError("measurement_repetitions must be 1 or 3")
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
            from aka.core.filesystem import open_private_directory
            with open_private_directory(self.audit_root):
                pass
        from aka.workflow.light.measurements import SharedMeasurements
        from aka.experiment.measurement_store import MeasurementStore
        self.measurement_root = config.measurement_root or supervisor_measurement_root(config.workspace or self.root)
        self.measurements = MeasurementStore(self.measurement_root)
        self.experiments = SharedMeasurements(factory=services.experiments, executor=self.executor,
            evaluator=services.evaluator, namespace_workspace=config.workspace or self.root,
            evaluation_input=self.evaluation_input, workspace_provider=workspace_provider,
            command_evaluator=services.command_evaluator, store=self.measurements)
        # Resources owns both the selection identity and its validation. Workflow
        # stores the opaque value without interpreting plugin or Skill locks.
        resource_lock = (self.audit_root or self.root) / "resources.json"
        if resource_lock.exists():
            self.plugins.check_resource_snapshot(json.loads(read_input(resource_lock.parent, resource_lock.name)))
        else:
            publish(resource_lock.parent, resource_lock.name, json.dumps(self.plugins.resource_snapshot()).encode())
        self.wiki = WikiService((self.audit_root or self.root) / "wiki", config.wiki_options,
                                self.plugins.resource_snapshot(), provider=getattr(self.plugins, "wiki_provider", None))
        # Wiki workers execute trusted host resources, independently of the
        # selected GPU transport. Keep the caller's filesystem scope enforced
        # by the local Execution plugin rather than dropping it for Gateway.
        from aka.execution.assembly import compose_execution
        self._wiki_execution = compose_execution(profile="local", patches=({
            "id": "executor", "config": {"proc_mode": self.agent.capabilities.proc_mode},
        },))
        self.wiki_executor = self._wiki_execution.executor
        if config.workspace:
            from aka.workflow.light.support.paths import record_campaign_runtime_configuration
            record_campaign_runtime_configuration(config.workspace, {
                "hardware": config.hardware, "optimization_mode": config.optimization_mode,
                "transport": "ssh" if config.ssh else "agate_sdk",
                "evaluation_timeout_seconds": config.timeout,
                "evaluation_step_timeout_seconds": config.evaluation_step_timeout,
                "warmup_iters": services.configuration.warmup_iters,
                "bench_iters": services.configuration.bench_iters,
                "shape_batch_size": services.configuration.shape_batch_size,
                "agate_tools": list(config.agate_tools),
                "request_timeout_seconds": self.request_timeout_seconds,
                "queue_timeout_seconds": config.queue_timeout,
                "measurement_repetitions": config.measurement_repetitions,
                "pause_agent_during_requests": config.pause_agent_during_requests,
                "research_synchronization": config.research_synchronization,
            })


    def archive_gateway_response(self, capability, request, result, *, status, exit_code):
        """Retain Env observations and pre-execution failures in the central Store."""
        from aka.evaluation.tool_requests import parse_tool_arguments
        try:
            args = parse_tool_arguments(request.get("argv", []))
        except (ValueError, TypeError):
            return result  # No recognized measurement operation was requested.
        if "--help" in request.get("argv", []) or "-h" in request.get("argv", []):
            return result
        kind = args.kind
        operation = "evaluate" if kind in {"auto", "evaluate"} else kind
        identity = {"operation": operation, "runtime_request_id": capability.request_id,
                    "options": {"argv": request.get("argv", [])}, "hardware": self.config.hardware}
        response = {"exit_code": exit_code if exit_code is not None else (0 if status < 400 else 2),
                    "stdout": result.get("stdout", ""), "stderr": result.get("stderr", ""),
                    "agent_result": result, "http_status": status}
        with self.measurements.reserve(identity, {}, request_id=capability.request_id or "", reusable=False) as task:
            returned = dict(result, gateway_record_id=task.record_id,
                            record_path=f"records/{task.record_id}.json")
            response["agent_result"] = returned
            task.finish(response, cacheable=False)
        if capability.journal:
            service = capability.journal.service
            service.observe_measurement(returned)
            service.publish_research_files(synchronize=False, direction_ids=frozenset(),
                experiment_ids=frozenset(), record_ids=frozenset({task.record_id}))
        return returned

    @property
    def request_timeout_seconds(self) -> float:
        if self.config.request_timeout is not None:
            return self.config.request_timeout
        # Preserve this Workflow request deadline independently of Execution waiting.
        grace = 14_400 if self.gateway_queue_wait_grace is None else self.gateway_queue_wait_grace
        return 2 * (self.config.timeout + grace) + 120

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            capabilities = list(self.capabilities.values())
            self.capabilities.clear()
        # Attempt every cancellation even if one process or filesystem cleanup
        # fails, and revoke Agent work before removing its Runtime staging area.
        with ExitStack() as cleanup:
            cleanup.callback(self._wiki_execution.dispose)
            cleanup.callback(self.temporary.cleanup)
            for capability in capabilities:
                if capability.agent_session is not None:
                    cleanup.callback(capability.agent_session.cancel)

    def revoke(self, token: str):
        with self.lock:
            capability = self.capabilities.pop(token, None)
        if capability and capability.agent_session is not None:
            capability.agent_session.cancel()



    def register_episode(self, workspace, *, episode, base_commit, branch, memory_version=1,
                         baseline_kernel_commit_id=None, managed_agent_workspace=True,
                         require_gluon=False, session_capture_root=None, origin=None):
        workspace = Path(workspace).resolve(strict=True)
        if type(episode) is not int or episode < 1 or not branch:
            raise ValueError("Invalid Episode audit segment")
        from aka.task.candidate_workspace.agent_git import AgentGit
        from aka.task.source.formats.storage import from_git
        git = AgentGit(workspace)
        baseline = baseline_kernel_commit_id or base_commit
        if not git.commit_reachable(baseline):
            raise ValueError("Episode starting commit is not reachable in Agent Git")
        self.measurements.kernel(from_git(workspace, baseline), baseline)
        with self.lock:
            if self.closed:
                raise RuntimeError("Supervisor Runtime is closed")
            previous = self.journals.get(workspace)
            if previous and previous.service.episode == episode:
                return workspace
            if any(cap.workspace == workspace for cap in self.capabilities.values()):
                raise RuntimeError("Cannot advance an active Session's Episode binding")
            view = EpisodeWorkspace(workspace, research_records=self.journal_factory.research_records,
                research_modules=self.journal_factory.tool_modules, resources_registry=self.plugins,
                workspace_policy=self.config.workspace_policy)
            view.prepare()
            lineage_root = self.audit_root or self.root
            binding = JournalBinding(self.journal_factory.open(workspace=workspace, git_workspace=workspace,
                campaign_root=lineage_root / "research", evidence_root=supervisor_research_catalog_root(workspace) / "reviews",
                measurement_root=self.measurement_root, submissions_root=supervisor_submission_root(workspace),
                episode=episode, base_commit=base_commit, base_kernel_id=baseline, episode_branch=branch,
                tool_modules=self.journal_factory.tool_modules, research_root=lineage_root / "research",
                research_synchronization=self.config.research_synchronization,
                session_capture_root=session_capture_root, origin=origin,
                claim_review_enabled=getattr(self, "claim_review_enabled", False),
                claim_reviewer=getattr(self, "research_claim_reviewer", None)),
                view=view, require_gluon=require_gluon, baseline_kernel_commit_id=baseline)
            self.journals[workspace] = binding
            binding.service.publish_research_files()
            self._write_binding_lineage_state(binding, segment_status="idle")
        return workspace

    def available_tools(self, capability):
        from aka.contracts.runtime_tools import runtime_tools
        binding = getattr(capability, "journal", None)
        return runtime_tools(self.config.agate_tools,
            research_modules=binding.service.tool_modules if binding else (),
            submissions=binding is not None, wiki=getattr(self.config, "wiki_enabled", False))

    @contextmanager
    def session(self, workspace: Path, environment: dict[str, str], *, access, deadline=None):
        from aka.contracts.agent import AgentToolBinding
        from aka.contracts.filesystem import FilesystemAccess
        from aka.contracts.filesystem_access import normalize_access
        workspace = workspace.resolve(strict=True)
        capability = Capability(workspace, {key: environment[key] for key in (
            "ATREX_TELEMETRY_TRACE", "ATREX_TELEMETRY_ATTEMPT_ID", "ATREX_TELEMETRY_CAMPAIGN_ID",
            "ATREX_TELEMETRY_ITERATION_ID", "ATREX_ENVIRONMENT_STATE_FILE") if key in environment})
        capability.journal = self.journals.get(workspace)
        values = scrub_environment(environment)
        values["AKA_AGATE_TOOLS"] = json.dumps(self.config.agate_tools)
        values["AKA_RUNTIME_TOOLS"] = json.dumps(self.available_tools(capability))
        if capability.journal and capability.journal.view:
            with capability.journal.lock:
                protected = capability.journal.view.prepare_session()
            hidden = access.hidden_paths
            readonly = []
            for path in sorted(set((*access.read_only_paths, *(workspace / path for path in protected))),
                               key=lambda path: len(path.parts)):
                if any(path.is_relative_to(root) for root in (*hidden, *readonly)):
                    continue
                resolved = path.resolve(strict=True)
                if resolved != path:
                    # Resource discovery links are protected by their already
                    # declared read-only source mount, not by a symlink overlay.
                    mount = max((mount for mount in access.mounts if resolved.is_relative_to(mount.target)),
                                key=lambda mount: len(mount.target.parts), default=None)
                    if mount is None or mount.writable:
                        raise PermissionError("Readonly resource link lacks a read-only filesystem grant")
                    continue
                readonly.append(path)
            access = FilesystemAccess(access.mounts, hidden, tuple(readonly))
            values["GIT_CEILING_DIRECTORIES"] = str(workspace.parent)
        key = uuid.uuid4().hex
        with self.lock:
            if self.closed:
                raise RuntimeError("Workflow operations are closed")
            self.capabilities[key] = capability
            if capability.journal:
                self._write_binding_lineage_state(capability.journal, segment_status="active")
        bindings = tuple(AgentToolBinding(name, _OperationHandler(self, capability, route))
            for name, route in (("workflow.gateway", "/v1/gateway/execute"),
                ("workflow.journal", "/v1/journal/execute"), ("workflow.wiki", "/v1/wiki/query"),
                ("workflow.plugins", "/v1/plugins/execute")))
        try:
            with self.agent.tool_session(workspace, access=access, bindings=bindings, deadline=deadline) as grant:
                with self.lock:
                    live = self._live(capability)
                    if live:
                        capability.agent_session = grant
                if not live:
                    # close()/revoke() may race with owner grant creation.
                    grant.cancel()
                    raise SessionRevokedError("Runtime closed while binding the Agent Session")
                incremental = grant.filesystem_access
                combined = normalize_access(FilesystemAccess((*access.mounts, *incremental.mounts),
                    tuple(dict.fromkeys((*access.hidden_paths, *incremental.hidden_paths))),
                    tuple(dict.fromkeys((*access.read_only_paths, *incremental.read_only_paths)))))
                yield _ToolGrant({**values, **grant.environment}, combined)
        finally:
            self.revoke(key)
            if capability.journal and capability.journal.view:
                with capability.journal.lock:
                    primary = sys.exc_info()[1]
                    try:
                        capability.journal.view.publish()
                        capability.journal.service.publish_research_files()
                        with self.lock:
                            if not any(cap.journal is capability.journal for cap in self.capabilities.values()):
                                self._write_binding_lineage_state(capability.journal, segment_status="idle")
                    except Exception:
                        if primary is None:
                            raise
                        logging.getLogger(__name__).exception("Episode draft publication failed during cleanup")

    def _live(self, capability):
        with self.lock:
            if self.closed or (capability.cancel_event is not None and capability.cancel_event.is_set()):
                return False
            if capability.parent is not None:
                return self._live(capability.parent)
            return any(value is capability for value in self.capabilities.values())

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

    def _check_request_live(self, capability):
        if not self._live(capability):
            raise SessionRevokedError("Tool binding revoked")
        if capability.deadline is not None and time.monotonic() >= capability.deadline:
            raise RequestDispatchTimeout("Caller execution budget expired before the operation started")

    @contextmanager
    def request_context(self, capability, deadline, request_id):
        if not self._live(capability):
            raise SessionRevokedError("Tool binding revoked")
        previous = capability.deadline, capability.request_id
        capability.deadline, capability.request_id = deadline, request_id
        try:
            yield
        finally:
            capability.deadline, capability.request_id = previous

    def run_executor(self, command, cwd, environment, capability, *, deadline=None):
        from aka.contracts.execution import CommandSpec, UnsupportedExecution
        effective = min(deadline or float("inf"), capability.deadline or
                        (time.monotonic() + self.request_timeout_seconds))
        remaining = effective - time.monotonic()
        if not self._live(capability):
            raise SessionRevokedError("Session revoked before execution")
        if remaining <= 0:
            raise RequestDispatchTimeout("Caller execution budget has expired")
        self._acquire(self.slots, capability, effective)
        try:
            environment = dict(environment)
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
                str(ROOT.parents[2]), environment.get("PYTHONPATH", ""))))
            result = self.wiki_executor.run(CommandSpec(Path(cwd), tuple(command),
                timeout=max(1, math.ceil(remaining)), environment=environment, operation="dev",
                cancel_event=_OperationCancellation(self, capability),
                filesystem_access=self._wiki_access(capability, Path(cwd), command, environment)))
        except UnsupportedExecution as error:
            logging.getLogger(__name__).exception("Wiki execution scope rejected")
            raise RuntimeStateError("The local Wiki executor cannot enforce this request's execution scope.",
                code="wiki_execution_unavailable", next_action=(
                    "Continue without Wiki and report the execution configuration to the operator. "
                    "No GPU job was submitted by this Wiki request.")) from error
        finally:
            self.slots.release()
        if result.transport_status == "timed_out":
            if not self._live(capability):
                raise SessionRevokedError("Session revoked")
            raise WikiUnavailable("wiki_query_timeout", "Wiki query did not complete within its execution limits.")
        process = subprocess.CompletedProcess(command,
            result.returncode if result.returncode is not None else 2, result.stdout, result.stderr)
        process.execution_result = result
        return process

    def _wiki_access(self, capability, staged, command, environment):
        """Follow the Agent's sandbox selection for the local Wiki worker.

        Native mode retains host-user authority, even with bwrap installed.
        Sandbox mode needs an explicit Python/CLI closure in its filesystem grants.
        """
        # Tool grants also describe authorized paths in native mode. Their
        # presence does not select OS isolation: Wiki follows the Agent's
        # explicit sandbox selection, including when bwrap is installed.
        if not self.agent.capabilities.filesystem_isolation:
            return None
        access = self._request_access(capability, staged)
        if access is None:
            return None
        from aka.agent.execution.installations import installation_mounts
        from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
        from aka.execution.local.filesystem import RUNTIME_PATHS
        # This trusted worker consumes only its prepared Wiki request. Inherit
        # the staged workspace restrictions, not unrelated Agent Skill/reference
        # mounts (which can contain hardlinked, versioned resource snapshots).
        mounts = [mount for mount in access.mounts if mount.target == staged]
        visible = tuple(mount.target for mount in mounts) + tuple(map(Path, RUNTIME_PATHS))
        installations = installation_mounts(list(command), ("claude",), environment, Path.home(),
            hidden_paths=access.hidden_paths, forbidden_paths=(staged, capability.workspace),
            visible_paths=visible)
        mounts.extend(FilesystemMount(source, target, False) for source, target in installations)
        # Installed wheels are under the selected Python library. Also support
        # the explicitly selected checkout and Wiki resource tree in development.
        resources = [Path(path) for _, path in self.config.wiki_reference_paths]
        resources += [ROOT.parents[1],
                     self.plugins.resource_file("gpu-wiki", "tools/query_nl.py").parent.parent]
        if self.config.wiki_options.settings_file:
            resources.append(Path(self.config.wiki_options.settings_file))
        if self.config.wiki_profile_root:
            resources.append(self.config.wiki_profile_root)
        resources.extend(Path(path) for path in ("/etc/resolv.conf", "/etc/hosts",
            "/etc/nsswitch.conf", "/etc/ssl/certs", "/etc/pki/tls/certs"))
        for path in resources:
            if path.exists() and not any(path.is_relative_to(mount.target) for mount in mounts):
                mounts.append(FilesystemMount(path.resolve(strict=True), path, False))
        # Selected installation and Wiki resource roots can overlap. Collapse
        # only equivalent mappings; conflicting grants must still fail.
        from aka.contracts.filesystem_access import normalize_access
        return normalize_access(FilesystemAccess(tuple(mounts),
            tuple(path for path in access.hidden_paths if path.is_relative_to(staged)),
            tuple(path for path in access.read_only_paths if path.is_relative_to(staged))))

    def _request_access(self, capability, staged):
        from aka.contracts.filesystem import FilesystemAccess
        scope = capability.filesystem_access
        if scope is None:
            return None
        def project(path):
            return staged / path.relative_to(capability.workspace) if path.is_relative_to(capability.workspace) else path
        from aka.contracts.filesystem import FilesystemMount
        mounts = tuple(FilesystemMount(project(item.source), project(item.target), item.writable) for item in scope.mounts)
        return FilesystemAccess(mounts, tuple(project(path) for path in scope.hidden_paths),
            tuple(project(path) for path in scope.read_only_paths if project(path).exists()))

    def _retained_request(self, command):
        """Retain host-prepared inputs at a stable physical path, without Job state."""
        if command.prepared_inputs is not None and not self.executor.capabilities.prepared_inputs:
            from aka.contracts.execution import UnsupportedExecution
            raise UnsupportedExecution("selected executor cannot transport closed prepared inputs")
        from dataclasses import replace
        from aka.contracts.content import WorkspaceFile
        from aka.core.filesystem import open_private_directory
        files = []
        for path in sorted(command.workspace.rglob("*")):
            if path.is_symlink():
                raise RequestValidationError("Prepared request input cannot be a symlink")
            if path.is_file():
                data = read_input(command.workspace, path.relative_to(command.workspace).as_posix())
                files.append((path.relative_to(command.workspace).as_posix(), data, path.stat().st_mode & 0o777))
        # Typed requests already close their semantic inputs; Dev captures its
        # whole declared input view. Secrets enter only the digest, never a log.
        canonical = tuple(command.command) == ("python3", "test_kernel.py")
        typed = command.submission is not None
        identity = {"command": list(command.command), "operation": command.operation,
            "hardware": command.hardware, "timeout": command.timeout,
            "submission": command.submission.decode() if command.submission is not None else None,
            "requirements": list(command.requirements), "environment": dict(command.environment),
            "num_gpus": command.num_gpus, "languages": list(command.languages), "artifacts": list(command.artifacts),
            "source_archives": [(archive.role, archive.path, archive.entry_point, hashlib.sha256(archive.content).hexdigest())
                                for archive in command.source_archives],
            "explicit_inputs": [(item.path, item.mode, hashlib.sha256(item.content).hexdigest()) for item in command.input_files],
            "task_inputs": self.evaluation_identity,
            "source": self.workspace_provider.source_identity(self.workspace_provider.read_source(command.workspace))
                if typed or canonical else None,
            "inputs": [] if typed or canonical else [(name, hashlib.sha256(data).hexdigest(), mode) for name, data, mode in files]}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        destination = (self.audit_root or self.root) / "prepared-requests" / digest / "workspace"
        from aka.task.source.formats.private_state import read as source_binding
        binding = source_binding(command.workspace)
        if binding is not None:
            self.workspace_provider.bind_source_contract(destination, file_system=binding[0], contract=binding[1])
        existed = destination.exists()
        with open_private_directory(destination):
            pass
        if not existed:
            self.workspace_provider.install_files(destination, tuple(WorkspaceFile(name,
                source=command.workspace / name) for name, _, _ in files), immutable=True)
        elif typed or canonical:
            original = self.workspace_provider.read_source(destination)
            current = self.workspace_provider.read_source(command.workspace)
            if self.workspace_provider.source_identity(original) != self.workspace_provider.source_identity(current):
                raise RequestValidationError("Retained preparation source is damaged; query the original Job before recovery")
        scope = command.filesystem_access
        capabilities = self.executor.capabilities
        if scope is not None and not capabilities.filesystem_isolation:
            from aka.contracts.execution import ExecutionFile, PreparedExecutionInputs, UnsupportedExecution
            # The Agent scope has already authorized the physical snapshot.
            # Remote execution consumes that closed view, not host mount paths.
            if command.submission is None and command.prepared_inputs is None:
                if not capabilities.prepared_inputs:
                    raise UnsupportedExecution("selected executor cannot transport the authorized request inputs")
                explicit = {item.path for item in command.input_files}
                command = replace(command, prepared_inputs=PreparedExecutionInputs(tuple(
                    ExecutionFile(name, data, mode) for name, data, mode in files if name not in explicit)))
            command = replace(command, filesystem_access=None)
            scope = None
        if scope is not None:
            from aka.contracts.filesystem import FilesystemAccess
            def project(path):
                return destination / path.relative_to(command.workspace) if path.is_relative_to(command.workspace) else path
            from aka.contracts.filesystem import FilesystemMount
            mounts = tuple(FilesystemMount(project(item.source), project(item.target), item.writable) for item in scope.mounts)
            scope = FilesystemAccess(mounts, tuple(project(path) for path in scope.hidden_paths),
                tuple(project(path) for path in scope.read_only_paths))
        return replace(command, workspace=destination, filesystem_access=scope), digest

    def _measurement_environment(self, command):
        from aka.contracts.execution_environment import EnvironmentQuery
        identity = None
        check = getattr(self.executor, "check_environment", None)
        if callable(check):
            support = check(EnvironmentQuery(command.hardware, command.operation, command.languages,
                command.requirements, command.num_gpus, features=("source_archives",) if command.source_archives else ()))
            if support.support == "supported" and support.availability == "available":
                identity = support.environment.identity
        self.experiments.environment_identity = identity



    def execute(self, capability: Capability, route: str, request: dict) -> dict:
        if route == "/v1/plugins/execute":
            return self.execute_plugin(capability, request)
        if route == "/v1/journal/execute":
            return self.execute_journal(capability, request)
        with _validating_request():
            allowed = {"argv"} if route.endswith("execute") else {"argv", "tool"}
            optional = {"predictions"} if route == "/v1/gateway/execute" else set()
            if not allowed <= set(request) or set(request) - allowed - optional:
                raise ValueError(f"Request fields must be {sorted(allowed)}; optional: {sorted(optional)}")
            from aka.workflow.light.predictions import validate_predictions
            predictions = validate_predictions(request.get("predictions", []))
            argv = validated_argv(request["argv"])
        if route == "/v1/gateway/execute":
            words = argv[:argv.index("--")] if "--" in argv else argv
            from aka.contracts.agate_tools import AGATE_TOOLS
            explicit_kinds = [words[i + 1] for i, word in enumerate(words[:-1]) if word == "--kind"]
            explicit_kinds += [word.split("=", 1)[1] for word in words if word.startswith("--kind=")]
            for kind in explicit_kinds:
                if kind in AGATE_TOOLS and kind not in self.config.agate_tools:
                    raise AgentRequestError(f"Runtime tool {kind!r} is disabled for this Campaign.",
                        code="runtime_tool_disabled", next_action="Use python3 tools/sandbox.py --help for enabled tools.",
                        available_kinds=list(self.config.agate_tools), job_submitted=False)
            if words in (["--help"], ["-h"]):
                return {"operation": "help", "status": "completed", "result": {
                    "available_kinds": list(self.available_tools(capability)),
                    "instruction": "Use --kind OPERATION --help; research and submission requests use --request-file."}}
            with _validating_request():
                from aka.evaluation.tool_requests import parse_tool_arguments
                from aka.execution.command_syntax import requested_operation
                access_argv = [word for word in words if word not in {"--help", "-h"}]
                access_argv.extend(argv[len(words):])  # Preserve remote argv, including its --help.
                parsed_access = parse_tool_arguments(access_argv, enabled_kinds=self.config.agate_tools)
                kind = parsed_access.kind
                if kind == "auto":
                    kind = requested_operation("auto", parsed_access.command) if parsed_access.command else "run"
                    kind = {"run": "evaluate", "same_allocation_abba": "evaluate"}.get(kind, kind)
            if predictions and kind not in {"evaluate", "command", "dev"}:
                raise AgentRequestError("Predictions are supported only for Evaluate and Command/Dev", code="invalid_prediction_operation")
            if kind not in self.config.agate_tools:
                raise AgentRequestError(f"Agate Runtime tool {kind!r} is disabled for this Campaign.",
                    code="runtime_tool_disabled", next_action=(
                        "Use one of the enabled tools: " + ", ".join(self.config.agate_tools)
                        if self.config.agate_tools else "No Agate tools are enabled; use existing saved evidence."),
                    available_kinds=list(self.config.agate_tools), job_submitted=False)
        if route == "/v1/gateway/execute" and any(arg in {"--help", "-h"} for arg in argv[:argv.index("--") if "--" in argv else len(argv)]):
            from aka.evaluation.tool_requests import tool_argument_parser
            from aka.workflow.light.host.request_feedback import gateway_schema
            parser = tool_argument_parser(enabled_kinds=self.config.agate_tools)
            words = argv[:argv.index("--")] if "--" in argv else argv
            kinds = [words[i + 1] for i, word in enumerate(words[:-1]) if word == "--kind"]
            kinds += [word.split("=", 1)[1] for word in words if word.startswith("--kind=")]
            kind = kinds[0] if len(kinds) == 1 else None
            with _validating_request():
                schema = gateway_schema(kind, parser, enabled_kinds=self.config.agate_tools)
                if "supported_kinds" in schema:
                    raise ValueError(f"--help requires one supported --kind ({', '.join(schema['supported_kinds'])}); got {kind!r}")
            return {"operation": "help", "status": "completed", "result": {
                "kind": kind, "request_schema": schema,
                "optional_predictions": "--predictions JSON: JSON array of {scope, prediction}; frozen before new execution, not sent to Agate. Evaluate and Command/Dev only.",
                "campaign_defaults": {"warmup": self.evaluation_configuration.warmup_iters,
                    "bench": self.evaluation_configuration.bench_iters,
                    "step_timeout_seconds": self.config.evaluation_step_timeout},
                "timing_policy": "Omitted timing options inherit Campaign policy. ABBA run timeout is derived from its schedule; explicit values are not clipped."}}

        environment = dict(self.environment)
        environment.update(capability.context)
        if route == "/v1/wiki/query":
            with tempfile.TemporaryDirectory(dir=self.root, prefix="wiki-") as temporary:
                staged = Path(temporary)
                with _validating_request():
                    command = self._wiki_command(capability, request["tool"], argv, staged, environment)
                return self.wiki.execute(self, capability, command, staged, environment, argv)
        from dataclasses import replace
        from aka.evaluation.tool_requests import request_from_argv
        from aka.workflow.light.measurement_policy import measurement_defaults
        with tempfile.TemporaryDirectory(dir=self.root, prefix="request-") as temporary:
            staged = Path(temporary)
            with _validating_request():
                argv = filter_options(argv, AUTHORITY, FORBIDDEN)
                from aka.task.source.formats.private_state import read as source_binding
                binding = source_binding(capability.workspace)
                if binding is not None:
                    self.workspace_provider.bind_source_contract(staged, file_system=binding[0], contract=binding[1])
                snapshot(capability.workspace, staged,
                         custom_tools=bool(capability.journal and capability.journal.view),
                         filesystem_access=capability.filesystem_access)
                # Input authority belongs to the retained Task, never a mutable
                # Agent copy. Evaluation prepares its selected references.
                from aka.evaluation.command_inputs import host_evaluation_environment
                configuration = replace(self.evaluation_configuration,
                    environment=host_evaluation_environment(environment))
                evaluation_input = self.evaluation_input
                from aka.evaluation.tool_requests import parse_tool_arguments
                parsed = parse_tool_arguments(argv)
                baseline_source = None
                if parsed.baseline_commit:
                    if capability.journal is None:
                        raise ValueError("ABBA requires a bound Agent Git workspace")
                    baseline_source = self.workspace_provider.read_source(
                        capability.journal.service.git_workspace, revision=parsed.baseline_commit)
                converted = request_from_argv(argv, workspace=staged,
                    evaluation_input=evaluation_input, configuration=configuration,
                    workspace_provider=self.workspace_provider, baseline_source=baseline_source,
                    defaults=measurement_defaults(step_timeout=configuration.evaluation_step_timeout,
                        warmup_iters=configuration.warmup_iters, bench_iters=configuration.bench_iters))
                args = converted.args
                if converted.kind == "env":
                    from aka.contracts.execution_environment import EnvironmentQuery
                    from aka.execution.gateway.native_options import apply_request
                    query = EnvironmentQuery(hardware=args.env_gpu or configuration.hardware,
                        fresh=args.env_force, parameters=apply_request({}, args, "env"))
                    if args.dry_run:
                        return {"exit_code": 0, "stdout": json.dumps({"operation": "env", "hardware": query.hardware,
                            "prepared": True}) + "\n", "stderr": ""}
                    facts = self.executor.query_environment(query)
                    environment_facts = facts.as_dict()
                    requirements = getattr(evaluation_input, "worker_requirements", None)
                    if requirements is not None:
                        environment_facts["task_worker_requirements"] = (
                            requirements.source.read_text(encoding="utf-8")
                            if requirements.source is not None else requirements.text
                        )
                    from aka.workflow.light.host.projection import credential_values, redact_diagnostics
                    environment_facts = redact_diagnostics(environment_facts,
                        private_paths=(str(staged), str(self.root), str(self.audit_root or "")),
                        private_values=credential_values(environment))
                    from aka.core.service_audit import audit_service_exchange
                    from aka.workflow.light.host.projection import runtime_environment_result
                    audit_service_exchange(service="execution.environment", response=environment_facts)
                    environment_facts = runtime_environment_result(environment_facts, self.config.agate_tools,
                        runtime_kinds=self.available_tools(capability))
                    upstream_errors = [{key: value[key] for key in ("error", "http_status") if key in value}
                        for value in environment_facts.get("raw", {}).values()
                        if isinstance(value, dict) and value.get("error") is not None]
                    if upstream_errors:
                        detail = dict(upstream_errors[0])
                        if len(upstream_errors) > 1:
                            detail["errors"] = upstream_errors
                        return {"exit_code": 1, "stdout": json.dumps(environment_facts, allow_nan=False)+"\n", "stderr": "",
                                "agent_result": {"operation": "env", "status": "failed", "result": detail}}
                    return {"exit_code": 0, "stdout": json.dumps(environment_facts, allow_nan=False)+"\n", "stderr": ""}
                if args.dry_run:
                    return {"exit_code": 0, "stdout": json.dumps({"operation": converted.kind, "prepared": True})+"\n", "stderr": ""}
                for path in converted.request.artifacts:
                    if relative_path(path).parts[0] not in {"profiles", "scratch"}:
                        raise ValueError("--sync must be inside profiles/ or scratch/")
                command = replace(converted.request, cancel_event=_OperationCancellation(self, capability),
                    filesystem_access=self._request_access(capability, staged)
                        if converted.request.submission is None else converted.request.filesystem_access)
            self._measurement_environment(command)
            command, request_digest = self._retained_request(command)
            from aka.task.source.formats import snapshot as pack_source
            from aka.task.source.formats.artifacts import Artifact
            from aka.task.candidate_workspace.agent_git import AgentGit, DirtyKernelWorktree
            kernels, inputs, snapshots = {}, {}, {}
            baseline_path = None
            source = pack_source(command.workspace)
            if capability.journal and converted.kind in {"run", "same_allocation_abba", "profile", "check", "disassemble", "dev", "command"}:
                try:
                    commit = AgentGit(capability.journal.service.git_workspace).kernel_commit(source)
                except DirtyKernelWorktree as error:
                    raise AgentRequestError(str(error), code="uncommitted_kernel", paths=list(error.paths),
                        next_action="Commit the complete candidate source, then retry.") from error
                kernels["candidate"] = self.measurements.kernel(source, commit)
                artifact = Artifact(source)
                snapshots["candidate"] = artifact.digest
                inputs.update({name: hashlib.sha256(data).hexdigest() for name, data in artifact.files.items()})
            if parsed.baseline_commit:
                from aka.task.source.formats.storage import from_git
                baseline_bytes = from_git(capability.journal.service.git_workspace, parsed.baseline_commit)
                kernels["baseline"] = self.measurements.kernel(baseline_bytes, parsed.baseline_commit)
                artifact = Artifact(baseline_bytes)
                snapshots["baseline"] = artifact.digest
                inputs.update({"baseline/" + name: hashlib.sha256(data).hexdigest() for name, data in artifact.files.items()})
                baseline_path = "baseline" if artifact.layout.archived else "baseline/" + artifact.layout.python_entrypoint
            operation = "evaluate" if converted.kind == "run" else converted.kind
            options = {"evaluation_mode": args.evaluation_mode or "full", "source_snapshots": snapshots,
                "command": list(command.command),
                "baseline_path": baseline_path, "shape_ids": args.shape_id}
            identity = {"operation": operation, "options": options, "inputs": inputs,
                "typed_request_digest": request_digest, "kernels": kernels,
                "hardware": self.config.hardware}
            def check_wait():
                if not self._live(capability):
                    raise SessionRevokedError("Session revoked while waiting for a shared measurement")
            response = self.experiments.execute(command, request_id=capability.request_id,
                identity=identity, kernels=kernels, deadline=capability.deadline,
                check_wait=check_wait, predictions=predictions)
            agent_result = response.get("agent_result", {})
            record_id = agent_result.get("gateway_record_id")
            if capability.journal and record_id:
                service = capability.journal.service
                service.observe_measurement(agent_result)
                if not response.get("reused"):
                    service.record_passing_full_evaluate_checkpoint(self.measurements.read(record_id))
                service.publish_research_files(synchronize=False, direction_ids=frozenset(),
                    experiment_ids=frozenset(), record_ids=frozenset({record_id}))
                hint = service.experiment_recording_hint(agent_result)
                if hint:
                    agent_result["research_next_action"] = hint
            self.audit_process(capability, "gateway", argv,
                subprocess.CompletedProcess([], response["exit_code"], response["stdout"], response["stderr"]), record_id)
            return response

    def execute_journal(self, capability: Capability, request: dict) -> dict:
        binding = capability.journal
        if binding is None:
            raise AgentRequestError("Research and submission tools require an active Campaign binding", repairable=False,
                                    code="journal_unavailable", next_action="Report the missing Campaign binding to the operator.")
        if not self._live(capability):
            raise SessionRevokedError()
        try:
            agent_git = None
            if request.get("operation") == "kernel_submit":
                from aka.task.candidate_workspace.agent_git import AgentGit
                agent_git = (
                    binding.view.agent_git
                    if binding.view is not None
                    else AgentGit(binding.service.workspace)
                )
            review_kernel = None
            if agent_git is not None and request.get("operation") == "kernel_submit":
                def review_kernel(
                    commit_id,
                    submission_id,
                    submission_revision,
                    resume_session_id,
                ):
                    return self._review_submitted_kernel(
                        capability,
                        binding,
                        agent_git,
                        commit_id,
                        submission_id,
                        submission_revision,
                        resume_session_id,
                    )
            value = binding.service.execute(
                request,
                review_kernel=review_kernel,
                check_wait=lambda: self._check_request_live(capability),
            )
        except (AgentRequestError, RuntimeStateError):
            raise
        except ValueError as error:
            evidence_hint = "Read PROGRESS.md, research-library/ and records/ for current evidence."
            raise AgentRequestError(str(error), next_action=(
                "Correct the indicated field or lifecycle state, then retry. Read .agents/skills/runtime-records/references/journal.md. "
                + evidence_hint)) from error
        response = {"exit_code": 0, "stdout": json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", "stderr": ""}
        self.audit_process(capability, "journal", [str(request.get("operation"))],
                           subprocess.CompletedProcess([], 0, response["stdout"], ""))
        return response

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

    def execute_plugin(self, capability: Capability, request: dict) -> dict:
        with _validating_request():
            if request == {"action": "list"}:
                return {"exit_code": 0, "stdout": json.dumps(self.plugins.catalog()) + "\n", "stderr": ""}
            if set(request) != {"action", "tool", "input"} or request["action"] != "call":
                raise ValueError("Use action=list, or action=call with tool and input")
            if not isinstance(request["tool"], str):
                raise ValueError("tool must be a catalog name")
        if not self._live(capability):
            raise SessionRevokedError("Session revoked before tool invocation")
        from aka.core.plugin_runtime.schema import PluginError
        try:
            if self.wiki.provider is not None:
                plugin_id, _, operation = request["tool"].partition(".")
                plugin = next((p for p in self.plugins.plugins if p.id == plugin_id), None)
                tool = plugin.tools.get(operation) if plugin is not None else None
                if tool is not None and tool.get("runtime_tool") == "wiki":
                    self.plugins.check_lock(capability.workspace)
                    if not self.config.wiki_enabled:
                        raise RuntimeStateError("Wiki is disabled by the selected composition", code="wiki_disabled")
                    from aka.core.plugin_runtime.schema import validate_schema
                    from aka.workflow.light.host.projection import project_response, credential_values, redact_diagnostics
                    validate_schema(tool["input_schema"], request["input"])
                    with tempfile.TemporaryDirectory(dir=self.root, prefix="wiki-") as temporary:
                        staged = Path(temporary)
                        environment = {name: value for name, value in self.environment.items()
                                       if not name.startswith(("AGATE_", "AKA_", "ATREX_"))}
                        def project(value):
                            public = project_response(subprocess.CompletedProcess([operation], 0,
                                json.dumps(value, ensure_ascii=False), ""), wiki=True,
                                private_values=credential_values(self.environment),
                                private_paths=(str(staged), str(self.root), str(self.audit_root or "")))
                            result = json.loads(public["stdout"])
                            try:
                                validate_schema(tool["output_schema"], result, "output")
                            except PluginError as error:
                                raise PluginError("invalid_output", str(error)) from error
                            if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) + 1 > MAX_OUTPUT_BYTES:
                                raise_output_error(request["tool"], "result_too_large")
                            return result
                        def redact(value):
                            return redact_diagnostics(value, private_values=credential_values(self.environment),
                                private_paths=(str(staged), str(self.root), str(self.audit_root or "")))
                        result = self.wiki.invoke(self, capability, operation, request["input"], staged,
                                                  environment, project=project, redact_error=redact)
                    if not self._live(capability):
                        raise SessionRevokedError("Session revoked during tool invocation")
                    return {"exit_code": 0, "stdout": json.dumps(result, ensure_ascii=False) + "\n", "stderr": ""}
            result = self.plugins.call(request["tool"], request["input"], capability.workspace,
                deadline=capability.deadline, cancel_event=_OperationCancellation(self, capability))
        except PluginError as error:
            if error.code == "invalid_output":
                raise_output_error(request["tool"], "invalid_output")
            if error.code in {"schema_validation", "tool_not_found"}:
                raise AgentRequestError(str(error), code=error.code) from error
            raise
        if not self._live(capability):
            raise SessionRevokedError("Session revoked during tool invocation")
        rendered = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        if len(rendered.encode()) > MAX_OUTPUT_BYTES:
            raise_output_error(request["tool"], "result_too_large")
        return {"exit_code": 0, "stdout": rendered, "stderr": ""}

    def _wiki_command(self, capability, tool, argv, staged, environment):
        if not self.config.wiki_enabled:
            raise AgentRequestError("GPU Wiki is disabled in the selected Workflow configuration",
                                    code="wiki_disabled", repairable=False,
                                    next_action="Continue with the enabled tools. Only the operator can enable Wiki; do not retry or access its private store directly.")
        if not isinstance(tool, str) or tool not in WIKI_STORES:
            raise ValueError("Unsupported Wiki tool")
        argv = normalize_wiki_argv(tool, argv)
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
        return [sys.executable, str(self.plugins.resource_file("gpu-wiki", f"tools/{tool}.py")), *argv]



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
    def _publish_request_audit(path: Path, value: dict) -> None:
        publish(
            path.parent,
            path.name,
            (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode(),
        )
    @staticmethod
    def _read_request_audit(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
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
    def _review_submitted_kernel(
        self,
        capability,
        binding,
        agent_git,
        commit_id: str,
        submission_id: str | None = None,
        submission_revision: int = 1,
        resume_session_id: str | None = None,
    ) -> dict:
        """Review one exact Agent commit and its saved full-Evaluate evidence."""
        from aka.task.source.formats.storage import from_git

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
        source_contract = "pass"
        review: dict[str, object] = {
            "verdict": "not_applicable",
            "checks": [],
            "dependencies": [],
            "summary": "No independent final-source Review applies.",
        }
        if production_applicable and policy_validator is not None:
            policy_result = policy_validator(
                binding.service.git_workspace,
                source,
                parent=capability,
                require_gluon=binding.require_gluon,
                submission_id=submission_id,
                submission_revision=submission_revision,
                resume_session_id=resume_session_id,
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
                raw_source_contract = policy_result.get("source_contract")
                if raw_source_contract in {"pass", "fail"}:
                    source_contract = str(raw_source_contract)
                raw_policy_review = policy_result.get("review")
                if isinstance(raw_policy_review, Mapping):
                    review = dict(raw_policy_review)
            else:
                violations = list(policy_result)
                implementation_review = {"availability": "unavailable"}
        evaluation_evidence = kernel_full_evaluation_evidence(
            self.measurements,
            commit_id,
        )
        return {
            "accepted": (
                not violations
                and evaluation_evidence.get("status") == "passed"
            ),
            "source_contract": source_contract,
            "review": review,
            "implementation_review": implementation_review,
            "diagnostics": violations,
            "evaluation_evidence": evaluation_evidence,
        }
    def _write_binding_lineage_state(
        self, binding: JournalBinding, *, segment_status: str
    ) -> None:
        """Project the live Session state for one lineage binding."""
        service = binding.service
        self._write_lineage_state(
            episode=service.episode,
            branch=service.episode_branch,
            base_commit=service.base_commit,
            base_kernel_commit_id=service.base_kernel_id,
            agent_head_commit_id=(
                binding.view.agent_git.head()
                if binding.view is not None
                else service.base_commit
            ),
            tool_modules=service.tool_modules,
            segment_status=segment_status,
            origin=service.origin,
        )
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
        from aka.workflow.light.search import merge_lineage_state

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
            "agent_request": {**request, "argv": audited_argv} if operation == "gateway" else json.loads(json.dumps(request)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._publish_request_audit(path, value)
    def audit_service_exchange(self, **exchange):
        from aka.core.service_audit import audit_service_exchange
        audit_service_exchange(**exchange)

    def _record_service_exchange(
        self, request_id: str, *, service: str, request: object | None = None,
        response: object | None = None, attempt: int | None = None,
        exchange_id: str | None = None,
    ) -> None:
        """Attach one exact Runtime-to-service exchange to the active request audit."""
        if not self.audit_root or not request_id:
            return
        from aka.workflow.light.host.projection import credential_values, redact_diagnostics
        redaction = {"private_paths": (str(self.root), str(self.audit_root), str(ROOT),
            str(self.config.private_reference_dir or ""), self.config.url),
            "private_values": credential_values(self.environment)}
        request = redact_diagnostics(request, **redaction)
        response = redact_diagnostics(response, **redaction)
        # Final-source and supplemental reviewers can run concurrently.  Keep
        # each read/modify/write atomic so one completed exchange cannot erase
        # another one that finished at nearly the same time.
        with self.lock:
            path = self._request_audit_path(request_id)
            value = self._read_request_audit(path)
            exchanges = value.get("service_exchanges")
            if not isinstance(exchanges, list):
                exchanges = []
            row = next(
                (
                    item for item in reversed(exchanges)
                    if isinstance(item, dict)
                    and item.get("service") == service
                    and (
                        item.get("exchange_id") == exchange_id
                        if exchange_id is not None
                        else item.get("attempt") == attempt
                    )
                ),
                None,
            )
            if row is None:
                row = {"service": service}
                exchanges.append(row)
            if exchange_id is not None:
                row["exchange_id"] = exchange_id
            if attempt is not None:
                row["attempt"] = attempt
            if request is not None:
                row["request"] = json.loads(json.dumps(request))
            if response is not None:
                row["response"] = json.loads(json.dumps(response))
            value["service_exchanges"] = exchanges
            value["updated_at"] = datetime.now(timezone.utc).isoformat()
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
        response_detail = response.get("result")
        response_detail = response_detail if isinstance(response_detail, dict) else {}
        nested_error = response_detail.get("error", response.get("error"))
        error = dict(nested_error) if isinstance(nested_error, dict) else {}
        # Retain compatibility with executor responses that still put error
        # identity at the top level, while treating the Agent-facing structured
        # error as authoritative for HTTP request failures.
        for key in (
            "code", "reason", "error_class", "message", "outcome", "phase",
            "next_action",
        ):
            if key not in error and response.get(key) is not None:
                error[key] = response[key]
        if response.get("detail") is not None and "detail" not in error:
            error["detail"] = response["detail"]
        if isinstance(response.get("issues"), list):
            error["issues"] = response["issues"]
        details = response.get("details")
        if isinstance(details, dict) and isinstance(details.get("validation_errors"), list):
            error["validation_errors"] = details["validation_errors"]
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
            "agent_response": result,
            "execution_exit_code": execution_exit_code,
            "error": error or None,
        })
        if isinstance(gateway_record_id, str) and gateway_record_id:
            value["gateway_record_id"] = gateway_record_id
        self._publish_request_audit(path, value)
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
