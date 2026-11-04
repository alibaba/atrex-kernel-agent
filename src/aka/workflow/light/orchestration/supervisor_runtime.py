"""Campaign-owned GPU/Wiki, Journal and sealed Episode handoff service."""
from __future__ import annotations

from aka.contracts.evaluation import derived_evaluation_timeout

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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from aka.workflow.light.host.workspace import (
    MAX_FILE_BYTES, MAX_TOTAL_BYTES, InputSizeLimitError, publish, read_input, relative_path, snapshot,
)
from aka.workflow.light.measurements import QUERY_KINDS
from aka.workflow.light.host.journal import SupervisorJournalService, load_journal, journal_lock
from aka.workflow.light.host.errors import AgentRequestError, RuntimeStateError, error_response
from aka.workflow.light.host.request_feedback import augment_error
from aka.workflow.light.host.plugin_output import (
    MAX_OUTPUT_BYTES, raise_output_error,
)
from .constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_HARNESS
from .episode_workspace import EpisodeWorkspace
from aka.workflow.light.candidate_gate import CandidateValidation, CandidateGate
from aka.workflow.light.support.errors import (
    RequestValidationError, RequestDispatchTimeout, GatewayExecutionUnavailable, SessionRevokedError,
)
from aka.workflow.light.support.paths import supervisor_campaign_root
from aka.resources.wiki.config import WikiConfig
from aka.workflow.light.host.wiki import WikiService

OWNER_ENV = "ATREX_AKA_RUNTIME_OWNER"
URL_ENV = "ATREX_AKA_RUNTIME_URL"
TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"
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
    # In-memory controller object, never a token or PID supplied by Agent.
    session_control: object | None = field(default=None, repr=False)
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


class _OperationHandler:
    """Workflow operation adapter; the selected Agent owns transport authority."""
    def __init__(self, owner, capability, route):
        self.owner, self.capability, self.route = owner, capability, route

    def handle(self, payload, context):
        from dataclasses import replace
        owner = self.owner
        capability = replace(self.capability, parent=self.capability,
                             cancel_event=context.cancel_event, filesystem_access=context.filesystem_access)
        request, status, exit_code = None, 200, None
        try:
            if context.workspace != capability.workspace:
                raise SessionRevokedError("Tool workspace differs from its binding")
            with _validating_request():
                if len(payload) > MAX_REQUEST_BYTES:
                    raise ValueError("Request exceeds 2 MiB")
                request = json.loads(payload)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
            deadline = time.monotonic() + owner.request_timeout_seconds
            if context.deadline is not None:
                deadline = min(deadline, context.deadline)
            if owner.is_wiki_request(self.route, request):
                capability.deadline = deadline
                result = owner.execute(capability, self.route, request)
            else:
                with owner.request_lock(capability, deadline, uuid.uuid4().hex):
                    if not owner._live(capability):
                        raise SessionRevokedError("Tool binding revoked")
                    result = owner.execute(capability, self.route, request)
            if self.route == "/v1/gateway/execute" and isinstance(result.get("agent_result"), dict):
                exit_code, result = result["exit_code"], result["agent_result"]
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
        except SessionRevokedError:
            status, result = 401, failure("Tool binding revoked", repairable=False, code="session_revoked")
        except Exception:
            status, result = 503, failure("Operation outcome could not be confirmed", repairable=False)
            logging.getLogger(__name__).exception("Workflow tool operation failed")
        if status >= 400:
            binding = capability.journal
            result = augment_error(result, route=self.route, request=request,
                research_records=owner.journal_factory.research_records,
                supervisor_git=binding.service.supervisor_git if binding else True,
                parser=None,
                catalog=owner.plugins.catalog() if self.route == "/v1/plugins/execute" and status == 400 else ())
            from aka.workflow.light.host.projection import credential_values, redact_diagnostics
            result = redact_diagnostics(result,
                private_paths=(str(owner.root), str(owner.audit_root or ""), str(ROOT),
                    str(owner.config.private_reference_dir or ""), owner.config.url,
                    str(owner.config.atrex_bench_root or "")),
                private_values=credential_values(owner.environment))
        return json.dumps({"status": status, "result": result,
                           "execution_exit_code": exit_code}, ensure_ascii=False,
                          allow_nan=False).encode()


class SupervisorRuntime:
    def __init__(self, config: RuntimeConfig, *, workspace_provider=None, services=None):
        self.config = config
        if services is None or workspace_provider is None:
            raise TypeError("Workflow operations require explicitly selected shared modules")
        self.agent, self.executor = services.agent, services.executor
        self.acceptance = CandidateGate()
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
        self.measurements = SharedMeasurements(factory=services.experiments, executor=self.executor,
            evaluator=services.evaluator, namespace_workspace=config.workspace or self.root,
            evaluation_input=self.evaluation_input, workspace_provider=workspace_provider,
            command_evaluator=services.command_evaluator)
        self.experiments = self.measurements
        # Resources owns both the selection identity and its validation. Workflow
        # stores the opaque value without interpreting plugin or Skill locks.
        resource_lock = (self.audit_root or self.root) / "resources.json"
        if resource_lock.exists():
            self.plugins.check_resource_snapshot(json.loads(read_input(resource_lock.parent, resource_lock.name)))
        else:
            publish(resource_lock.parent, resource_lock.name, json.dumps(self.plugins.resource_snapshot()).encode())
        self.wiki = WikiService((self.audit_root or self.root) / "wiki", config.wiki_options,
                                self.plugins.resource_snapshot())

    @property
    def journals_root(self) -> Path:
        """Private Journal collection, sibling to the Campaign's Measurement Store."""
        return (self.audit_root or self.root) / "journals"

    @property
    def request_timeout_seconds(self) -> float:
        if self.config.request_timeout is not None:
            return self.config.request_timeout
        # Preserve this Workflow request deadline independently of Execution waiting.
        grace = 14_400 if self.gateway_queue_wait_grace is None else self.gateway_queue_wait_grace
        return 2 * (self.config.timeout + grace) + 120

    def close(self):
        with self.lock:
            self.closed = True
            capabilities = list(self.capabilities.values())
            self.capabilities.clear()
        for capability in capabilities:
            if capability.session_control:
                capability.session_control.kill()
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
                from aka.workflow.light.history import EpisodeHistory
                history = (EpisodeHistory(
                    episode=episode,
                    episodes_root=self.config.workspace.resolve() / ".atrex_long_horizon/episodes",
                ) if self.config.workspace else None)
                view = EpisodeWorkspace(workspace, evidence / "agent",
                                        is_ppu=hardware_vendor(self.config.hardware) == "ppu",
                                        research_records=self.journal_factory.research_records,
                                        resources_registry=self.plugins, workspace_provider=self.workspace_provider,
                                        inherit_state=getattr(self, "inherit_state", True),
                                        adaptive_root=(self.audit_root or self.root) / "adaptive-state",
                                        episode=episode, history=history)
                agent_workspace = view.prepare()
            binding = JournalBinding(self.journal_factory.open(
                workspace=agent_workspace, git_workspace=workspace,
                campaign_root=self.journals_root, evidence_root=evidence,
                supervisor_git=supervisor_git, workspace_provider=self.workspace_provider,
                evidence_reader=self.measurements.evidence), view=view)
            self.journals[workspace] = binding
            self.journals[agent_workspace] = binding
            return agent_workspace

    @contextmanager
    def session(self, workspace: Path, environment: dict[str, str], *, access, deadline=None):
        from aka.contracts.agent import AgentToolBinding
        from aka.contracts.filesystem import FilesystemAccess
        workspace = workspace.resolve(strict=True)
        capability = Capability(workspace, {key: environment[key] for key in (
            "ATREX_TELEMETRY_TRACE", "ATREX_TELEMETRY_ATTEMPT_ID", "ATREX_TELEMETRY_CAMPAIGN_ID",
            "ATREX_TELEMETRY_ITERATION_ID", "ATREX_ENVIRONMENT_STATE_FILE") if key in environment})
        capability.journal = self.journals.get(workspace)
        values = scrub_environment(environment)
        if capability.journal and capability.journal.view:
            with capability.journal.lock:
                protected = capability.journal.view.prepare_session()
            hidden = tuple(dict.fromkeys((*access.hidden_paths, workspace / ".git")))
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
        bindings = tuple(AgentToolBinding(name, _OperationHandler(self, capability, route))
            for name, route in (("workflow.gateway", "/v1/gateway/execute"),
                ("workflow.journal", "/v1/journal/execute"), ("workflow.wiki", "/v1/wiki/query"),
                ("workflow.plugins", "/v1/plugins/execute")))
        try:
            with self.agent.tool_session(workspace, access=access, bindings=bindings, deadline=deadline) as grant:
                incremental = grant.filesystem_access
                combined = FilesystemAccess((*access.mounts, *incremental.mounts),
                    tuple(dict.fromkeys((*access.hidden_paths, *incremental.hidden_paths))),
                    tuple(dict.fromkeys((*access.read_only_paths, *incremental.read_only_paths))))
                yield _ToolGrant({**values, **grant.environment}, combined)
        finally:
            self.revoke(key)
            if capability.journal and capability.journal.view:
                with capability.journal.lock:
                    primary = sys.exc_info()[1]
                    try:
                        capability.journal.view.publish(sealed_source=capability.journal.service.sealed_source())
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

    @contextmanager
    def request_lock(self, capability, deadline, request_id):
        self._acquire(capability.lock, capability, deadline)
        capability.deadline, capability.request_id = deadline, request_id
        try:
            yield
        finally:
            capability.deadline, capability.request_id = None, ""
            capability.lock.release()

    def run_executor(self, command, cwd, environment, capability, *, deadline=None):
        from aka.contracts.execution import CommandSpec
        effective = min(deadline or float("inf"), capability.deadline or
                        (time.monotonic() + self.request_timeout_seconds))
        remaining = effective - time.monotonic()
        if not self._live(capability):
            raise SessionRevokedError("Session revoked before execution")
        if remaining <= 0:
            raise RequestDispatchTimeout("Caller execution budget has expired")
        self._acquire(self.slots, capability, effective)
        try:
            result = self.executor.run(CommandSpec(Path(cwd), tuple(command),
                timeout=max(1, math.ceil(remaining)), environment=environment, operation="dev",
                cancel_event=_OperationCancellation(self, capability),
                filesystem_access=self._request_access(capability, Path(cwd))))
        finally:
            self.slots.release()
        process = subprocess.CompletedProcess(command,
            result.returncode if result.returncode is not None else 2, result.stdout, result.stderr)
        process.execution_result = result
        return process

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
        self.measurements.environment_identity = identity

    def is_wiki_request(self, route, request):
        return route == "/v1/wiki/query"

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
                reuse_completed=False, reuse_correctness=False, candidate_source=None,
                execution_timeout: int | None = None, reference_snapshot: Path | None = None) -> dict:
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
                snapshot(capability.workspace, staged,
                         custom_tools=bool(capability.journal and capability.journal.view),
                         filesystem_access=capability.filesystem_access)
                if candidate_source is not None:
                    self.workspace_provider.install_source(staged, candidate_source)
                # Input authority belongs to the retained Task, never a mutable
                # Agent copy. Evaluation prepares its selected references.
                configuration = replace(self.evaluation_configuration, environment=environment)
                evaluation_input = self.evaluation_input
                if reference_snapshot is not None:
                    # Only execute_trusted supplies this host-owned probe bundle.
                    # HTTP Agent requests cannot select reference authority.
                    from aka.contracts.content import WorkspaceFile
                    from aka.evaluation.private_inputs import selected_content
                    reference_files = selected_content(evaluation_input)
                    for name in ("input.py", "reference.py", "shapes.json"):
                        reference_files[name] = None
                    files = []
                    for name, item in reference_files.items():
                        data = read_input(reference_snapshot, name) if item is None else item.content
                        files.append(WorkspaceFile(name, text=data.decode("utf-8")))
                    evaluation_input = replace(evaluation_input, files=tuple(files))
                if execution_timeout is not None:
                    if type(execution_timeout) is not int or execution_timeout < 1:
                        raise ValueError("Trusted execution timeout must be positive")
                    configuration = replace(configuration, timeout=execution_timeout)
                converted = request_from_argv(argv, workspace=staged,
                    evaluation_input=evaluation_input, configuration=configuration,
                    workspace_provider=self.workspace_provider, defaults=measurement_defaults())
                args = converted.args
                if converted.kind in QUERY_KINDS:
                    if converted.kind == "kernel-pareto-frontier":
                        declared = request_from_argv(["--kind", "run", "--mode", "full", "--no-sync"],
                            workspace=staged, evaluation_input=self.evaluation_input, configuration=configuration,
                            workspace_provider=self.workspace_provider, defaults=measurement_defaults())
                        self._measurement_environment(declared.request)
                        self.measurements.query_request = declared.request
                    return self.measurements.query(args, capability.workspace)
                if converted.kind == "env":
                    from aka.contracts.execution_environment import EnvironmentQuery
                    query = EnvironmentQuery(hardware=args.env_gpu or configuration.hardware,
                        fresh=args.env_force)
                    facts = self.executor.query_environment(query)
                    return {"exit_code": 0, "stdout": json.dumps(facts.as_dict(), allow_nan=False)+"\n", "stderr": ""}
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
            request_id = "trusted-" + request_digest if capability.request_id == "acceptance" else capability.request_id
            response = self._execute_measurement(command, request_digest=request_digest, request_id=request_id,
                reuse_completed=reuse_completed, reuse_correctness=reuse_correctness)
            self.audit_process(capability, "gateway", argv, subprocess.CompletedProcess([], response["exit_code"], response["stdout"], response["stderr"]),
                (response.get("agent_result", {}).get("identity") or {}).get("gateway_record_id"))
            self.publish_evaluation_log(capability.workspace, command.workspace)
            if response["exit_code"] == 0:
                self.publish_legacy_outputs(capability.workspace, command.workspace, args)
            return response

    def _execute_measurement(self, command, *, request_digest, request_id, **options):
        # A live equivalent call has one owner even across different Sessions.
        # Durable completed/recovery state remains exclusively in Experiment.
        with self.lock:
            active = getattr(self, "_active_measurements", None)
            if active is None:
                active = self._active_measurements = {}
            if request_digest in active:
                original, deadline = active[request_digest]
                value = {"code": "gateway_task_running", "status": "running",
                    "message": "The same request is still running. Wait for its original response.",
                    "original_request_id": original,
                    "remaining_seconds": max(0, math.ceil(deadline - time.monotonic())),
                    "estimate_basis": "runtime_request_deadline", "completion_guaranteed": False,
                    "gateway_record_id": None}
                return {"exit_code": 75, "stdout": json.dumps(value) + "\n", "stderr": "",
                        "agent_result": value}
            active[request_digest] = (request_id, time.monotonic() + command.timeout)
        try:
            return self.measurements.execute(command, request_id=request_id, **options)
        finally:
            with self.lock:
                active.pop(request_digest, None)

    def _validate_candidate_correctness(self, capability: Capability, source) -> dict:
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
            source_identity=self.workspace_provider.source_identity,
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
                return {"exit_code": 0, "stdout": json.dumps(self.plugins.catalog()) + "\n", "stderr": ""}
            if set(request) != {"action", "tool", "input"} or request["action"] != "call":
                raise ValueError("Use action=list, or action=call with tool and input")
            if not isinstance(request["tool"], str):
                raise ValueError("tool must be a catalog name")
        if not self._live(capability):
            raise SessionRevokedError("Session revoked before tool invocation")
        from aka.core.plugin_runtime.schema import PluginError
        try:
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
        return [sys.executable, str(self.plugins.resource_file("gpu-wiki", f"tools/{tool}.py")), *argv]

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
            from aka.core.filesystem import read_regular_bytes
            data = read_regular_bytes(log, limit=MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Evaluation log exceeds the size limit")
            binding = self.journals.get(workspace)
            destination = binding.service.git_workspace if binding and binding.view else workspace
            publish(destination, EPISODE_EVALUATIONS_PATH,
                    data, append=True)
