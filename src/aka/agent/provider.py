"""One Agent owner combines CLI backends with independent process execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import signal
from threading import Condition, RLock
import uuid

from aka.contracts.agent import AgentAccess, AgentMount
from .backends import DEFAULT_BACKEND_REGISTRY
from .execution.home import PREPARED_ENV, prepare_home
from .execution.native import NativeExecution
from .execution.process import (
    descendant_process_groups,
    run_bounded,
    signal_process_groups,
    spawn_session,
)
from .resources import shell_guard_path


def validate_timeout(value):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Agent timeout must be a positive finite number")


@dataclass(frozen=True)
class AgentOptions:
    sandbox: str = "none"
    bwrap_executable: str = "bwrap"
    timeout_seconds: float | None = None
    state_directory: Path | None = None

    def __post_init__(self):
        if self.sandbox not in {"none", "bwrap"}:
            raise ValueError("Agent sandbox must be none or bwrap")
        validate_timeout(self.timeout_seconds)


@dataclass
class _Invocation:
    backend: object
    host_home: Path
    access: AgentAccess
    workspace: Path
    authority: dict[str, str | None]
    active: int = 0
    processes: set = field(default_factory=set)
    cancelled: bool = False


class Agent:
    def __init__(
        self,
        options=AgentOptions(),
        *,
        registry=None,
        process_runner=None,
        spawn=None,
        unavailable=None,
        violation=None,
        host_environment=None,
        shell_guard=None,
        session_name="",
    ):
        self.options = options
        self._registry = registry if registry is not None else DEFAULT_BACKEND_REGISTRY
        self._process_runner = process_runner
        self._spawn = spawn or spawn_session
        self._unavailable = unavailable
        self._violation = violation
        self._host_environment = host_environment
        self._shell_guard = shell_guard
        self._session_name = session_name
        self._runtimes = {}
        self._sessions = set()
        self._invocations = {}
        self._processes = set()
        self._closed = False
        self._lock = RLock()
        self._idle = Condition(self._lock)
        self._running = 0
        self._execution = NativeExecution()
        if options.sandbox == "bwrap":
            if process_runner is not None:
                raise ValueError("Bubblewrap requires an FD-aware spawn adapter")
            from .execution.bubblewrap import BubblewrapExecution

            self._execution = BubblewrapExecution(options.bwrap_executable)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Agent has been closed")

    def _guard_path(self):
        resource = (
            self._shell_guard() if callable(self._shell_guard) else self._shell_guard
        )
        return shell_guard_path(resource)

    def _prepare(
        self,
        workspace,
        session_id,
        backend,
        environment,
        access,
        *,
        explicit_environment=(),
    ):
        self._check_open()
        workspace = Path(workspace).absolute()
        values = dict(environment)
        if any(
            not isinstance(k, str)
            or not k
            or "=" in k
            or "\0" in k
            or not isinstance(v, str)
            or "\0" in v
            for k, v in values.items()
        ):
            raise ValueError("Agent environment names and values must be valid strings")
        # Internal markers are created here, never accepted from caller environments.
        values.pop(PREPARED_ENV, None)
        values.pop("AKA_AGENT_INVOCATION", None)
        access = access if access is not None else AgentAccess()
        if not isinstance(access, AgentAccess):
            raise TypeError("Agent access must be AgentAccess")
        guard = self._guard_path()
        mounts = list(access.mounts)
        authority = {}
        if guard is not None:
            values["BASH_ENV"] = authority["BASH_ENV"] = str(guard)
        if self.options.sandbox == "bwrap":
            standard = {
                "PATH",
                "HOME",
                "TERM",
                "LANG",
                "LANGUAGE",
                "TZ",
                "IS_SANDBOX",
                "ATREX_AGENT_CLI",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "REQUESTS_CA_BUNDLE",
                "GIT_AUTHOR_NAME",
                "GIT_AUTHOR_EMAIL",
                "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL",
                "PIP_ONLY_BINARY",
                "PIP_INDEX_URL",
                "PIP_DISABLE_PIP_VERSION_CHECK",
            }
            permitted = (
                standard
                | set(backend.environment_variables)
                | set(explicit_environment)
                | set(authority)
            )
            permitted.update(key for key in values if key.startswith("LC_"))
            values = {key: value for key, value in values.items() if key in permitted}
            from .execution.bubblewrap import local_path, overlap, system_visible

            workspace = local_path(workspace)
            if any(
                overlap(workspace, Path(root))
                for root in ("/usr", "/bin", "/sbin", "/lib", "/lib64")
            ):
                raise ValueError("Agent workspace overlaps system runtime storage")
            from .execution.bubblewrap import validate_access

            host_before_prepare = (
                Path(values.get("HOME") or Path.home()).expanduser().resolve()
            )
            state_root = Path(
                self.options.state_directory or workspace.parent / ".aka-agent-homes"
            ).absolute()
            validate_access(
                workspace,
                state_root / "pending-home",
                host_before_prepare,
                access,
                home_exists=False,
            )
            if guard is not None:
                if any(overlap(guard, Path(hidden)) for hidden in access.hidden_paths):
                    raise ValueError("Selected shell guard is hidden from the Agent")
                if guard.is_relative_to(workspace):
                    access = AgentAccess(
                        access.mounts,
                        access.hidden_paths,
                        access.read_only_paths
                        if guard in access.read_only_paths
                        else (*access.read_only_paths, guard),
                    )
                elif not system_visible(guard) and not any(
                    m.source == guard and m.target == guard and not m.writable
                    for m in mounts
                ):
                    mounts.append(AgentMount(guard, guard))
            values, host, home = prepare_home(
                workspace,
                session_id,
                backend,
                values,
                state_root=self.options.state_directory,
            )
            names = (
                "HOME",
                "XDG_CONFIG_HOME",
                "XDG_CACHE_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
                PREPARED_ENV,
            )
            authority.update((name, values[name]) for name in names)
            authority.update(
                (name, values[name]) for name, _ in backend.files.environment_roots
            )
        else:
            host = Path(values.get("HOME") or Path.home()).expanduser().resolve()
            state_names = (
                "HOME",
                *(name for name, _ in backend.files.environment_roots),
            )
            authority.update((name, values.get(name)) for name in state_names)
        if self._host_environment is not None:
            values = self._host_environment(values, explicit_environment)
        key = uuid.uuid4().hex
        values["AKA_AGENT_INVOCATION"] = key
        with self._lock:
            self._check_open()
            self._invocations[key] = _Invocation(
                backend,
                host,
                AgentAccess(tuple(mounts), access.hidden_paths, access.read_only_paths),
                workspace,
                authority,
            )
        return values

    def _release(self, environment):
        key = environment.get("AKA_AGENT_INVOCATION", "")
        with self._idle:
            self._idle.wait_for(
                lambda: (
                    key not in self._invocations or not self._invocations[key].active
                ),
                timeout=10,
            )
            self._invocations.pop(key, None)
            self._sessions = {s for s in self._sessions if not s._closed}

    def _cancel(self, environment):
        with self._lock:
            context = self._invocations.get(environment.get("AKA_AGENT_INVOCATION", ""))
            if context is None:
                return
            context.cancelled = True
            processes = tuple(context.processes)
        for process in processes:
            signal_process_groups(
                {process.pid} | descendant_process_groups(process.pid), signal.SIGKILL
            )

    def _spawn_owned(self, context, command, **options):
        # Serializing registration with close prevents a late spawn escaping ownership.
        with self._lock:
            self._check_open()
            if context.cancelled:
                raise RuntimeError("Agent session has been cancelled")
            process = self._spawn(command, **options)
            self._processes.add(process)
            context.processes.add(process)
            return process

    def _execute(self, command, cwd, timeout, env=None):
        limit = timeout if timeout is not None else self.options.timeout_seconds
        validate_timeout(limit)
        values = dict(env or {})
        key = values.pop("AKA_AGENT_INVOCATION", "")
        with self._lock:
            self._check_open()
            context = self._invocations.get(key)
            if context is None or context.workspace != Path(cwd).absolute():
                raise ValueError("Agent invocation has no matching execution context")
            if context.cancelled:
                raise RuntimeError("Agent session has been cancelled")
            if context.active:
                raise RuntimeError("Agent session already has an active invocation")
            context.active += 1
            self._running += 1
        launch = None
        try:
            guard = self._guard_path()
            if guard is not None:
                if context.authority.get("BASH_ENV") != str(guard):
                    raise ValueError(
                        "Selected shell guard changed within an Agent session"
                    )
            for name, expected in context.authority.items():
                if values.get(name) != expected:
                    raise ValueError(f"Agent invocation cannot override owned {name}")
            launch = self._execution.launch(
                command,
                Path(cwd),
                values,
                context.backend,
                context.host_home,
                context.access,
            )
            if self._process_runner is not None:
                return self._process_runner(
                    launch.command, cwd=cwd, timeout=limit, env=launch.environment
                )
            return run_bounded(
                launch.command,
                cwd,
                limit,
                launch.environment,
                spawn=lambda command, **kw: self._spawn_owned(context, command, **kw),
                unavailable=self._unavailable,
                violation=self._violation,
                inherited_descriptors=launch.pass_fds,
            )
        finally:
            if launch is not None:
                launch.close()
            with self._lock:
                context.processes.clear()
                context.active -= 1
                self._running -= 1
                self._processes = {p for p in self._processes if p.poll() is None}
                self._idle.notify_all()

    def runtime(self, runtime_id):
        from .runtime import build_agent_runtime

        with self._lock:
            self._check_open()
            if runtime_id not in self._runtimes:
                self._runtimes[runtime_id] = build_agent_runtime(
                    runtime_id,
                    registry=self._registry,
                    process_runner=self._execute,
                    check_open=self._check_open,
                    session_name=self._session_name,
                    prepare_environment=self._prepare,
                    release_environment=self._release,
                )
            return self._runtimes[runtime_id]

    def open_session(
        self,
        runtime_id,
        workspace,
        *,
        session_id="",
        environment=None,
        output_directory=None,
        access=None,
    ):
        from .native import NativeSession

        with self._lock:
            self._check_open()
            session = NativeSession(
                runtime_id,
                Path(workspace),
                session_id or str(uuid.uuid4()),
                lambda command, cwd, timeout, env: self._execute(
                    command, cwd, timeout, env
                ),
                environment,
                self._check_open,
                runtime=self.runtime(runtime_id),
                output_directory=output_directory,
                prepare_environment=self._prepare,
                access=access,
                release_environment=self._release,
                cancel_environment=self._cancel,
            )
            self._sessions.add(session)
            return session

    def export_checkpoint(self, workspace, destination):
        from .checkpoint import export_checkpoint

        return export_checkpoint(workspace, destination)

    def validate_checkpoint(self, snapshot, *, check_destinations=False):
        from .checkpoint import validate_checkpoint

        return validate_checkpoint(snapshot, check_destinations=check_destinations)

    def restore_checkpoint(self, snapshot):
        from .checkpoint import restore_checkpoint

        return restore_checkpoint(snapshot)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            processes = tuple(self._processes)
        for process in processes:
            signal_process_groups(
                {process.pid} | descendant_process_groups(process.pid), signal.SIGKILL
            )
        with self._idle:
            self._idle.wait_for(lambda: not self._running, timeout=10)
        for session in tuple(self._sessions):
            session.close()
        with self._lock:
            self._sessions.clear()
            self._runtimes.clear()
            self._invocations.clear()
