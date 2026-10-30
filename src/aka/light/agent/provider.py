"""One Agent owner; Workflow retains retries, handoff and promotion decisions."""
from dataclasses import dataclass
import json
import threading
import uuid
from weakref import WeakSet

from aka.contracts.agent import AgentRunResult, AgentCapabilities
from aka.light.orchestrator.agent_runtime.model import AgentRunRequest as LegacyRequest
from aka.light.orchestrator.agent_runtime.runtime import build_agent_runtime
from aka.light.orchestrator.agent_runtime.process import run_bounded
from aka.light.runtime.paths import SESSION_SHELL_GUARD


@dataclass(frozen=True)
class AgentOptions:
    sandbox: str = "none"
    capture: bool = False
    bwrap_executable: str = "bwrap"
    read_only_paths: tuple[str, ...] = ()
    proc_mode: str = "private"
    timeout_seconds: float = 28800

    def __post_init__(self):
        from aka.agent.provider import validate_timeout
        if self.sandbox not in {"none", "bwrap"} or self.proc_mode not in {"private", "projected"}:
            raise ValueError("Invalid Agent sandbox or proc mode")
        validate_timeout(self.timeout_seconds)
        if self.timeout_seconds is None:
            raise ValueError("timeout_seconds must be positive")

    def environment(self):
        return {
            "ATREX_AGENT_SANDBOX": self.sandbox,
            "ATREX_SESSION_CAPTURE": "1" if self.capture else "0",
            "ATREX_BWRAP_EXECUTABLE": self.bwrap_executable,
            "ATREX_AGENT_READ_ONLY_PATHS": json.dumps(self.read_only_paths),
            "ATREX_AGENT_PROC_MODE": self.proc_mode,
            "ATREX_AGENT_TIMEOUT_SECONDS": str(self.timeout_seconds),
        }


class _Runtime:
    def __init__(self, owner, runtime):
        self.owner, self.runtime = owner, runtime
        self.id = runtime.id

    def run(self, request):
        self.owner.check_open()
        environment = dict(request.environment or {})
        environment.update(self.owner.options.environment())
        token = self.owner._access.set(request.access)
        try:
            result = self.runtime.run(LegacyRequest(
                workspace=request.workspace, prompt=request.prompt,
                timeout_s=request.timeout_s, reasoning_effort=request.reasoning_effort,
                session_id=request.session_id, extra_environment=environment,
                observation_directory=request.observation_directory,
            ))
        finally:
            self.owner._access.reset(token)
        return AgentRunResult(
            runtime_id=self.id, exit_status=result.exit_status, timed_out=result.timed_out,
            terminal_usage=result.terminal_usage, events=result.events,
            capabilities=result.capabilities, observation_errors=result.observation_errors,
            stdout_tail=result.stdout_tail, stderr_tail=result.stderr_tail,
            session_id=result.session_id,
        )


class LightAgent:
    """Public PR114 Agent contract, with no process-global policy overrides."""
    supports_tool_runtime = True

    def __init__(self, options=AgentOptions(), *, process_runner=run_bounded, environment=None):
        import os
        from contextvars import ContextVar
        self._environment = dict(os.environ if environment is None else environment)
        if options.sandbox == "bwrap" and process_runner is not run_bounded:
            raise ValueError("Bubblewrap requires the FD-aware owned process runner")
        self._access = ContextVar("light_agent_access", default=None)
        self.options = options
        self._process = process_runner
        self._runtimes = {}
        self._sessions = WeakSet()
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._controls = {}
        self._closed = False

    @property
    def capabilities(self):
        return AgentCapabilities(self.options.sandbox == "bwrap", True)

    @property
    def configuration_identity(self):
        import hashlib
        from dataclasses import asdict
        settings = {key: value for key, value in self._environment.items()
                    if key.endswith("_SESSION_SETTINGS") or key == "ATREX_CLAUDE_SESSION_NAME"}
        return hashlib.sha256(json.dumps({"options": asdict(self.options), "settings": settings},
                                        sort_keys=True).encode()).hexdigest()

    def export_observations(self, workspace, destination):
        from aka.agent.public_output import export_observations
        return export_observations(workspace, destination)

    def export_checkpoint(self, workspace, destination):
        from aka.agent.checkpoint import export_checkpoint
        return export_checkpoint(workspace, destination)

    def validate_checkpoint(self, snapshot, *, check_destinations=False):
        from aka.agent.checkpoint import validate_checkpoint
        return validate_checkpoint(snapshot, check_destinations=check_destinations)

    def restore_checkpoint(self, snapshot):
        from aka.agent.checkpoint import restore_checkpoint
        return restore_checkpoint(snapshot)

    def check_open(self):
        if self._closed:
            raise RuntimeError("Agent selection has been closed")

    def _execute(self, command, cwd, timeout, env=None, *, access=None):
        environment = dict(env or {})
        requested_timeout = environment.get("ATREX_AGENT_TIMEOUT_SECONDS")
        environment.update(self.options.environment())
        if timeout is None and requested_timeout is not None:
            from aka.agent.provider import validate_timeout
            requested_timeout = float(requested_timeout)
            validate_timeout(requested_timeout)
            environment["ATREX_AGENT_TIMEOUT_SECONDS"] = str(min(
                self.options.timeout_seconds, requested_timeout,
            ))
        from aka.agent.execution.control import ProcessControl
        with self._lock:
            self.check_open()
            control = ProcessControl() if self._process is run_bounded else None
            if control is not None:
                self._controls[control] = environment.get("ATREX_AGENT_PREPARED_HOME")
        if control is None:
            # A custom runner retains its own process ownership contract.
            return self._process(command, cwd=cwd, timeout=timeout, env=environment)
        try:
            return self._process(command, cwd=cwd, timeout=timeout, env=environment,
                                 session_control=control, agent_access=access or self._access.get())
        finally:
            control.close()
            with self._idle:
                self._controls.pop(control, None)
                self._idle.notify_all()

    def _cancel(self, environment):
        home = environment.get("ATREX_AGENT_PREPARED_HOME")
        with self._lock:
            controls = tuple(c for c, value in self._controls.items() if home and value == home)
        for control in controls:
            control.kill()

    def runtime(self, runtime_id):
        with self._lock:
            self.check_open()
            if runtime_id not in self._runtimes:
                runtime = build_agent_runtime(runtime_id, process_runner=self._execute, environment=self._environment)
                self._runtimes[runtime_id] = _Runtime(self, runtime)
            return self._runtimes[runtime_id]

    def open_session(self, runtime_id, workspace, *, session_id="", environment=None, output_directory=None, access=None):
        from .native import NativeSession
        with self._lock:
            self.check_open()
            # The shared NativeSession constructor otherwise selects Legacy's
            # BASH_ENV. Light mounts its own packaged guard for both call paths.
            values = {"BASH_ENV": str(SESSION_SHELL_GUARD), **(environment or {})}
            values.update(self.options.environment())
            session = NativeSession(
                runtime_id, workspace, session_id or str(uuid.uuid4()),
                lambda command, cwd, timeout, env: self._execute(command, cwd, timeout, env, access=access),
                values, self.check_open, runtime=self.runtime(runtime_id).runtime,
                cancel_environment=self._cancel,
                output_directory=output_directory,
            )
            self._sessions.add(session)
            return session

    def close(self):
        with self._lock:
            self._closed = True
            controls = tuple(self._controls)
            sessions = tuple(self._sessions)
        for control in controls:
            control.kill()
        with self._idle:
            self._idle.wait_for(lambda: not self._controls, timeout=10)
        for session in sessions:
            session.close()
        with self._lock:
            self._sessions.clear()
            self._runtimes.clear()
