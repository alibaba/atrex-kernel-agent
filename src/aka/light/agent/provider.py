"""One Agent owner; Workflow retains retries, handoff and promotion decisions."""
from dataclasses import dataclass
import json
import threading
import uuid
from weakref import WeakSet

from aka.contracts.agent import AgentRunResult
from aka.light.agent.runtime.model import AgentRunRequest as LegacyRequest
from aka.light.agent.runtime.runtime import build_agent_runtime
from aka.light.agent.runtime.process import run_bounded
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
        from .execution_clock import validate_timeout
        validate_timeout(self.timeout_seconds, "timeout_seconds")

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
        result = self.runtime.run(LegacyRequest(
            workspace=request.workspace, prompt=request.prompt,
            timeout_s=request.timeout_s, reasoning_effort=request.reasoning_effort,
            session_id=request.session_id, extra_environment=environment,
            observation_directory=request.observation_directory,
        ))
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

    def __init__(self, options=AgentOptions(), *, process_runner=run_bounded):
        self.options = options
        self._process = process_runner
        self._runtimes = {}
        self._sessions = WeakSet()
        self._lock = threading.RLock()
        self._closed = False

    def check_open(self):
        if self._closed:
            raise RuntimeError("Agent selection has been closed")

    def _execute(self, command, cwd, timeout, env=None):
        self.check_open()
        environment = dict(env or {})
        environment.update(self.options.environment())
        return self._process(command, cwd=cwd, timeout=timeout, env=environment)

    def runtime(self, runtime_id):
        with self._lock:
            self.check_open()
            if runtime_id not in self._runtimes:
                runtime = build_agent_runtime(runtime_id, process_runner=self._execute)
                self._runtimes[runtime_id] = _Runtime(self, runtime)
            return self._runtimes[runtime_id]

    def open_session(self, runtime_id, workspace, *, session_id="", environment=None,
                     output_directory=None):
        from .native import NativeSession
        with self._lock:
            self.check_open()
            # The shared NativeSession constructor otherwise selects Legacy's
            # BASH_ENV. Light mounts its own packaged guard for both call paths.
            values = {"BASH_ENV": str(SESSION_SHELL_GUARD), **(environment or {})}
            values.update(self.options.environment())
            session = NativeSession(
                runtime_id, workspace, session_id or str(uuid.uuid4()), self._execute,
                values, self.check_open, runtime=self.runtime(runtime_id).runtime,
                output_directory=output_directory,
            )
            self._sessions.add(session)
            return session

    def close(self):
        with self._lock:
            self._closed = True
            for session in self._sessions:
                session.close()
            self._sessions.clear()
            self._runtimes.clear()
