"""Own selected CLI adapters while preserving the existing per-call settings policy."""

from threading import RLock


class LegacyAgent:
    def __init__(self, *, process_runner=None, registry=None):
        self._options = {}
        if process_runner is not None:
            self._options["process_runner"] = process_runner
        if registry is not None:
            self._options["registry"] = registry
        self._runtimes = {}
        self._closed = False
        self._lock = RLock()
        from weakref import WeakSet

        self._sessions = WeakSet()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Agent selection has been closed")

    def runtime(self, runtime_id):
        from .runtime import build_agent_runtime

        with self._lock:
            self._check_open()
            if runtime_id not in self._runtimes:
                runtime = build_agent_runtime(
                    runtime_id, check_open=self._check_open, **self._options
                )
                self._runtimes[runtime_id] = runtime
            return self._runtimes[runtime_id]

    def open_session(self, runtime_id, workspace, *, session_id="", environment=None):
        import uuid
        from .native import NativeSession
        from orchestrator.agent_runtime.process import run_bounded

        with self._lock:
            self._check_open()
            runtime = self.runtime(runtime_id)
            process = self._options.get("process_runner", run_bounded)

            def executor(command, cwd, timeout, env):
                return process(command, cwd=cwd, timeout=timeout, env=env)

            session = NativeSession(
                runtime_id,
                workspace,
                session_id or str(uuid.uuid4()),
                executor,
                environment,
                self._check_open,
                runtime=runtime,
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
