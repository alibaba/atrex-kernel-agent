"""Own selected CLI adapters while preserving the existing per-call settings policy."""

from threading import RLock


class LegacyAgent:
    def __init__(self, *, process_runner=None, registry=None, session_name=""):
        self._options = {"session_name":session_name}
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

    def open_session(self, runtime_id, workspace, *, session_id="", environment=None, output_directory=None):
        import uuid
        from .native import NativeSession
        from aka.legacy.agent.process import run_bounded

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
                runtime=runtime, output_directory=output_directory,
            )
            self._sessions.add(session)
            return session

    def export_checkpoint(self, workspace, destination):
        from .checkpoint import export_checkpoint
        return export_checkpoint(workspace, destination)

    def validate_checkpoint(self, snapshot, *, check_destinations=False):
        from .checkpoint import validate_checkpoint
        validate_checkpoint(snapshot, check_destinations=check_destinations)

    def restore_checkpoint(self, snapshot):
        from .checkpoint import restore_checkpoint
        restore_checkpoint(snapshot)

    def close(self):
        with self._lock:
            self._closed = True
            for session in self._sessions:
                session.close()
            self._sessions.clear()
            self._runtimes.clear()
