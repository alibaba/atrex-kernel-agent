"""Explicit internal variation for native observation, independent of Workflow."""


class SessionHooks:
    def prepare_environment(self, workspace, environment, overrides, session_id):
        return self.invocation_environment(environment, overrides)

    def invocation_environment(self, environment, overrides):
        values = dict(environment)
        for key, value in (overrides or {}).items():
            if value is None:
                values.pop(str(key), None)
            else:
                values[str(key)] = str(value)
        return values

    def observer(self, environment):
        from .codex_ledger import CodexSessionLedgerObserver, codex_home
        return CodexSessionLedgerObserver(codex_home(environment))

    def stream_usage(self, runtime_id, stdout):
        from .runtime import terminal_usage_from_stream
        return terminal_usage_from_stream(stdout)

    def observe(self, runtime_id, stdout, session_id, observer, backend):
        from .native import normalize_stream
        return normalize_stream(runtime_id, stdout, session_id=session_id,
                                codex_observer=observer, adapter=backend)

    def check_invocation(self, invoked, resume):
        pass
