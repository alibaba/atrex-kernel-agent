"""Workflow host for Agent sessions and campaign execution instructions."""

from aka.legacy.workflow.agent_session import run_session
from .execution_instructions import sandbox_directive, fast_sandbox_directive


class CampaignExecution:
    def __init__(self, settings, *, environment=None):
        self.settings = settings
        # Session instructions expose declared execution configuration, not the
        # host operation's inherited environment or private evaluator binding.
        names = {"ATREX_SANDBOX_GPU", "ATREX_SANDBOX_TIMEOUT", "ATREX_SANDBOX_PROFILE",
            "ATREX_SANDBOX_URL", "ATREX_SANDBOX_SSH", "ATREX_SANDBOX_SSH_INIT",
            "ATREX_SANDBOX_SSH_GPU", "ATREX_SANDBOX_HEALTH_COMMAND", "ATREX_SANDBOX_SSH_RUNTIME_BINDS",
            "ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND", "ATREX_ENVIRONMENT_STATE_FILE", "AKA_EXECUTION_SELECTION"}
        self._environment = {key: value for key, value in (environment or {}).items() if key in names}
        self.uses_ssh = bool(self._environment.get("ATREX_SANDBOX_SSH"))
        from types import SimpleNamespace
        self.connection = SimpleNamespace(
            profile=self._environment.get("ATREX_SANDBOX_PROFILE", ""),
            url=self._environment.get("ATREX_SANDBOX_URL", ""),
            ssh=self._environment.get("ATREX_SANDBOX_SSH", ""),
            ssh_init=self._environment.get("ATREX_SANDBOX_SSH_INIT", ""),
            health_command=self._environment.get("ATREX_SANDBOX_HEALTH_COMMAND", ""))

    def environment(self):
        s = self.connection
        from aka.legacy.execution.environment import session_environment
        return session_environment(hardware=self.settings.hardware, timeout=self.settings.timeout,
                                   profile=s.profile, url=s.url, ssh=s.ssh, ssh_init=s.ssh_init,
                                   health_command=s.health_command, extra=self._environment)

    def directive(self, *, fast=False):
        render = fast_sandbox_directive if fast else sandbox_directive
        s = self.connection
        return render(self.settings.hardware, s.profile if s else "", s.url if s else "", s.ssh if s else "")

    def run_session(
        self,
        workspace,
        prompt,
        timeout,
        *,
        agent_cli,
        runtime,
        environment=None,
        remote=False,
        **options,
    ):
        return run_session(
            workspace,
            prompt,
            timeout,
            agent_cli=agent_cli,
            runtime=runtime,
            extra_environment={**(self.environment() if remote else {}), **(environment or {})},
            **options,
        )

    def framework_baseline_directive(self) -> str:
        """Concise V1-specific boundary without generic repeated full-evaluator examples."""
        endpoint = (
            ((self.connection.ssh or self.connection.url or self.connection.profile) if self.connection else "")
            or "agate configuration"
        )
        hardware = self.settings.hardware or "the configured remote GPU"
        return (
            "## V1 GPU sandbox boundary\n\n"
            f"- Target `{hardware}` through `{endpoint}`. Every GPU import, compile, smoke, "
            "correctness check, and timer must cross `python -m aka.legacy.execution.cli`; never execute "
            "`kernel.py`, `test_kernel.py`, a profiler, or a JIT-capable GPU import on the host.\n"
            "- Use only the bounded smoke command below during the ordinary V1 turn. Do not "
            "run a full-workload evaluator, `--multi-seed`, a separate benchmark, or profiling.\n"
            "- Keep `--no-memory`: the supervisor parses evaluator output and owns canonical "
            "memory. Sandbox uploads are allowlist-only; declare inputs for any custom smoke "
            "helper, and never upload optimizer memory or private evaluator inputs.\n"
            "- The remote executor is supervisor-owned infrastructure. Do not start, stop, "
            "restart, signal, reconfigure, or cancel its jobs. Report an infrastructure "
            "failure and exit.\n"
        )
