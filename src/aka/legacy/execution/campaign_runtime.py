"""Map retained sandbox settings to invocation environment for Workflow."""

from aka.legacy.agent.session import run_session
from aka.legacy.execution.sandbox import sandbox_directive, fast_sandbox_directive


class CampaignExecution:
    def __init__(self, settings):
        self.settings = settings
        from aka.legacy.execution.configuration import selected_ssh, legacy_settings
        self.uses_ssh = selected_ssh()
        from .selection import selected_execution
        self.connection = legacy_settings() if selected_execution().get("profile", "legacy") == "legacy" else None

    def environment(self):
        from aka.legacy.execution.configuration import legacy_settings
        from aka.legacy.execution.selection import selected_execution
        if selected_execution().get("profile", "legacy") != "legacy":
            from aka.legacy.execution.environment import session_environment
            return session_environment(hardware=self.settings.hardware, timeout=self.settings.timeout)
        s = legacy_settings()
        from aka.legacy.execution.environment import session_environment
        return session_environment(hardware=self.settings.hardware, timeout=self.settings.timeout,
                                   profile=s.profile, url=s.url, ssh=s.ssh, ssh_init=s.ssh_init,
                                   health_command=s.health_command)

    def directive(self, *, fast=False):
        from aka.legacy.execution.configuration import legacy_settings
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
