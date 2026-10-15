"""Map retained sandbox settings to invocation environment for Workflow."""

from aka.legacy.agent.session import run_session
from aka.legacy.execution.sandbox import sandbox_directive, fast_sandbox_directive


class CampaignExecution:
    def __init__(self, settings):
        self.settings = settings
        for name in ("hardware", "profile", "url", "ssh", "ssh_init", "ssh_gpu", "health_command"):
            setattr(self, "sandbox_" + name, getattr(settings, name))
        self.sandbox_timeout = settings.timeout

    def environment(self):
        environment = {}
        if self.sandbox_ssh:
            environment["ATREX_SANDBOX_SSH"] = self.sandbox_ssh
            environment["ATREX_SANDBOX_SSH_INIT"] = self.sandbox_ssh_init
            if self.sandbox_ssh_gpu is not None:
                environment["ATREX_SANDBOX_SSH_GPU"] = str(self.sandbox_ssh_gpu)
            if self.sandbox_health_command:
                environment["ATREX_SANDBOX_HEALTH_COMMAND"] = (
                    self.sandbox_health_command
                )
        return environment

    def directive(self, *, fast=False):
        render = fast_sandbox_directive if fast else sandbox_directive
        return render(
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_ssh,
        )

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
            sandbox_hardware=self.sandbox_hardware if remote else "",
            sandbox_profile=self.sandbox_profile if remote else "",
            sandbox_url=self.sandbox_url if remote else "",
            sandbox_ssh=self.sandbox_ssh if remote else "",
            sandbox_ssh_init=self.sandbox_ssh_init if remote else "",
            sandbox_health_command=self.sandbox_health_command if remote else "",
            sandbox_timeout=self.sandbox_timeout if remote else 600,
            extra_environment=environment,
            **options,
        )

    def framework_baseline_directive(self) -> str:
        """Concise V1-specific boundary without generic repeated full-evaluator examples."""
        endpoint = (
            self.sandbox_ssh
            or self.sandbox_url
            or self.sandbox_profile
            or "agate configuration"
        )
        hardware = self.sandbox_hardware or "the configured remote GPU"
        return (
            "## V1 GPU sandbox boundary\n\n"
            f"- Target `{hardware}` through `{endpoint}`. Every GPU import, compile, smoke, "
            "correctness check, and timer must cross `tools/sandbox.py`; never execute "
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
