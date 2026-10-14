"""Original ABBA transport failure and retry classification."""
import subprocess

from aka.contracts.execution import JobResult
from orchestrator.infrastructure_retry import InfrastructureUnavailable, check_transport


def check_abba_result(result: JobResult) -> None:
    code = result.transport_returncode or (result.exit_code or 0)
    if code == 0:
        return
    check_transport(subprocess.CompletedProcess((), code, result.stdout, result.stderr))
    raise RuntimeError(
        f"gateway ABBA command exited {code}: "
        + (result.stdout + "\n" + result.stderr)[-3000:]
    )


class ABBAExecutor:
    """Translate the old ABBA wait timeout for its existing outage retry loop."""
    def __init__(self, executor):
        self.executor = executor

    @property
    def capabilities(self):
        return self.executor.capabilities

    @property
    def resource_identity(self):
        return self.executor.resource_identity

    def run(self, command):
        try:
            return self.executor.run(command)
        except subprocess.TimeoutExpired as exc:
            raise InfrastructureUnavailable("GPU transport wait deadline exceeded") from exc
