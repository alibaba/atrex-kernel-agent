"""Compatibility adapter for callers of the former ABBA executor wrapper."""
from aka.contracts.execution import JobResult


def check_abba_result(result: JobResult) -> None:
    result.require_transport()
    if result.returncode:
        raise RuntimeError(f"gateway ABBA command exited {result.returncode}: "
                           + (result.stdout + "\n" + result.stderr)[-3000:])


class ABBAExecutor:
    def __init__(self, executor):
        self.executor = executor

    @property
    def capabilities(self):
        return self.executor.capabilities

    @property
    def resource_identity(self):
        return self.executor.resource_identity

    def run(self, command):
        result = self.executor.run(command)
        result.require_transport()
        return result
