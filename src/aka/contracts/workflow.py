"""Campaign entry with explicitly borrowed dependencies and preserved stop reasons."""

from typing import Protocol
from pathlib import Path
from collections.abc import Callable


class Campaign(Protocol):
    workspace: Path

    def prepare(self, *, recovery_ready: Callable[[], None] | None = None) -> None:
        """Prepare or resume the baseline before signalling the host as ready."""
        ...

    def run(self) -> str:
        """Run the baseline and episode sequence and return its stop reason."""
        ...


class Workflow(Protocol):
    def campaign(self, **configuration: object) -> Campaign:
        """Construct a configured campaign without model or evaluator calls."""
        ...
