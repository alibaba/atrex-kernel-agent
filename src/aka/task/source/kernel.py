"""Describe the original kernel copy without writing a workspace."""
from dataclasses import dataclass
from pathlib import Path
from aka.contracts.content import SourceContent, WorkspaceFile


@dataclass(frozen=True)
class KernelSourceProvider:
    kernel_demo: str

    def prepare(self, request=None):
        return SourceContent((WorkspaceFile("kernel.py", source=Path(self.kernel_demo), copy="cp"),))

