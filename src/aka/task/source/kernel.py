"""Describe the original kernel copy without writing a workspace."""
from dataclasses import dataclass
from pathlib import Path
from aka.contracts.content import SourceContent, WorkspaceFile
from .manifest import prepare_candidate_manifest


@dataclass(frozen=True)
class KernelSourceProvider:
    kernel_demo: str

    prepare_candidate_manifest = staticmethod(prepare_candidate_manifest)

    def prepare(self, request=None):
        return SourceContent((WorkspaceFile("kernel.py", source=Path(self.kernel_demo), copy="cp"),))

