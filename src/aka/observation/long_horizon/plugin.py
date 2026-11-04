"""Expose the long-horizon source through the neutral observation seam."""
from pathlib import Path

from aka.contracts.observation import OBSERVATION_VERSION, ObservationReader
from .reader import WorkspaceReader
from .demo import create_demo_workspace

name = "long-horizon"
provide = ("observation_source",)
inject = ("workflow_export",)
identity_files = ("__init__.py", "plugin.py", "reader.py", "artifacts.py", "demo.py", "portable.py")
identity_packages = ("aka.contracts",)


class LongHorizonSource:
    protocol_version = OBSERVATION_VERSION

    def __init__(self, workflow_reader=None):
        self.workflow_reader = workflow_reader

    def open(self, workspace: Path) -> ObservationReader:
        return WorkspaceReader(workspace, self.workflow_reader)

    def create_preview(self, temporary_directory: Path) -> Path:
        return create_demo_workspace(temporary_directory)


def apply(ctx, config):
    ctx.provide("observation_source", LongHorizonSource(ctx.workflow_export))
