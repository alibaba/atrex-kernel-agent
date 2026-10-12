"""Expose the long-horizon source through the neutral observation seam."""
from pathlib import Path

from aka.contracts.observation import OBSERVATION_VERSION, ObservationReader
from .reader import WorkspaceReader
from .demo import create_demo_workspace

name = "long-horizon"
provide = ("observation_source",)
identity_files = ("__init__.py", "plugin.py", "reader.py", "demo.py")
identity_packages = ("aka.contracts",)


class LongHorizonSource:
    protocol_version = OBSERVATION_VERSION

    def open(self, workspace: Path) -> ObservationReader:
        return WorkspaceReader(workspace)

    def create_preview(self, temporary_directory: Path) -> Path:
        return create_demo_workspace(temporary_directory)


def apply(ctx, config):
    ctx.provide("observation_source", LongHorizonSource())
