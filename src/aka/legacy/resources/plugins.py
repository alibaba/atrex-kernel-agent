"""AKA bindings for the reusable tools/Skills runtime."""

from __future__ import annotations

from pathlib import Path
import os
from contextvars import ContextVar

_selected_catalog = ContextVar("aka_tool_catalog", default=None)

from plugin_runtime import Plugin, PluginError
from plugin_runtime import PluginRegistry as Registry
from plugin_runtime.registry import STATE_DIR
from plugin_runtime.schema import decode_json, read_json

from aka.legacy.resources.paths import REPO_ROOT
from aka.legacy.execution import selection as execution

from aka.legacy.resources.registry import AgentSkillsRegistryMixin

PLUGIN_DIR = REPO_ROOT / "plugins"


class PluginRegistry(AgentSkillsRegistryMixin, Registry):
    def __init__(self, plugin_dir: Path | str | None = None):
        super().__init__(plugin_dir if plugin_dir is not None else
                         (_selected_catalog.get() or os.environ.get("ATREX_PLUGIN_DIR", PLUGIN_DIR)))
        self.bind_agent_skills(execution.selected_agent_skills())

    def instructions(self, phase: str, **values: str) -> str:
        return (
            "Use `python3 tools/plugin.py list` to discover enabled tools and Skills. "
            "Invoke a tool with `python3 tools/plugin.py call <plugin.tool> --input <request.json>`.\n\n"
            + super().instructions(phase, **values)
        )

    def environment(self, workspace: Path, campaign_name: str = "") -> dict[str, str]:
        environment = super().environment(workspace, {"campaign_name": campaign_name})
        reserved = {
            "ATREX_PLUGIN_WORKSPACE": str(workspace.resolve()),
            "ATREX_PLUGIN_DIR": str(self.plugin_dir),
        }
        if environment.keys() & reserved.keys():
            raise PluginError(
                "invalid_manifest", "plugin environment conflicts with AKA context"
            )
        return {**environment, **reserved}

    def _tool_environment(self, plugin: Plugin) -> dict[str, str]:
        return {
            **super()._tool_environment(plugin),
            "ATREX_PLUGIN_ROOT": str(plugin.root),
        }


__all__ = [
    "PLUGIN_DIR",
    "STATE_DIR",
    "PluginError",
    "PluginRegistry",
    "decode_json",
    "read_json",
]
