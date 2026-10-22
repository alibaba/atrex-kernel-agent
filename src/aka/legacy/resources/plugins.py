"""Installed Resources bindings for the selected tools, Skills and runtime files."""
from pathlib import Path
import os
import json
from contextvars import ContextVar

from aka.core.plugin_runtime.schema import PluginError, decode_json, read_json
from .plugin_runtime import Plugin
from .plugin_runtime.registry import PluginRegistry as Registry, STATE_DIR
from .registry import AgentSkillsRegistryMixin
from .contributions import _selected_scope, selected_resources, selection_environment, SELECTION_ENV

_selected_catalog = ContextVar("aka_tool_catalog", default=None)
_selected_plugins = ContextVar("aka_explicit_tool_plugins", default=None)


class PluginRegistry(AgentSkillsRegistryMixin, Registry):
    def __new__(cls, plugin_dir=None, *, plugins=None, resources=None, agent_skills=None):
        if cls is PluginRegistry:
            from .core_adapter import CoreResourcesFacade, selected_core_resources
            bound = selected_core_resources()
            if bound is not None:
                if (plugin_dir is not None or plugins is not None or resources is not None
                        or agent_skills is not None or os.environ.get("ATREX_PLUGIN_DIR")
                        or os.environ.get("ATREX_PLUGIN_PATHS") not in (None, "", "[]")):
                    raise PluginError("plugin_changed", "cannot override a frozen Core tool selection")
                return CoreResourcesFacade(bound)
        return super().__new__(cls)

    def __init__(self, plugin_dir: Path | str | None = None, *, plugins=None, resources=None, agent_skills=None):
        scoped = _selected_scope.get() is not None
        if plugin_dir is None:
            plugin_dir = _selected_catalog.get() if scoped else os.environ.get("ATREX_PLUGIN_DIR")
        if plugins is None:
            plugins = _selected_plugins.get() if scoped else decode_json(os.environ.get("ATREX_PLUGIN_PATHS", "[]"))
        super().__init__(plugin_dir, plugins=plugins if plugins is not None else ())
        if resources is None or agent_skills is None:
            selected, skills = selected_resources()
            resources = selected if resources is None else resources
            agent_skills = skills if agent_skills is None else agent_skills
        self.bind_agent_skills(agent_skills)
        self.bind_workspace_resources(resources)

    def instructions(self, phase: str, **values: str) -> str:
        return ("Use `python -m aka.legacy.resources.cli list` to discover enabled tools and Skills. "
                "Invoke `python -m aka.legacy.resources.cli call <plugin.tool> --input <request.json>`.\n\n"
                + super().instructions(phase, **values))

    def environment(self, workspace: Path, campaign_name: str = "") -> dict[str, str]:
        environment = super().environment(workspace, {"campaign_name": campaign_name})
        reserved = {"ATREX_PLUGIN_WORKSPACE": str(workspace.resolve()),
                    "ATREX_PLUGIN_DIR": str(self.plugin_dir) if self.plugin_dir else "",
                    "ATREX_PLUGIN_PATHS": json.dumps([str(p) for p in self.plugin_paths]),
                    SELECTION_ENV: selection_environment(self._workspace_resources, self._agent_skills)}
        if environment.keys() & reserved.keys():
            raise PluginError("invalid_manifest", "plugin environment conflicts with AKA context")
        return {**environment, **reserved}

    def _tool_environment(self, plugin: Plugin) -> dict[str, str]:
        return {**super()._tool_environment(plugin), "ATREX_PLUGIN_ROOT": str(plugin.root)}
