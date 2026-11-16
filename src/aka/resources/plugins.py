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
    def __init__(self, plugin_dir: Path | str | None = None, *, plugins=None, resources=None, agent_skills=None):
        scoped = _selected_scope.get() is not None
        if plugin_dir is None:
            plugin_dir = _selected_catalog.get() if scoped else os.environ.get("ATREX_PLUGIN_DIR")
        if plugins is None:
            plugins = _selected_plugins.get() if scoped else decode_json(os.environ.get("ATREX_PLUGIN_PATHS", "[]"))
        from .wiki.providers import selected_provider
        self.wiki_provider = selected_provider()
        plugins = tuple(plugins or ())
        if self.wiki_provider is not None:
            plugins = (*plugins, self.wiki_provider.catalog_root)
        super().__init__(plugin_dir, plugins=plugins)
        if resources is None or agent_skills is None:
            selected, skills = selected_resources()
            resources = selected if resources is None else resources
            agent_skills = skills if agent_skills is None else agent_skills
        self.bind_agent_skills(agent_skills)
        self.bind_workspace_resources(resources)

    def instructions(self, phase: str, **values: str) -> str:
        return ("Use `python -m aka.resources.cli list` to discover enabled tools and Skills. "
                "Invoke `python -m aka.resources.cli call <plugin.tool> --input <request.json>`.\n\n"
                + super().instructions(phase, **values))

    def wiki_prompt(self, slot: str) -> str | None:
        """Read a selected provider's replacement for a Workflow Wiki fragment."""
        from .plugin_runtime.registry import local_file
        from aka.core.filesystem import read_regular_bytes
        for plugin in self.plugins:
            relative = plugin.manifest.get("instructions", {}).get("wiki_" + slot)
            if relative:
                return read_regular_bytes(local_file(plugin.root, relative), limit=128 * 1024).decode().strip()
        return None

    def resource_file(self, name: str, relative: str) -> Path:
        """Return a member of exactly one selected declared resource, mounted or private."""
        if not isinstance(name, str) or not name or not isinstance(relative, str) or not relative:
            raise PluginError("invalid_config", "Resource member requires a name and relative path")
        member = Path(relative)
        if member.is_absolute() or ".." in member.parts:
            raise PluginError("invalid_config", "Resource member requires a name and relative path")
        matches = [resource[0] for plugin in self.plugins
                   if (resource := plugin.resources.get(name)) is not None]
        if len(matches) != 1:
            raise PluginError("invalid_config", "Resource name is missing or ambiguous in the selection")
        root = matches[0].resolve()
        path = (root / member).resolve()
        if not root.is_dir() or not path.is_relative_to(root) or not path.is_file():
            raise PluginError("invalid_manifest", "Selected resource member is missing or outside its root")
        return path

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
