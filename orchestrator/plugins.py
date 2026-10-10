"""Operator-owned plugin catalog; only public Skills enter Agent workspaces."""
from __future__ import annotations

from pathlib import Path

from plugin_runtime import PluginError
from plugin_runtime import PluginRegistry as Registry
from plugin_runtime.schema import validate_schema
from plugin_runtime.files import read_text

from .constants import REPO_ROOT

PLUGIN_DIR = REPO_ROOT / "plugins"


class PluginRegistry(Registry):
    def __init__(self, *, plugin_root: Path | None = None):
        super().__init__(PLUGIN_DIR, plugin_root=plugin_root)

    def validate_call(self, name: str, value: object) -> dict:
        for plugin in self.plugins:
            for key, tool in plugin.tools.items():
                if name == f"{plugin.id}.{key}":
                    validate_schema(tool["input_schema"], value)
                    return tool
        raise PluginError("tool_not_found", "Tool is not enabled; use tools/plugin.py list")

    def public_catalog(self) -> dict:
        return {
            "tools": self.catalog(),
            "skills": [dict(row, path=f"skills/{row['name']}") for row in self.skill_catalog()],
        }

    def instructions(self, phase: str, **values: str) -> str:
        from plugin_runtime.registry import local_file
        if not self.plugins:
            return ""
        parts = [
            "Discover enabled plugin tools with `python3 tools/plugin.py list`; "
            "call `python3 tools/plugin.py call <plugin.tool> --input scratch/request.json`. "
            "Tools run through the Supervisor; plugin code and resources are not workspace files."
        ]
        for plugin in self.plugins:
            for key in dict.fromkeys(("common", phase)):
                relative = plugin.manifest.get("instructions", {}).get(key)
                if relative:
                    text = read_text(local_file(plugin.root, relative))
                    for name, value in values.items():
                        text = text.replace("{{" + name + "}}", str(value))
                    parts.append(text)
        return "\n\n".join(parts)

    def install_agent_skills(self, workspace: Path) -> list[str]:
        from .agent_skill_manifest import SKILL_MANIFEST
        from .episode_workspace import bounded_tree_entries
        from .workspace_runtime import RETIRED_SKILL_NAMES
        from supervisor.workspace import publish

        names = []
        for plugin in self.plugins:
            for name, skill in plugin.manifest.get("skills", {}).items():
                if name in RETIRED_SKILL_NAMES:
                    raise PluginError("mount_conflict", f"Retired workflow Skill cannot be installed: {name}")
                source = (plugin.root / skill["path"]).resolve()
                if not (source / "SKILL.md").is_file():
                    continue
                if name in SKILL_MANIFEST:
                    expected = (REPO_ROOT / SKILL_MANIFEST[name].root).resolve()
                    if source != expected:
                        raise PluginError("mount_conflict", f"Plugin cannot replace Runtime Skill {name}")
                    continue
                if name in names:
                    raise PluginError("mount_conflict", f"Duplicate plugin Skill {name}")
                names.append(name)
                for relative, content in bounded_tree_entries(source):
                    publish(workspace, f"skills/{name}/{relative}", content)
        return names
