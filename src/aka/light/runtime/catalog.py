"""Operator-owned plugin catalog; only public Skills enter Agent workspaces."""
from __future__ import annotations

from pathlib import Path

from aka.light.runtime.plugins import PluginError
from aka.light.runtime.plugins import PluginRegistry as Registry
from aka.light.runtime.plugins.schema import validate_schema
from aka.light.runtime.plugins.files import read_text

from aka.light.workflow.constants import REPO_ROOT

PLUGIN_DIR = Path(__file__).resolve().parent / "catalogs"


class PluginRegistry(Registry):
    def __init__(self, *, plugin_root: Path | None = None, enabled: bool = False, provider=None):
        from aka.light.runtime.paths import legacy_resource_root
        plugin_dir = PLUGIN_DIR
        if enabled and provider is not None and plugin_root is None:
            # The selected Wiki provider owns the Agent-visible catalog.
            plugin_root = Path(provider.catalog_root).resolve()
            plugin_dir = plugin_root.parent
        super().__init__(plugin_dir, plugin_root=plugin_root,
                         resource_root=legacy_resource_root(), enabled=enabled)

    def wiki_prompt(self, slot: str) -> str | None:
        """Provider-supplied replacement for a built-in Wiki prompt fragment, if declared."""
        from aka.light.runtime.plugins.registry import local_file
        for plugin in self.plugins:
            relative = plugin.manifest.get("instructions", {}).get("wiki_" + slot)
            if relative:
                return read_text(local_file(plugin.root, relative)).strip()
        return None

    def knowledge_skills(self) -> list[str]:
        names = ["KernelWiki"] if any(plugin.id == "gpu-wiki" for plugin in self.plugins) else []
        return names + [row["name"] for row in self.skill_catalog() if row["name"] not in names]

    def resource_snapshot(self) -> dict:
        """Freeze Light resources for the shared startup's launch context.

        Plugin fingerprints include their tools, Skills and declared resources.
        Built-in Light assets are already pinned by the launch selection's
        ``aka.light`` package identity. Keep an explicit owner/schema: this is
        not a Legacy resource lock or a portable Legacy checkpoint.
        """
        return {"schema": "aka.light.resources.v1", "plugins": self.snapshot()}

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
        from aka.light.runtime.plugins.registry import local_file
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
        from aka.light.workspace.skill_manifest import SKILL_MANIFEST
        from aka.light.workspace.episode import bounded_tree_entries
        from aka.light.workspace.assets import RETIRED_SKILL_NAMES
        from aka.light.workspace.files import publish

        names = []
        for plugin in self.plugins:
            for name, skill in plugin.manifest.get("skills", {}).items():
                if name in RETIRED_SKILL_NAMES:
                    raise PluginError("mount_conflict", f"Retired workflow Skill cannot be installed: {name}")
                source = (plugin.root / skill["path"]).resolve()
                if not (source / "SKILL.md").is_file():
                    continue
                if name in SKILL_MANIFEST:
                    expected = (REPO_ROOT / "workspace/resources" / SKILL_MANIFEST[name].root).resolve()
                    if source != expected:
                        raise PluginError("mount_conflict", f"Plugin cannot replace Runtime Skill {name}")
                    continue
                if name in names:
                    raise PluginError("mount_conflict", f"Duplicate plugin Skill {name}")
                names.append(name)
                for relative, content in bounded_tree_entries(source):
                    publish(workspace, f"skills/{name}/{relative}", content)
        return names
