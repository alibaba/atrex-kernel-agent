"""AKA bindings for the reusable tools/Skills runtime."""

from __future__ import annotations

from pathlib import Path

from plugin_runtime import HostLayout, Plugin, PluginError
from plugin_runtime import PluginRegistry as Registry
from plugin_runtime.schema import decode_json, read_json

from .constants import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "plugins/default.json"
STATE_DIR = ".atrex_plugins"
AKA_LAYOUT = HostLayout(
    state_dir=STATE_DIR,
    skill_roots=(".claude/skills", ".qoder/skills", ".agents/skills"),
    reserved_mounts=frozenset(
        {
            "tools",
            "reference",
            "skills",
            "reference-projects",
            "atrex-bench",
            "memory",
            "plans",
            "profiles",
        }
    ),
)


class PluginRegistry(Registry):
    def __init__(self, config: Path | str = DEFAULT_CONFIG):
        super().__init__(config, layout=AKA_LAYOUT)

    def instructions(self, phase: str, **values: str) -> str:
        return (
            "Use `python3 tools/plugin.py list` to discover enabled tools and Skills. "
            "Invoke a tool with `python3 tools/plugin.py call <plugin.tool> --input <request.json>`.\n\n"
            + super().instructions(phase, **values)
        )

    def environment(self, workspace: Path, campaign_name: str = "") -> dict[str, str]:
        environment = super().environment(workspace, {"campaign_name": campaign_name})
        reserved = {
            "ATREX_PLUGIN_CONFIG": str(self.config),
            "ATREX_PLUGIN_WORKSPACE": str(workspace.resolve()),
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
    "DEFAULT_CONFIG",
    "STATE_DIR",
    "PluginError",
    "PluginRegistry",
    "decode_json",
    "read_json",
]
