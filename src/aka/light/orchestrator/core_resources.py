"""Light's bounded projection of a Core-bound AgentResources selection.

The Core binding remains the authority for tool identity, plans, receipts and
source references.  This facade only presents the small Agent-facing surface
that Light needs; it never serializes the bound plan or its host paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aka.light.supervisor.workspace import publish


class CoreResourceProjection:
    """Expose Core contributions through Light's existing tools/plugin.py API."""

    def __init__(self, bound):
        self.bound = bound
        # Keep Core receipts and source refs discoverable to generic artifact
        # collection while leaving all Core resource mounts out of Light's
        # provider-visible workspace.
        self.bound.record_binding()

    def catalog(self) -> list[dict[str, Any]]:
        # Bound catalog validation also detects drift in frozen implementations.
        return [dict(row) for row in self.bound.catalog()]

    def skill_catalog(self) -> list[dict[str, Any]]:
        # Absolute source paths are supervisor-only. The Agent receives only
        # the relative mount path, while installation reads the trusted row.
        rows = self.bound.skill_catalog()
        return [
            {key: value for key, value in row.items() if key != "path"}
            | {"path": f"skills/{row['install_name']}"}
            for row in rows
        ]

    def public_catalog(self) -> dict[str, Any]:
        return {"tools": self.catalog(), "skills": self.skill_catalog()}

    def snapshot(self) -> dict[str, Any]:
        """Return the Agent-visible lock projection without host paths."""
        snapshot = self.resource_snapshot()
        return {
            "schema": "aka.light.core-resources.v1",
            "digest": hashlib.sha256(
                json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
            "scope_identity": snapshot.get("scope_identity", ""),
            "tools": [
                {"name": row.get("name"), "identity": row.get("identity")}
                for row in snapshot.get("tools", [])
            ],
            "skills": [
                {
                    "name": row.get("name"),
                    "install_name": row.get("install_name"),
                    "identity": row.get("identity", {}).get("digest", ""),
                }
                for row in snapshot.get("skills", [])
            ],
        }

    def validate_call(self, name: str, value: object) -> dict[str, Any]:
        from aka.core.plugin_runtime.schema import PluginError, validate_schema

        row = next((item for item in self.catalog() if item.get("name") == name), None)
        if row is None:
            raise PluginError(
                "tool_not_found", "Tool is not enabled; use tools/plugin.py list"
            )
        validate_schema(dict(row["input_schema"]), value)
        return row

    @property
    def plugins(self):
        # EpisodeWorkspace's legacy-only bookkeeping iterates this collection.
        # Core plugins are intentionally not projected into that registry.
        return ()

    def environment(self, workspace=None):
        return {}

    def instructions(self, phase: str, **values: str) -> str:
        rendered = self.bound.instructions(phase, **values)
        wiki_tools = [
            row.get("name") for row in self.catalog()
            if isinstance(row.get("name"), str)
            and row["name"] in {"gpu-wiki.query", "internal-gpu-wiki.query"}
        ]
        if not wiki_tools:
            return rendered
        bridge = (
            "Light exposes selected Core Wiki tools through its bounded plugin client. "
            "Discover the enabled tool names with `python3 tools/plugin.py list`; call each "
            "selected Wiki separately with `python3 tools/plugin.py call <tool> --input "
            "scratch/wiki_request.json`, where the JSON file matches that tool's declared input schema. "
            "The command prints the full `aka.tool-result.v1` envelope. Use `data` for the query "
            "response and retain `query_id`, canonical Wiki IDs, and `references` for attribution. "
            "Do not access Wiki stores or legacy resource CLIs directly."
        )
        bridge_lines = [bridge]
        bridge_lines.extend(
            f"- `{name}`: `python3 tools/plugin.py call {name} --input scratch/wiki_request.json`"
            for name in wiki_tools
        )
        return "\n".join([*bridge_lines, rendered] if rendered else bridge_lines)

    def managed_paths(self) -> list[str]:
        """Return only relative paths Light must exclude from candidate source."""
        paths = {".aka", ".atrex_plugins"}
        for destination in self.bound.mounts():
            paths.add(destination)
        for row in self.bound.skill_catalog():
            paths.add(f"skills/{row['install_name']}")
        for key, value in self.bound.environment(
            workspace=self.bound.workspace, campaign_id=self.bound.campaign_id
        ).items():
            if not any(marker in key.upper() for marker in ("ROOT", "DIR", "PATH", "PROFILE", "STORE")):
                continue
            candidate = Path(value)
            try:
                relative = candidate.resolve().relative_to(self.bound.workspace)
            except (OSError, ValueError):
                continue
            if relative.parts:
                paths.add(relative.as_posix())
        safe = []
        for value in paths:
            path = Path(value)
            if (not path.is_absolute() and path.parts and ".." not in path.parts
                    and path.as_posix() == value):
                safe.append(value)
        return sorted(safe)

    def resource_snapshot(self) -> dict[str, Any]:
        """Return the exact trusted frozen snapshot for Core validation/recovery.

        Callers must keep this controller-side. Use ``snapshot()`` for files in
        the Agent workspace.
        """
        return self.bound.resource_snapshot()

    def check_lock(self, workspace=None) -> None:
        # The Core host checks its own canonical bound target and durable
        # selection. Light's Episode workspace is only the controlled cwd.
        self.bound.check_lock()

    def install_agent_skills(self, workspace: Path) -> list[str]:
        from aka.light.orchestrator.agent_skill_manifest import SKILL_MANIFEST
        from aka.light.orchestrator.episode_workspace import bounded_tree_entries

        installed: list[str] = []
        for row in self.bound.skill_catalog():
            name = row.get("install_name")
            source_value = row.get("path")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("Core Agent Skill has an unsafe install name")
            # The Core Wiki Skill intentionally replaces the old packaged
            # Light KernelWiki Skill at the same Agent path during migration.
            if name in SKILL_MANIFEST and name != "KernelWiki":
                raise ValueError(
                    f"Core Agent Skill conflicts with Light Runtime Skill: {name}"
                )
            if not isinstance(source_value, str) or not source_value:
                raise ValueError(f"Core Agent Skill source is unavailable: {name}")
            source = Path(source_value)
            if source.is_symlink() or not source.is_dir():
                raise ValueError(f"Core Agent Skill source is unavailable: {name}")
            # This controller-selected source may have a stable installation
            # prefix alias (for example /tmp -> /private/tmp on macOS). Resolve
            # that prefix before descriptor-based, no-follow reads of the tree.
            # Agent workspace paths and Skill descendants remain no-follow.
            source = source.resolve(strict=True)
            for relative, content in bounded_tree_entries(source):
                publish(workspace, f"skills/{name}/{relative}", content)
            installed.append(name)
        return installed

    def call_result(
        self, name: str, request: Any, workspace: Path, *, cwd: Path, executor
    ):
        """Use Core's receipt/attribution pipeline with Light-owned execution."""
        return self.bound.call_result(
            name,
            request,
            self.bound.workspace,
            cwd=cwd,
            executor=executor,
        )
