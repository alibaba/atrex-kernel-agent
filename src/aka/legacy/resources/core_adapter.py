"""Raw-data compatibility views over a frozen Core resource selection."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from aka.core.plugin_runtime.schema import PluginError


def selected_core_resources(workspace=None):
    """Read an explicit frozen plan; never scan a plugin directory or boot Core."""
    from aka.bootstrap.continuation import base_payload, read, restore

    payload = read(os.environ)
    if payload is not None:
        value = json.loads(payload)
        if value.get("version") == 3:
            from aka.resources import restore_bound

            restore(payload)
            bound = restore_bound(
                value["agent_resources"],
                base_selection=json.loads(base_payload(payload)),
            )
            if workspace is not None and Path(workspace).resolve() != bound.workspace:
                raise PluginError(
                    "plugin_changed", "workspace differs from recorded Core resources"
                )
            return bound
    plan, expected = (
        os.environ.get("AKA_AGENT_TOOL_PLAN"),
        os.environ.get("AKA_AGENT_TOOL_PLAN_DIGEST"),
    )
    if plan is None and expected is None:
        return None
    if not plan or not expected:
        raise PluginError(
            "plugin_changed", "tool plan path and digest must travel together"
        )
    from aka.resources import read_plan, restore_bound

    value = read_plan(plan, expected)
    binding = {
        "schema": "aka.bound-agent-resources.v1",
        "workspace": value["workspace"],
        "campaign_id": value["campaign_id"],
        "plan": str(Path(plan).resolve()),
        "digest": expected,
    }
    bound = restore_bound(binding)
    if workspace is not None and Path(workspace).resolve() != bound.workspace:
        raise PluginError(
            "plugin_changed", "workspace differs from recorded Core resources"
        )
    return bound


class CoreResourcesFacade:
    """Preserve the old Python/CLI shape without creating legacy locks."""

    plugin_dir = None
    plugin_paths = ()
    plugins = ()

    def __init__(self, bound):
        self.bound = bound

    @staticmethod
    def _error(exc):
        code = getattr(exc, "code", "invalid_config")
        code = {
            "selection_changed": "plugin_changed",
            "resource_changed": "plugin_changed",
            "tool_output_limit": "invalid_output",
        }.get(code, code)
        return PluginError(code, str(exc))

    def catalog(self):
        return [
            {
                "name": row["name"],
                "version": row.get("version", "1.0.0"),
                "description": row["description"],
                "input_schema": row["input_schema"],
                "output_schema": row["output_schema"],
            }
            for row in self.bound.catalog()
        ]

    def skill_catalog(self):
        result = []
        for row in self.bound.skill_catalog():
            name = row.get("install_name", row["name"])
            plugin = {
                "KernelWiki": "gpu-wiki",
                "InternalKernelWiki": "internal-gpu-wiki",
            }.get(name)
            result.append(
                {
                    "id": f"{plugin}.{name}" if plugin else row["id"],
                    "plugin": plugin or row.get("plugin", "selected-modules"),
                    "name": name,
                    "version": "1.0.0" if plugin else row.get("version", "1"),
                    "path": row["path"],
                    "description": row.get("description", ""),
                }
            )
        return result

    def resource_result(self, workspace=None):
        value = self.bound.resource_result(workspace)
        return replace(
            value, _tools=tuple(self.catalog()), _skills=tuple(self.skill_catalog())
        )

    def call(self, name, request, workspace, *, cwd=None):
        try:
            if Path(workspace).resolve() != self.bound.workspace:
                raise PluginError(
                    "plugin_changed",
                    "tool call workspace differs from frozen selection",
                )
            result = self.bound.call_result(name, request, cwd=cwd)
            if result["receipt"]["status"] != "ok":
                details = result["diagnostics"][0]
                raise PluginError(
                    {
                        "selection_changed": "plugin_changed",
                        "resource_changed": "plugin_changed",
                        "tool_output_limit": "invalid_output",
                    }.get(details["code"], details["code"]),
                    details["message"],
                )
            for diagnostic in result.get("diagnostics", ()):
                print(f"{diagnostic['code']}: {diagnostic['message']}", file=sys.stderr)
            return result["data"]
        except ValueError as exc:
            raise self._error(exc) from exc

    def instructions(self, phase, **values):
        result = self.bound.instructions(phase, **values)
        return (
            result
            if isinstance(result, str)
            else "\n\n".join(item.text for item in result)
        )

    def snapshot(self):
        return self.bound.resource_snapshot()

    def check_lock(self, workspace=None):
        try:
            return self.bound.check_lock(workspace)
        except ValueError as exc:
            raise self._error(exc) from exc

    def environment(self, workspace=None, campaign_name=""):
        try:
            return self.bound.environment(workspace, campaign_name or None)
        except ValueError as exc:
            raise self._error(exc) from exc

    def install(self, workspace=None):
        try:
            return self.bound.install(workspace)
        except ValueError as exc:
            raise self._error(exc) from exc

    def __getattr__(self, name):
        return getattr(self.bound, name)
