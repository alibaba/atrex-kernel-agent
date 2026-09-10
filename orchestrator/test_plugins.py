from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.constants import REPO_ROOT
from orchestrator.plugins import STATE_DIR, PluginError, PluginRegistry
from orchestrator.workspace_runtime import link_runtime


class PluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.config = self.root / "plugins.json"
        self.plugin = self.make_plugin("local-docs")
        self.configure([{"path": "local-docs"}])

    def write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value))

    def configure(self, entries: list) -> PluginRegistry:
        self.write_json(self.config, {"schema_version": 1, "plugins": entries})
        return PluginRegistry(self.config)

    def make_plugin(self, name: str) -> Path:
        root = self.root / name
        root.mkdir()
        self.write_json(
            root / "input.json",
            {
                "type": "object",
                "required": ["request"],
                "additionalProperties": False,
                "properties": {"request": {"type": "string", "minLength": 1}},
            },
        )
        self.write_json(
            root / "output.json",
            {
                "type": "object",
                "required": ["answer", "cwd"],
                "properties": {"answer": {"type": "string"}, "cwd": {"type": "string"}},
            },
        )
        (root / "adapter.py").write_text(
            "import json, os, sys\n"
            "request = json.load(sys.stdin)\n"
            "print(json.dumps({'answer': request['request'], 'cwd': os.getcwd()}))\n"
        )
        (root / "instructions.md").write_text(f"Use {name}.query for local references.")
        (root / "data").mkdir()
        (root / "data/source.txt").write_text("local fact")
        (root / "skill").mkdir()
        (root / "skill/SKILL.md").write_text(
            "---\nname: local-docs\n---\nRead local sources."
        )
        self.write_json(
            root / "plugin.json",
            {
                "id": name,
                "version": "1.0.0",
                "api_version": 1,
                "instructions": {"common": "instructions.md"},
                "resources": {name + "-data": "data"},
                "skills": {name: {"path": "skill", "optional": False}},
                "tools": {
                    "query": {
                        "description": "Read local reference facts.",
                        "entrypoint": "adapter.py",
                        "input_schema": "input.json",
                        "output_schema": "output.json",
                        "timeout_seconds": 1,
                    }
                },
            },
        )
        return root

    def assert_error(self, code: str, function, *args, **kwargs) -> None:
        with self.assertRaises(PluginError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def test_second_plugin_installs_and_runs_without_core_changes(self) -> None:
        registry = self.configure(
            [
                {"path": str(REPO_ROOT / "plugins/gpu-wiki")},
                {"path": "local-docs"},
            ]
        )
        link_runtime(self.workspace, plugin_registry=registry)
        self.assertEqual(
            {item["name"] for item in registry.catalog()},
            {"gpu-wiki.query", "local-docs.query"},
        )
        self.assertEqual(
            (self.workspace / "local-docs-data/source.txt").read_text(), "local fact"
        )
        for backend in (".claude", ".qoder", ".agents"):
            self.assertTrue(
                (self.workspace / backend / "skills/local-docs/SKILL.md").is_file()
            )
        self.assertIn(
            "local-docs.query",
            (self.workspace / STATE_DIR / "instructions.md").read_text(),
        )
        value = registry.call(
            "local-docs.query", {"request": "reference"}, self.workspace
        )
        self.assertEqual(value["answer"], "reference")
        self.assertEqual(value["cwd"], str(self.workspace.resolve()))
        registry.install(self.workspace)  # Idempotent on resume.

    def test_skill_only_package_supports_an_independent_host(self) -> None:
        from plugin_runtime import HostLayout
        from plugin_runtime import PluginRegistry as Registry

        manifest = json.loads((self.plugin / "plugin.json").read_text())
        del manifest["tools"]
        manifest["skills"]["local-docs"] = {
            "path": "skill",
            "description": "Review a document.",
        }
        manifest["instructions"]["document-review"] = "instructions.md"
        self.write_json(self.plugin / "plugin.json", manifest)
        layout = HostLayout(state_dir=".extensions", skill_roots=("native/skills",))
        registry = Registry(self.config, layout=layout)
        registry.install(self.workspace)
        self.assertEqual(registry.catalog(), [])
        self.assertEqual(registry.skill_catalog()[0]["id"], "local-docs.local-docs")
        self.assertTrue(
            (self.workspace / "native/skills/local-docs/SKILL.md").is_file()
        )
        self.assertFalse((self.workspace / ".agents").exists())
        self.assertIn("Review a document", registry.instructions("document-review"))
        self.assertNotIn("wiki_usage", registry.instructions("document-review"))
        self.assert_error(
            "plugin_changed",
            Registry(
                self.config,
                layout=HostLayout(
                    state_dir=".extensions", skill_roots=("other/skills",)
                ),
            ).check_lock,
            self.workspace,
        )

    def test_non_python_command_uses_json_without_shell_interpolation(self) -> None:
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        tool = manifest["tools"]["query"]
        del tool["entrypoint"]
        tool["command"] = ["/bin/sh", "{plugin_root}/query.sh"]
        (self.plugin / "query.sh").write_text(
            'cat >/dev/null\nprintf \'{"answer":"shell","cwd":"%s"}\\n\' "$PWD"\n'
        )
        self.write_json(self.plugin / "plugin.json", manifest)
        registry = PluginRegistry(self.config)
        result = registry.call(
            "local-docs.query", {"request": "$(touch unexpected)"}, self.workspace
        )
        self.assertEqual(result["answer"], "shell")
        self.assertFalse((self.workspace / "unexpected").exists())

    def test_tool_settings_are_validated_delivered_and_locked(self) -> None:
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        manifest["settings_schema"] = "settings.json"
        self.write_json(
            self.plugin / "settings.json",
            {
                "type": "object",
                "required": ["prefix"],
                "properties": {"prefix": {"type": "string"}},
            },
        )
        self.write_json(self.plugin / "plugin.json", manifest)
        self.assert_error("invalid_config", PluginRegistry, self.config)
        (self.plugin / "adapter.py").write_text(
            'import os,json\nprint(json.dumps({"answer":json.loads(os.environ["PLUGIN_SETTINGS_JSON"])["prefix"],"cwd":os.getcwd()}))\n'
        )
        registry = self.configure(
            [{"path": "local-docs", "settings": {"prefix": "private-value"}}]
        )
        registry.install(self.workspace)
        result = registry.call("local-docs.query", {"request": "test"}, self.workspace)
        self.assertEqual(result["answer"], "private-value")
        self.assertNotIn(
            "private-value", (self.workspace / STATE_DIR / "lock.json").read_text()
        )
        changed = self.configure(
            [{"path": "local-docs", "settings": {"prefix": "changed"}}]
        )
        self.assert_error("plugin_changed", changed.check_lock, self.workspace)

    def test_skill_collisions_and_empty_packages_fail(self) -> None:
        self.make_plugin("other")
        manifest = json.loads((self.root / "other/plugin.json").read_text())
        manifest["skills"] = {"local-docs": {"path": "skill"}}
        self.write_json(self.root / "other/plugin.json", manifest)
        self.assert_error(
            "invalid_manifest",
            self.configure,
            [{"path": "local-docs"}, {"path": "other"}],
        )
        manifest = {"id": "other", "version": "1", "api_version": 1}
        self.write_json(self.root / "other/plugin.json", manifest)
        self.assert_error("invalid_manifest", self.configure, [{"path": "other"}])

    def test_missing_executable_fails_during_loading(self) -> None:
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        tool = manifest["tools"]["query"]
        del tool["entrypoint"]
        tool["command"] = ["/missing/plugin-interpreter"]
        self.write_json(self.plugin / "plugin.json", manifest)
        self.assert_error("invalid_manifest", PluginRegistry, self.config)

    def test_no_plugins_removes_legacy_links_and_omits_wiki_instructions(self) -> None:
        registry = self.configure([{"path": "missing-plugin", "enabled": False}])
        (self.workspace / "gpu-wiki").symlink_to(REPO_ROOT / "gpu-wiki")
        for backend in (".claude", ".qoder", ".agents"):
            path = self.workspace / backend / "skills/KernelWiki"
            path.parent.mkdir(parents=True)
            path.symlink_to(REPO_ROOT / "gpu-wiki/3rdparty/KernelWiki")
        link_runtime(self.workspace, plugin_registry=registry)
        self.assertFalse((self.workspace / "gpu-wiki").is_symlink())
        for backend in (".claude", ".qoder", ".agents"):
            self.assertFalse(
                (self.workspace / backend / "skills/KernelWiki").is_symlink()
            )
        for phase in (
            "common",
            "setup",
            "episode",
            "fast_episode",
            "framework_baseline",
        ):
            text = registry.instructions(phase)
            self.assertNotIn("gpu-wiki.query", text)
            self.assertNotIn("required bounded GPU Wiki query", text)
            self.assertNotIn("wiki_usage_status", text)
        self.assertEqual(registry.catalog(), [])
        self.assertNotIn(
            "ATREX_WIKI_PROFILE_ROOT", registry.environment(self.workspace, "task")
        )
        self.assert_error(
            "tool_not_found", registry.call, "gpu-wiki.query", {}, self.workspace
        )

    def test_disabled_plugin_does_not_remove_user_owned_directory(self) -> None:
        registry = self.configure([])
        directory = self.workspace / "gpu-wiki"
        directory.mkdir()
        (directory / "user.txt").write_text("keep")
        link_runtime(self.workspace, plugin_registry=registry)
        self.assertEqual((directory / "user.txt").read_text(), "keep")

    def test_missing_enabled_plugin_fails_before_install(self) -> None:
        self.assert_error(
            "invalid_config", self.configure, [{"path": "missing-plugin"}]
        )
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_duplicate_plugin_id_and_reserved_mount_fail(self) -> None:
        self.assert_error(
            "invalid_config",
            self.configure,
            [{"path": "local-docs"}, {"path": "local-docs"}],
        )
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        manifest["resources"] = {"tools": "data"}
        self.write_json(self.plugin / "plugin.json", manifest)
        self.assert_error("invalid_manifest", self.configure, [{"path": "local-docs"}])

    def test_mount_conflict_does_not_overwrite_files(self) -> None:
        registry = PluginRegistry(self.config)
        conflict = self.workspace / "local-docs-data"
        conflict.write_text("user data")
        self.assert_error("mount_conflict", registry.install, self.workspace)
        self.assertEqual(conflict.read_text(), "user data")
        self.assertFalse((self.workspace / STATE_DIR).exists())

    def test_version_code_data_and_configuration_changes_are_detected(self) -> None:
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        for relative in (
            "adapter.py",
            "data/source.txt",
            "instructions.md",
            "skill/SKILL.md",
        ):
            path = self.plugin / relative
            original = path.read_text()
            path.write_text(original + "\n# change\n")
            self.assert_error("plugin_changed", registry.check_lock, self.workspace)
            self.assert_error(
                "plugin_changed", PluginRegistry(self.config).install, self.workspace
            )
            path.write_text(original)
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        manifest["version"] = "2.0.0"
        self.write_json(self.plugin / "plugin.json", manifest)
        self.assert_error(
            "plugin_changed",
            PluginRegistry(self.config).check_lock,
            self.workspace,
        )
        registry = self.configure([])
        self.assert_error("plugin_changed", registry.install, self.workspace)

    def test_python_cache_does_not_change_plugin_identity(self) -> None:
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        (self.plugin / "__pycache__").mkdir()
        (self.plugin / "__pycache__/adapter.pyc").write_bytes(b"cache")
        registry.check_lock(self.workspace)

    def test_optional_unmounted_resource_is_locked_without_exposing_it(self) -> None:
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        manifest["resources"]["optional-data"] = {
            "path": "../optional",
            "optional": True,
            "mount": False,
        }
        self.write_json(self.plugin / "plugin.json", manifest)
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        self.assertFalse((self.workspace / "optional-data").exists())
        (self.root / "optional").mkdir()
        (self.root / "optional/fact.txt").write_text("new data source")
        self.assert_error("plugin_changed", registry.check_lock, self.workspace)

    def test_schema_errors_are_not_executed_and_are_recorded_without_payloads(
        self,
    ) -> None:
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        for request in (
            {},
            {"request": " "},
            {"request": 3},
            {"request": "private payload", "unexpected": True},
        ):
            self.assert_error(
                "schema_validation",
                registry.call,
                "local-docs.query",
                request,
                self.workspace,
            )
        events = list((self.workspace / STATE_DIR / "calls").glob("*.json"))
        self.assertEqual(len(events), 4)
        for event in events:
            raw = event.read_text()
            self.assertNotIn("private payload", raw)
            self.assertEqual(json.loads(raw)["status"], "schema_validation")

    def test_invalid_output_and_process_failure_are_distinct(self) -> None:
        for code, source in (
            ("invalid_output", "print('not json')"),
            ("invalid_output", "print('{}')"),
            ("tool_failed", "import sys; sys.stderr.write('failed'); sys.exit(4)"),
        ):
            (self.plugin / "adapter.py").write_text(source)
            registry = PluginRegistry(self.config)
            self.assert_error(
                code,
                registry.call,
                "local-docs.query",
                {"request": "test"},
                self.workspace,
            )

    def test_timeout_terminates_process_group(self) -> None:
        (self.plugin / "adapter.py").write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "time.sleep(30)\n"
        )
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        started = time.monotonic()
        self.assert_error(
            "tool_timeout",
            registry.call,
            "local-docs.query",
            {"request": "test"},
            self.workspace,
        )
        self.assertLess(time.monotonic() - started, 5)
        event = next((self.workspace / STATE_DIR / "calls").glob("*.json"))
        self.assertEqual(json.loads(event.read_text())["status"], "tool_timeout")

    def test_unsupported_schema_and_manifest_api_fail_at_load(self) -> None:
        self.write_json(
            self.plugin / "input.json", {"type": "object", "$ref": "missing"}
        )
        self.assert_error("invalid_manifest", PluginRegistry, self.config)
        self.write_json(self.plugin / "input.json", {"type": "object"})
        manifest = json.loads((self.plugin / "plugin.json").read_text())
        manifest["api_version"] = 2
        self.write_json(self.plugin / "plugin.json", manifest)
        self.assert_error("invalid_manifest", PluginRegistry, self.config)

    def test_input_schema_checks_nested_types_and_ranges(self) -> None:
        self.write_json(
            self.plugin / "input.json",
            {
                "type": "object",
                "required": ["options"],
                "properties": {
                    "options": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                },
            },
        )
        registry = PluginRegistry(self.config)
        for options in ([True], [0], [4], ["1"]):
            self.assert_error(
                "schema_validation",
                registry.call,
                "local-docs.query",
                {"options": options},
                self.workspace,
            )

    def test_malformed_schema_types_fail_during_loading(self) -> None:
        for schema in (
            {"type": []},
            {"type": "string", "minimum": 1},
            {"type": "integer", "minimum": 4, "maximum": 1},
        ):
            self.write_json(self.plugin / "input.json", schema)
            self.assert_error("invalid_manifest", PluginRegistry, self.config)

    def test_cli_uses_campaign_lock_and_preserves_episode_cwd(self) -> None:
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        episode = self.root / "episode"
        episode.mkdir()
        environment = dict(os.environ)
        environment.pop("ATREX_PLUGIN_CONFIG", None)
        environment["ATREX_PLUGIN_WORKSPACE"] = str(self.workspace)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/plugin.py"),
                "call",
                "local-docs.query",
                "--input",
                "-",
            ],
            input='{"request":"episode fact"}',
            text=True,
            capture_output=True,
            cwd=episode,
            env=environment,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["cwd"], str(episode.resolve()))
        self.assertTrue(list((self.workspace / STATE_DIR / "calls").glob("*.json")))
        self.assertFalse((episode / STATE_DIR).exists())

    def test_failed_telemetry_does_not_replace_success(self) -> None:
        registry = PluginRegistry(self.config)
        registry.install(self.workspace)
        (self.workspace / STATE_DIR / "calls").write_text("cannot create directory")
        result = registry.call("local-docs.query", {"request": "ok"}, self.workspace)
        self.assertEqual(result["answer"], "ok")

    def test_campaign_and_episode_share_plugin_configuration(self) -> None:
        from long_horizon.main_adapter import episode_directives, link_episode_runtime
        from orchestrator.campaign import Campaign

        campaign = Campaign(
            name="fixture",
            kernel_demo=str(self.root / "reference.py"),
            platform="B200",
            arch="sm_100",
            framework="triton",
            work_dir=str(self.root),
            plugin_config=str(self.config),
        )
        campaign.workspace.mkdir()
        campaign.plugin_registry.install(campaign.workspace)
        for fast in (True, False):
            text = episode_directives(campaign, 2, fast=fast)["plugins"]
            self.assertIn("local-docs.query", text)
            self.assertNotIn("gpu-wiki.query", text)
        episode = self.root / "episode"
        episode.mkdir()
        with patch("long_horizon.main_adapter.install_workspace_policy"):
            link_episode_runtime(campaign, episode)
        self.assertEqual(
            (campaign.workspace / STATE_DIR / "lock.json").read_text(),
            (episode / STATE_DIR / "lock.json").read_text(),
        )
        self.assertNotIn("ATREX_WIKI_PROFILE_ROOT", campaign.agent_environment())

    def test_all_phase_templates_render_without_disabled_wiki_requirements(
        self,
    ) -> None:
        from long_horizon.campaign import _render as render_episode
        from orchestrator.session_io import _render as render_setup

        for entries in ([], [{"path": "local-docs"}]):
            registry = self.configure(entries)
            for phase in ("setup", "episode", "fast_episode", "framework_baseline"):
                path = REPO_ROOT / "orchestrator/prompts" / f"{phase}.md"
                values = {
                    key: "fixture"
                    for key in re.findall(r"\{\{(\w+)\}\}", path.read_text())
                }
                values["PLUGINS"] = registry.instructions(phase)
                text = (
                    render_episode(path.read_text(), values)
                    if phase in {"episode", "fast_episode"}
                    else render_setup(path, **values)
                )
                self.assertNotIn("{{", text)
                self.assertNotIn("gpu-wiki.query", text)
                self.assertNotIn("gpu-wiki/tools/query_nl.py", text)
                self.assertNotIn("required bounded GPU Wiki query", text)
                if entries:
                    self.assertIn("local-docs.query", text)

    def test_plugin_state_and_resources_are_not_uploaded_to_gpu(self) -> None:
        from tools.sandbox import _walk_files

        registry = PluginRegistry(self.config)
        link_runtime(self.workspace, plugin_registry=registry)
        (self.workspace / "kernel.py").write_text("# candidate")
        uploaded = [
            path.relative_to(self.workspace).as_posix()
            for path in _walk_files(self.workspace)
        ]
        self.assertIn("kernel.py", uploaded)
        self.assertFalse(
            any(path.startswith((STATE_DIR, "local-docs-data")) for path in uploaded)
        )


class WikiPluginIntegrationTest(unittest.TestCase):
    def test_real_query_preserves_existing_records_and_attribution(self) -> None:
        request = "Target hardware B200, DSL triton. Optimize operator rmsnorm and retrieve techniques and pitfalls."
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            registry = PluginRegistry()
            registry.install(workspace)
            # The standard request must work with no bridge CLI on PATH or network/model calls.
            environment = dict(os.environ, PATH="", ATREX_WIKI_BRIDGE_CLI="claude")
            environment.pop("ATREX_WIKI_STORE_ROOT", None)
            direct = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "gpu-wiki/tools/query_nl.py"),
                    request,
                ],
                capture_output=True,
                text=True,
                env=environment,
                timeout=30,
                check=False,
            )
            self.assertEqual(direct.returncode, 0, direct.stderr)
            expected = json.loads(direct.stdout)
            with patch.dict(os.environ, environment, clear=True):
                actual = registry.call(
                    "gpu-wiki.query", {"request": request}, workspace
                )
            self.assertEqual(actual["records"], expected["records"])
            self.assertEqual(actual["notes"], expected["notes"])
            self.assertTrue(actual["records"])
            self.assertRegex(actual["query_id"], r"^wiki-query-[0-9a-f]{32}$")
            self.assertNotEqual(actual["query_id"], expected["query_id"])
            for record in actual["records"].values():
                self.assertTrue(record["wiki_id"].startswith(record["store"] + "::"))
            event = json.loads(
                next((workspace / STATE_DIR / "calls").glob("*.json")).read_text()
            )
            self.assertEqual(event["status"], "ok")
            self.assertNotIn("records", event)

    def test_ambient_store_override_cannot_bypass_declared_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = PluginRegistry()
            with (
                patch.dict(os.environ, {"ATREX_WIKI_STORE_ROOT": temporary}),
                self.assertRaises(PluginError) as raised,
            ):
                registry.call("gpu-wiki.query", {"request": "test"}, Path(temporary))
            self.assertEqual(raised.exception.code, "tool_failed")
            self.assertIn(
                "differs from the locked plugin resource", str(raised.exception)
            )


if __name__ == "__main__":
    unittest.main()
