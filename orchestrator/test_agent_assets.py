"""Agent resource selection does not expose the Supervisor implementation."""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
import tarfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from orchestrator.agent_assets import (
    HTTP_CLIENT_FILES,
    REQUIRED_AGENT_SKILLS,
    SKILL_PATHS,
    initialize_writable_tools,
    materialize_agent_assets,
    resolve_agent_skills,
)
from orchestrator.constants import REPO_ROOT
from orchestrator.workspace_runtime import _agent_runtime_directive, link_runtime
from supervisor.gateway import _make_input_bundle


class AgentAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def test_default_view_contains_single_client_and_default_skills(self) -> None:
        link_runtime(self.workspace)
        self.assertEqual(
            {p.name for p in (self.workspace / "tools").iterdir()}, {"sandbox.py"}
        )
        self.assertEqual(
            {p.name for p in (self.workspace / "skills").iterdir()},
            {*REQUIRED_AGENT_SKILLS, "autonomous-gpu-kernel-timeline"},
        )
        for backend in (".claude", ".qoder", ".agents"):
            self.assertEqual(
                (self.workspace / backend / "skills").resolve(),
                (self.workspace / "skills").resolve(),
            )
        for name in ("reference", "reference-projects", "atrex-bench", "orchestrator", "gpu-wiki"):
            self.assertFalse((self.workspace / name).exists(), name)
        for name in (*REQUIRED_AGENT_SKILLS, "autonomous-gpu-kernel-timeline"):
            text = (self.workspace / "skills" / name / "SKILL.md").read_text()
            self.assertIn("Runtime execution contract", text)
            self.assertIn(name, _agent_runtime_directive("claude"))
        self.assertTrue(
            (self.workspace / "skills/autonomous-gpu-kernel-timeline/scripts/timeline.py").is_file()
        )
        self.assertTrue((self.workspace / "skills/gpu-measurement/references/requests.md").is_file())

    def test_gpu_and_record_skill_commands_parse_and_forward(self) -> None:
        import re
        import shlex

        from supervisor.gateway import build_parser
        from tools import sandbox

        link_runtime(self.workspace)
        operations = set()
        parser = build_parser()
        skill_paths = list((self.workspace / "skills/gpu-measurement").rglob("*.md"))
        for path in skill_paths:
            self.assertNotIn("record-read", path.read_text())
        record_reference = self.workspace / "skills/runtime-records/references/records.md"
        for path in [*skill_paths, record_reference, REPO_ROOT / "reference/CLAUDE.md"]:
            for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.DOTALL):
                for command in block.splitlines():
                    if not command.strip() or command.lstrip().startswith("#"):
                        continue
                    arguments = shlex.split(command)
                    self.assertEqual(arguments[:2], ["python3", "tools/sandbox.py"])
                    options = arguments[2:]
                    with self.subTest(command=command):
                        parsed = parser.parse_args(options)
                        operations.add(parsed.kind)
                        with patch.object(sandbox, "proxy_gateway", return_value=0) as proxy:
                            self.assertEqual(sandbox.main(options), 0)
                        proxy.assert_called_once_with(options)
        self.assertEqual(operations, {"run", "profile", "check", "disassemble", "dev", "env", "record-read"})

    def test_required_record_skill_has_complete_materialized_references(self) -> None:
        import re

        link_runtime(self.workspace, agent_skills=())
        skill = self.workspace / "skills/runtime-records"
        for path in skill.rglob("*.md"):
            for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text()):
                resolved = (path.parent / target).resolve()
                self.assertTrue(resolved.is_relative_to(skill.resolve()), target)
                self.assertTrue(resolved.is_file(), target)
        for backend in (".claude", ".qoder", ".agents"):
            for name in ("journal.md", "records.md"):
                self.assertEqual(
                    (self.workspace / backend / "skills/runtime-records/references" / name).read_bytes(),
                    (skill / "references" / name).read_bytes(),
                )

    def test_selection_replaces_optional_defaults_but_always_mounts_required_skills(self) -> None:
        link_runtime(
            self.workspace,
            agent_skills=("ppu-acu-joint-profile", "KernelWiki", "gpu-measurement"),
            agent_reference_projects=True,
        )
        self.assertEqual(
            {p.name for p in (self.workspace / "skills").iterdir()},
            {*REQUIRED_AGENT_SKILLS, "ppu-acu-joint-profile"},
        )
        self.assertEqual(
            (self.workspace / "reference-projects").resolve(), REPO_ROOT / "reference-projects"
        )
        link_runtime(self.workspace, agent_skills=())
        self.assertEqual(
            {p.name for p in (self.workspace / "skills").iterdir()}, set(REQUIRED_AGENT_SKILLS)
        )
        for backend in (".claude", ".qoder", ".agents"):
            for name in REQUIRED_AGENT_SKILLS:
                self.assertTrue((self.workspace / backend / "skills" / name / "SKILL.md").is_file())
        self.assertFalse((self.workspace / "reference-projects").exists())
        self.assertEqual(resolve_agent_skills((*REQUIRED_AGENT_SKILLS, *REQUIRED_AGENT_SKILLS)), REQUIRED_AGENT_SKILLS)

    def test_managed_legacy_links_are_replaced(self) -> None:
        for name in ("tools", "skills", "reference", "reference-projects", "atrex-bench", "gpu-wiki"):
            (self.workspace / name).symlink_to(REPO_ROOT / name)
        discovery = self.workspace / ".claude/skills"
        discovery.mkdir(parents=True)
        (discovery / "gen-plan").symlink_to(REPO_ROOT / "skills/gen-plan")
        link_runtime(self.workspace)
        self.assertFalse((discovery / "gen-plan").exists())
        self.assertFalse((self.workspace / "reference").exists())
        self.assertFalse((self.workspace / "atrex-bench").exists())

    def test_unmanaged_files_are_not_deleted(self) -> None:
        local = self.workspace / "tools"
        local.mkdir()
        (local / "custom.py").write_text("user data")
        with self.assertRaisesRegex(RuntimeError, "unmanaged data"):
            link_runtime(self.workspace)
        self.assertEqual((local / "custom.py").read_text(), "user data")

    def test_unknown_and_missing_skills_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Agent skills"):
            link_runtime(self.workspace, agent_skills=("missing",))
        with patch.dict("orchestrator.agent_assets.SKILL_PATHS", {"missing": "not-installed"}):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                link_runtime(self.workspace, agent_skills=("missing",))

    def test_views_are_content_keyed_and_leave_existing_sessions_unchanged(self) -> None:
        repository = self.root / "source"
        (repository / "tools").mkdir(parents=True)
        for name in HTTP_CLIENT_FILES:
            (repository / "tools" / name).write_text("first")
        for name in REQUIRED_AGENT_SKILLS:
            shutil.copytree(REPO_ROOT / SKILL_PATHS[name], repository / SKILL_PATHS[name])
        first = materialize_agent_assets(self.workspace, repository, ())
        self.assertEqual(first, materialize_agent_assets(self.workspace, repository, ()))
        (repository / "tools/sandbox.py").write_text("second")
        second = materialize_agent_assets(self.workspace, repository, ())
        self.assertNotEqual(first, second)
        self.assertEqual((first / "tools/sandbox.py").read_text(), "first")

    def test_writable_tools_are_isolated_and_preserve_edits_and_deletions_on_resume(self) -> None:
        assets = materialize_agent_assets(self.workspace, REPO_ROOT, ())
        original = (assets / "tools/sandbox.py").read_bytes()
        first = initialize_writable_tools(self.workspace, assets)
        self.assertFalse(first.is_symlink())
        (first / "sandbox.py").write_text("# local edit\n")
        (first / "helper.py").write_text("print('helper')\n")
        self.assertEqual(initialize_writable_tools(self.workspace, assets), first)
        self.assertEqual((first / "sandbox.py").read_text(), "# local edit\n")
        (first / "sandbox.py").unlink()
        initialize_writable_tools(self.workspace, assets)
        self.assertFalse((first / "sandbox.py").exists())
        self.assertTrue((first / "helper.py").exists())
        second_workspace = self.root / "episode-2"
        second_workspace.mkdir()
        second = initialize_writable_tools(second_workspace, assets)
        self.assertEqual((second / "sandbox.py").read_bytes(), original)
        self.assertFalse((second / "helper.py").exists())
        self.assertEqual((assets / "tools/sandbox.py").read_bytes(), original)

    def test_old_readonly_tools_link_migrates_without_touching_the_seed(self) -> None:
        link_runtime(self.workspace)
        assets = materialize_agent_assets(self.workspace, REPO_ROOT, ())
        previous = (self.workspace / "tools").resolve()
        original = (previous / "sandbox.py").read_bytes()
        writable = initialize_writable_tools(self.workspace, assets)
        self.assertFalse(writable.is_symlink())
        (writable / "sandbox.py").write_text("# edited\n")
        self.assertEqual((previous / "sandbox.py").read_bytes(), original)

    def test_unmanaged_tool_links_are_not_followed_or_removed(self) -> None:
        assets = materialize_agent_assets(self.workspace, REPO_ROOT, ())
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.py").write_text("user data")
        (self.workspace / "tools").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "not a managed asset"):
            initialize_writable_tools(self.workspace, assets)
        self.assertTrue((self.workspace / "tools").is_symlink())
        self.assertEqual((outside / "keep.py").read_text(), "user data")

    def test_dev_bundles_writable_tools_but_profile_helpers_stay_trusted(self) -> None:
        assets = materialize_agent_assets(self.workspace, REPO_ROOT, ())
        writable = initialize_writable_tools(self.workspace, assets)
        (writable / "probe.py").write_text("print('custom probe')\n")
        (writable / "profile_nvidia.sh").write_text("echo forged helper\n")
        selected = ("tools/probe.py", "tools/profile_nvidia.sh", "tools/sandbox.py")
        encoded, count, skipped = _make_input_bundle(self.workspace, 2**20, selected)
        self.assertEqual(count, 2)
        self.assertEqual(skipped, [])
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded))) as archive:
            self.assertEqual(archive.extractfile(selected[0]).read(), (writable / "probe.py").read_bytes())
            self.assertEqual(archive.extractfile(selected[1]).read(), (REPO_ROOT / selected[1]).read_bytes())
            self.assertNotIn("tools/sandbox.py", archive.getnames())

    def test_private_profile_helpers_and_selected_skill_inputs_still_bundle(self) -> None:
        link_runtime(self.workspace, agent_skills=("autonomous-gpu-kernel-timeline",))
        selected = (
            "tools/profile_nvidia.sh",
            "skills/autonomous-gpu-kernel-timeline/scripts/timeline.py",
        )
        encoded, count, skipped = _make_input_bundle(self.workspace, 2**20, selected)
        self.assertEqual(count, 2)
        self.assertEqual(skipped, [])
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded))) as archive:
            self.assertEqual(set(archive.getnames()), set(selected))
            self.assertEqual(
                archive.extractfile(selected[1]).read(), (self.workspace / selected[1]).read_bytes()
            )
        self.assertFalse((self.workspace / "tools/profile_nvidia.sh").exists())

    def test_cli_exposes_resource_selection(self) -> None:
        from orchestrator.optimize import _run_main

        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            _run_main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        for option in ("--agent-skill", "--no-agent-skills", "--agent-reference-projects"):
            self.assertIn(option, output.getvalue())

    def test_prompt_only_advertises_selected_assets(self) -> None:
        for backend in ("claude", "qodercli", "codex", "pi"):
            directive = _agent_runtime_directive(backend, agent_skills=())
            self.assertIn("`tools/` is writable", directive)
            for name in REQUIRED_AGENT_SKILLS:
                self.assertIn(name, directive)
            self.assertNotIn("autonomous-gpu-kernel-timeline", directive)
            self.assertNotIn("ncu-report-skill", directive)
            self.assertNotIn("gen-plan", directive)
            self.assertIn("No reference-project", directive)
        for name in ("episode.md", "framework_baseline.md"):
            text = (REPO_ROOT / "orchestrator/prompts" / name).read_text()
            for removed in (
                "iteration_trace.py",
                "memory_manager.py",
                "reference/v_iteration",
                "skills/gpu-kernel-episode-loop",
            ):
                self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
