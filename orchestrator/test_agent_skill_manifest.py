"""Verify the actual Agent view, including optional Skill dependency closures."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchestrator.agent_assets import HTTP_CLIENT_FILES, materialize_agent_assets
from orchestrator.agent_skill_manifest import SKILL_MANIFEST
from orchestrator.constants import REPO_ROOT


class AgentSkillManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def materialize(self, repository: Path = REPO_ROOT) -> Path:
        return materialize_agent_assets(self.workspace, repository, tuple(SKILL_MANIFEST))

    def copy_sources(self) -> Path:
        repository = self.root / "repository"
        sources = {f"tools/{name}" for name in HTTP_CLIENT_FILES}
        for manifest in SKILL_MANIFEST.values():
            sources.update(manifest.imports.get(name, f"{manifest.root}/{name}") for name in manifest.files)
        for name in sources:
            target = repository / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / name, target)
        return repository

    def test_every_selected_skill_has_exactly_its_reviewed_files(self) -> None:
        assets = self.materialize()
        self.assertEqual({p.name for p in (assets / "skills").iterdir()}, set(SKILL_MANIFEST))
        for name, manifest in SKILL_MANIFEST.items():
            skill = assets / "skills" / name
            with self.subTest(skill=name):
                self.assertEqual(
                    {p.relative_to(skill).as_posix() for p in skill.rglob("*") if p.is_file()},
                    set(manifest.files),
                )
                for filename, source in manifest.imports.items():
                    self.assertEqual((skill / filename).read_bytes(), (REPO_ROOT / source).read_bytes())
                    self.assertEqual((skill / filename).stat().st_mode & 0o111, (REPO_ROOT / source).stat().st_mode & 0o111)
        self.assertTrue((assets / "skills/autonomous-gpu-kernel-timeline/backends/cuda_backend/test_backend.cu").is_file())
        for name in ("gen-plan", "gpu-kernel-episode-loop"):
            self.assertFalse((assets / "skills" / name).exists())

    def test_unlisted_upstream_additions_do_not_change_the_snapshot(self) -> None:
        repository = self.copy_sources()
        first = self.materialize(repository)
        for manifest in SKILL_MANIFEST.values():
            root = repository / manifest.root
            (root / "README.md").write_text("obsolete installation and Git instructions")
            (root / "private-link").symlink_to(self.root)
        upstream = repository / "3rdparty/ncu-report-skill"
        (upstream / "SKILL.md").write_text("Build a standalone harness and profile twice")
        (upstream / "README.md").write_text("Install and git commit")
        self.assertEqual(first, self.materialize(repository))
        helper = upstream / "helpers/ncu_utils.py"
        helper.write_bytes(helper.read_bytes() + b"\n# upstream helper update\n")
        self.assertNotEqual(first, self.materialize(repository))

    def test_missing_imported_dependency_fails_even_with_a_cached_snapshot(self) -> None:
        repository = self.copy_sources()
        first = self.materialize(repository)
        (repository / "3rdparty/ncu-report-skill/helpers/ncu_utils.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "not installed.*ncu_utils"):
            self.materialize(repository)
        self.assertTrue((first / "skills/ncu-report-skill/helpers/ncu_utils.py").is_file())

    def test_listed_file_and_parent_symlinks_are_rejected(self) -> None:
        repository = self.copy_sources()
        source = repository / "3rdparty/ncu-report-skill/helpers"
        original = source / "ncu_utils.py"
        saved = source / "saved.py"
        original.rename(saved)
        original.symlink_to(saved)
        with self.assertRaisesRegex(RuntimeError, "cannot be a symlink"):
            self.materialize(repository)
        original.unlink()
        saved.rename(original)
        moved = source.with_name("saved-helpers")
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "cannot be a symlink"):
            self.materialize(repository)

    def test_manifest_paths_and_dependencies_are_validated(self) -> None:
        name = "KernelWiki"
        original = SKILL_MANIFEST[name]
        invalid = [
            replace(original, files=("SKILL.md", "SKILL.md")),
            replace(original, files=()),
            replace(original, imports={"unlisted.py": "tools/sandbox.py"}),
        ]
        for path in ("../secret", "/etc/passwd", "./SKILL.md", "a//b", "a\\b"):
            invalid.append(replace(original, files=("SKILL.md", path)))
            invalid.append(replace(original, imports={"SKILL.md": path}))
        for manifest in invalid:
            with self.subTest(manifest=manifest), patch.dict(SKILL_MANIFEST, {name: manifest}):
                with self.assertRaisesRegex(RuntimeError, "manifest|relative file path"):
                    self.materialize()

    def test_all_mounted_markdown_links_resolve_and_old_workflows_are_absent(self) -> None:
        skills = self.materialize() / "skills"
        for path in skills.rglob("*.md"):
            content = path.read_text()
            with self.subTest(path=path.relative_to(skills)):
                for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", content):
                    if "://" in target:
                        continue
                    resolved = (path.parent / target).resolve()
                    self.assertTrue(resolved.is_relative_to(skills.resolve()), target)
                    self.assertTrue(resolved.exists(), target)
                for obsolete in ("iteration_trace.py", "phase-start", "phase-end", "<PROFILE_DIR>", "episode profile directory", "git commit", "git push"):
                    self.assertNotIn(obsolete, content)
        ncu = skills / "ncu-report-skill"
        for excluded in ("README.md", "reference", "blackwell-cuda-programming.md", "helpers/README.md", "helpers/harness_template.cu", "helpers/list_flashinfer_workloads.py", "helpers/safetensors_loader.h"):
            self.assertFalse((ncu / excluded).exists(), excluded)

    def test_custom_gpu_examples_use_dev_and_explicit_inputs(self) -> None:
        from supervisor.gateway import build_parser

        assets = self.materialize()
        examples = (
            "ncu-report-skill/references/report-analysis.md",
            "autonomous-gpu-kernel-timeline/references/iket-quickstart.md",
            "ppu-acu-joint-profile/references/remote-capture.md",
        )
        count = 0
        for name in examples:
            content = (assets / "skills" / name).read_text().replace("\\\n", " ")
            for command in content.splitlines():
                if not command.startswith("python3 tools/sandbox.py "):
                    continue
                with self.subTest(command=command):
                    args = shlex.split(command)[2:]
                    parsed = build_parser().parse_args(args)
                    self.assertEqual(parsed.kind, "dev")
                    self.assertGreaterEqual(len(parsed.input), 2)
                    self.assertNotIn("--hardware", args)
                    self.assertIn("--sync", args)
                    self.assertIn("--", args)
                    count += 1
        self.assertEqual(count, 4)

    def test_installed_helper_entrypoints_keep_their_import_dependencies(self) -> None:
        assets = self.materialize()
        # NCU's SDK is a GPU-worker dependency. Stub only that SDK for CLI/import smoke tests;
        # loading real reports and compiling GPU probes are outside this CPU test.
        entrypoints = [
            "ncu-report-skill/helpers/analyze_reports.py",
            "ncu-report-skill/helpers/extract_stall_hotspots.py",
            "ncu-report-skill/helpers/plot_timeline.py",
            "autonomous-gpu-kernel-timeline/scripts/timeline.py",
            *[f"ppu-acu-joint-profile/scripts/{name}.py" for name in ("acu_report", "critical_path", "merge", "profile_report", "timeline")],
        ]
        loader = (
            "import runpy,sys,types; sys.modules['ncu_report']=types.ModuleType('ncu_report'); "
            "p=sys.argv[1]; sys.path.insert(0,str(__import__('pathlib').Path(p).parent)); "
            "sys.argv=[p,*sys.argv[2:]]; runpy.run_path(p,run_name='__main__')"
        )
        for name in entrypoints:
            with self.subTest(entrypoint=name):
                result = subprocess.run(
                    [sys.executable, "-c", loader, str(assets / "skills" / name),
                     *(["decode"] if name == "ppu-acu-joint-profile/scripts/timeline.py" else []), "--help"],
                    cwd=self.workspace, text=True, capture_output=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
