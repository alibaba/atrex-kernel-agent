"""Installation mounts must not restore masked credentials, histories or source trees."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.agent_installations import installation_mounts
from orchestrator.agent_sandbox import VISIBLE_WORKSPACE, wrap_agent_command


class AgentInstallationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "operator"
        self.home.mkdir()
        self.private = self.root / "private-runtime"
        self.private.mkdir()
        self.environment = {"HOME": str(self.home), "PATH": f"{self.home}/.local/bin:{os.defpath}"}

    def file(self, path, content="#!/bin/sh\necho INSTALLED\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)
        return path

    def npm(self, root, name, **fields):
        self.file(root / "package.json", json.dumps({"name": name, **fields}))
        return self.file(root / "bin/cli")

    def mounts(self, executable, *backends):
        return installation_mounts(
            [str(executable)], backends, self.environment, self.home,
            hidden_paths=(self.home,), forbidden_paths=(self.private, VISIBLE_WORKSPACE),
        )

    def assert_not_exposed(self, mounts, private_path):
        for source, destination in mounts:
            if private_path == destination or (source.is_dir() and private_path.is_relative_to(destination)):
                self.fail(f"Private path {private_path} exposed by {(source, destination)}")

    def test_native_cli_restores_only_its_file_not_provider_home(self):
        for relative in (".claude/local/claude", ".qoder/bin/qodercli/version-1", ".local/bin/claude"):
            with self.subTest(relative=relative):
                executable = self.file(self.home / relative)
                mounts = self.mounts(executable)
                self.assertIn((executable, executable), mounts)
                for private in (".claude/projects", ".claude/.credentials.json", ".qoder/tasks", ".local/share/secrets"):
                    self.assert_not_exposed(mounts, self.home / private)
                self.assertNotIn(self.home / relative.split("/")[0], [source for source, _ in mounts])

    def test_npm_only_restores_selected_package_and_declared_dependencies(self):
        modules = self.home / ".local/lib/node_modules"
        root = modules / "@openai/codex"
        target = self.npm(root, "@openai/codex", optionalDependencies={"@openai/codex-linux-arm64": "1"}, dependencies={"helper": "1"})
        self.npm(modules / "@openai/codex-linux-arm64", "@openai/codex-linux-arm64")
        self.npm(modules / "helper", "helper", dependencies={"@openai/codex": "1"})
        self.npm(modules / "unrelated-private-package", "unrelated-private-package")
        alias = self.home / ".local/bin/codex"
        alias.parent.mkdir(parents=True)
        alias.symlink_to(target)
        mounts = self.mounts(alias)
        self.assertIn((root, root), mounts)
        self.assertIn((modules / "helper", modules / "helper"), mounts)
        self.assertIn((target, alias), mounts)
        self.assert_not_exposed(mounts, modules / "unrelated-private-package")
        self.assert_not_exposed(mounts, self.home / ".local/share/private-transcript")

    def test_external_cli_symlink_restores_hidden_canonical_installation(self):
        root = self.home / ".local/lib/node_modules/@anthropic-ai/claude-code"
        target = self.npm(root, "@anthropic-ai/claude-code")
        alias = self.file(self.root / "system-bin/placeholder").with_name("claude")
        alias.symlink_to(target)
        mounts = self.mounts(alias)
        self.assertIn((root, root), mounts)
        self.assertNotIn(alias, [destination for _, destination in mounts])

    def test_legacy_claude_local_package_does_not_expose_sibling_histories(self):
        root = self.home / ".claude/local"
        target = self.npm(root, "@anthropic-ai/claude-code")
        self.assertIn((root, root), self.mounts(target))
        self.assert_not_exposed(self.mounts(target), self.home / ".claude/projects")

    def test_venv_in_project_restores_libraries_and_config_not_project(self):
        project = self.home / "project"
        prefix = project / ".venv"
        self.file(prefix / "pyvenv.cfg", "include-system-site-packages = false\n")
        self.file(prefix / "lib/python3.12/site-packages/example.py", "VALUE = 1\n")
        alias = prefix / "bin/python3"
        alias.parent.mkdir()
        alias.symlink_to(sys.executable)
        with patch("orchestrator.agent_installations.sys.executable", str(alias)):
            mounts = self.mounts("/bin/sh")
        self.assertIn((Path(sys.executable).resolve(), alias), mounts)
        self.assertIn((prefix / "lib", prefix / "lib"), mounts)
        self.assertIn((prefix / "pyvenv.cfg", prefix / "pyvenv.cfg"), mounts)
        self.assert_not_exposed(mounts, project / "supervisor/private.json")
        self.assert_not_exposed(mounts, self.home / "other-project")

    def test_user_installed_node_interpreter_is_an_individual_file(self):
        node = self.file(self.home / ".nvm/versions/node/v22/bin/node")
        cli = self.file(self.home / ".local/bin/codex", "#!/usr/bin/env node\nconsole.log('ok')\n")
        self.environment["PATH"] = f"{node.parent}:{self.environment['PATH']}"
        mounts = self.mounts(cli)
        self.assertIn((node, node), mounts)
        self.assert_not_exposed(mounts, self.home / ".nvm/private-state")

    def test_executable_symlinks_cannot_restore_provider_state_or_private_runtime(self):
        for target in (self.home / ".claude/.credentials.json", self.home / ".codex/sessions/private.json", self.private / "ledger.json"):
            with self.subTest(target=target):
                self.file(target, "private")
                alias = self.home / ".local/bin/claude"
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(target)
                try:
                    with self.assertRaisesRegex(RuntimeError, "Provider state|private Runtime"):
                        self.mounts(alias)
                finally:
                    alias.unlink()

    def test_dependency_symlinks_cannot_restore_provider_home_or_private_runtime(self):
        root = self.home / ".local/lib/node_modules/@openai/codex"
        target = self.npm(root, "@openai/codex", dependencies={"bad": "1"})
        for secret in (self.home / ".codex", self.private):
            with self.subTest(secret=secret):
                self.file(secret / "package.json", '{"name":"bad"}')
                link = root / "node_modules/bad"
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(secret, target_is_directory=True)
                try:
                    with self.assertRaisesRegex(RuntimeError, "Provider Home|private Runtime"):
                        self.mounts(target)
                finally:
                    link.unlink()

    def test_invalid_dependency_name_is_not_used_as_a_host_path(self):
        root = self.home / ".local/lib/node_modules/@openai/codex"
        target = self.npm(root, "@openai/codex", dependencies={"../../../.claude": "1"})
        with self.assertRaisesRegex(RuntimeError, "dependency name"):
            self.mounts(target)

    def test_directory_alias_to_arbitrary_project_is_rejected(self):
        root = self.home / ".local/lib/node_modules/@openai/codex"
        target = self.npm(root, "@openai/codex", dependencies={"bad": "1"})
        project = self.home / "unrelated-project"
        self.file(project / "package.json", '{"name":"bad"}')
        link = root / "node_modules/bad"
        link.parent.mkdir()
        link.symlink_to(project, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "Unrecognized Agent package symlink"):
            self.mounts(target)

    def test_command_alias_cannot_shadow_forbidden_destination(self):
        executable = self.file(self.home / ".local/bin/claude")
        with self.assertRaisesRegex(RuntimeError, "private Runtime"):
            installation_mounts(
                [str(executable)], (), self.environment, self.home,
                hidden_paths=(self.home,), forbidden_paths=(executable.parent,),
            )

    def sandbox_fixture(self, bwrap):
        project = self.home / "project"
        workspace = project / "episode"
        self.file(workspace / "kernel.py", "pass\n")
        assets = self.private / "assets"
        self.file(assets / "tools/sandbox.py", "pass\n")
        (assets / "skills").mkdir()
        view = self.private / "view"
        self.file(view / "kernel.py", "pass\n")
        provider_homes = self.private / "providers"
        provider_homes.mkdir()
        cli = self.npm(self.home / ".local/lib/node_modules/@anthropic-ai/claude-code", "@anthropic-ai/claude-code")
        alias = self.home / ".local/bin/claude"
        alias.parent.mkdir(parents=True)
        alias.symlink_to(cli)
        qoder = self.file(self.home / ".qoder/bin/qodercli/version-1")
        (self.home / ".qoder/skills").mkdir()
        (alias.parent / "qodercli").symlink_to(qoder)
        secrets = (
            self.home / ".claude/projects/old-session.jsonl",
            self.home / ".claude/.credentials.json",
            self.home / ".qoder/tasks/old-session.jsonl",
            self.home / ".local/share/operator-private.json",
            project / "supervisor/ledger.json",
        )
        for path in secrets:
            self.file(path, "DUMMY_PRIVATE_FIXTURE")
        prefix = project / ".venv"
        self.file(prefix / "pyvenv.cfg", f"home = {Path(sys.executable).resolve().parent}\ninclude-system-site-packages = false\n")
        library = prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        self.file(library / "installation_probe.py", "VALUE = 'VENV_OK'\n")
        python = prefix / "bin/python3"
        python.parent.mkdir()
        python.symlink_to(sys.executable)
        script = "set -eu; claude; qodercli; python3 -c 'import installation_probe; print(installation_probe.VALUE)'; "
        for path in secrets:
            script += f"test ! -e {str(path)!r}; "
        script += "test -r .claude/.credentials.json; test ! -e .qoder/tasks/old-session.jsonl; echo ISOLATION_OK"
        environment = {
            **self.environment, "ATREX_AGENT_CLI": "claude", "ATREX_PLAN_REVIEW_QODER_ENABLED": "1",
            "PATH": f"{prefix}/bin:{self.environment['PATH']}",
        }
        with (
            patch("orchestrator.agent_sandbox.platform.system", return_value="Linux"),
            patch("orchestrator.agent_sandbox.materialize_agent_assets", return_value=assets),
            patch("orchestrator.agent_installations.sys.executable", str(python)),
        ):
            argv, _ = wrap_agent_command(
                ["/bin/sh", "-c", script], workspace=workspace, environment=environment,
                repository_root=project, provider_homes=provider_homes,
                hidden_host_paths=(self.private,), mode="bwrap", bwrap_executable=bwrap,
                agent_workspace=view, agent_skills=(),
            )
        return argv, secrets

    def test_full_mount_plan_never_restores_operator_home_roots(self):
        argv, secrets = self.sandbox_fixture("/usr/bin/true")
        mounts = [(Path(argv[i+1]), Path(argv[i+2])) for i, arg in enumerate(argv) if arg in {"--bind", "--ro-bind"}]
        for path in secrets:
            # The initial '/' is masked before the selective installation mounts.
            self.assert_not_exposed([pair for pair in mounts if pair[1] != Path("/")], path)
        for relative in (".claude", ".qoder", "project"):
            self.assertNotIn(self.home / relative, [dest for _, dest in mounts])

    @unittest.skipUnless(platform.system() == "Linux" and shutil.which("bwrap"), "requires Linux Bubblewrap")
    def test_real_bwrap_executes_installations_without_exposing_original_histories(self):
        argv, _ = self.sandbox_fixture(shutil.which("bwrap"))
        result = subprocess.run(argv, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("VENV_OK", result.stdout)
        self.assertIn("ISOLATION_OK", result.stdout)

    @unittest.skipUnless(
        platform.system() == "Linux" and shutil.which("bwrap") and os.environ.get("ATREX_TEST_REAL_CLIS") == "1",
        "opt-in installed CLI version smoke test on Linux",
    )
    def test_installed_clis_start_without_host_credentials(self):
        # --version only: no model calls, no login state mounted, no global writes.
        project = self.root / "project"
        workspace = project / "episode"
        self.file(workspace / "kernel.py", "pass\n")
        assets = self.private / "assets"
        self.file(assets / "tools/sandbox.py", "pass\n")
        (assets / "skills").mkdir()
        for cli in ("claude", "codex", "qodercli"):
            executable = shutil.which(cli)
            if executable is None:
                self.fail(f"CLI smoke test requested but {cli} is not installed")
            with self.subTest(cli=cli):
                view = self.private / cli / "view"
                self.file(view / "kernel.py", "pass\n")
                with patch("orchestrator.agent_sandbox.materialize_agent_assets", return_value=assets):
                    argv, _ = wrap_agent_command(
                        [executable, "--version"], workspace=workspace,
                        environment={"HOME": str(Path.home()), "PATH": os.environ["PATH"]},
                        repository_root=project, provider_homes=self.private / cli / "providers",
                        hidden_host_paths=(self.private,), mode="bwrap",
                        bwrap_executable=shutil.which("bwrap"), agent_workspace=view, agent_skills=(),
                    )
                result = subprocess.run(argv, text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertTrue(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
