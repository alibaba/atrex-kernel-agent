"""Exercise the Light-only public launch and package boundary without a GPU."""
import argparse
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aka.bootstrap.launch import create_launch
from aka.bootstrap.workflow_cli import profile_path
from aka.bootstrap.workflow_request import register_request_options
from aka.contracts.workflow import WorkflowSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.startup import WorkflowStartup

ROOT = Path(__file__).resolve().parents[1]


class LightOnlyLaunchTest(unittest.TestCase):
    def test_only_current_modes_and_request_fields(self):
        parser = argparse.ArgumentParser()
        register_request_options(parser)
        help_text = parser.format_help()
        for old in ("framework-baseline", "handoff-resumes", "convert-after", "fast-trials"):
            self.assertNotIn(old, help_text)
        self.assertIn("--warmup-iters", help_text)
        self.assertIn("--evaluation-step-timeout", help_text)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            requests = []
            for mode in ("fast", "performance", "ultra"):
                request = create_launch(root / "task", workspace_base=root,
                    settings=WorkflowSettings(platform="L20N", framework="cuda" if mode == "ultra" else "", optimization_mode=mode))
                self.assertEqual(decode_launch(encode_launch(request)), request)
                requests.append(request.request.workspace)
            self.assertEqual(len(set(requests)), 3)
        for mode in ("production", "leaderboard"):
            with self.assertRaisesRegex(ValueError, "unsupported optimization_mode"):
                WorkflowSettings(platform="L20N", framework="cuda", optimization_mode=mode)
        with self.assertRaisesRegex(ValueError, "explicit framework"):
            WorkflowSettings(platform="L20N", framework="", optimization_mode="ultra")
        with self.assertRaises(TypeError):
            WorkflowSettings(platform="L20N", framework="cuda", framework_baseline="auto")

    def test_launch_does_not_dispatch_dsl_or_measure_initial_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            launch = create_launch(root / "task", workspace_base=root,
                settings=WorkflowSettings(platform="L20N", framework="", optimization_mode="performance"))
            calls = []
            # Only the current public Campaign methods exist. There is no V0,
            # Framework Baseline, worktree, handoff or promotion callback.
            campaign = SimpleNamespace(prepare=lambda **kw: calls.append("prepare"), run=lambda: calls.append("run"))
            problem = object()
            workflow = SimpleNamespace(_load_problem=Mock(return_value=problem), _campaign=Mock(return_value=campaign))
            startup = WorkflowStartup(workflow)
            with patch("shutil.which", return_value="/tool"), patch("aka.execution.configuration.selected_ssh", return_value=False), patch("aka.workflow.light.runtime_preparation.prepare_runtime", return_value=("sm_89", "SDK")):
                self.assertEqual(startup._run(launch), 0)
            self.assertEqual(calls, ["prepare", "run"])
            workflow._load_problem.assert_called_once_with(launch.request)
            self.assertEqual(workflow._campaign.call_args.args[0].workspace, launch.request.workspace)
            self.assertEqual(workflow._campaign.call_args.args[0].settings.arch, "sm_89")
            self.assertFalse(launch.request.workspace.exists())

    def test_task_has_no_supervisor_git_mutation_workflow(self):
        candidate = GitCandidateWorkspace()
        for old in ("record", "plan", "materialize", "seal_candidate", "promote", "restore_revision", "restore_baseline", "restore_sealed_candidate", "prepare_draft", "initialize_kernel"):
            self.assertFalse(hasattr(candidate, old), old)
        for current in ("initialize_agent", "read_source", "read_revision", "import_revision"):
            self.assertTrue(callable(getattr(candidate, current)))

    def test_agent_unlimited_and_exactly_two_proc_modes(self):
        from aka.agent.provider import AgentOptions
        from aka.agent.execution.bubblewrap import BubblewrapExecution
        from aka.agent.execution.home import PREPARED_ENV
        self.assertIsNone(AgentOptions().timeout_seconds)
        with self.assertRaisesRegex(ValueError, "private or host"):
            AgentOptions(proc_mode="projected")
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            backend = SimpleNamespace(files=SimpleNamespace(state_roots=(), code_roots=()))
            for mode in ("private", "host"):
                with patch("aka.agent.execution.bubblewrap.executable", return_value="bwrap"), patch("aka.agent.execution.bubblewrap.validate_access", return_value=(workspace, (), (), ())), patch("aka.agent.execution.bubblewrap.installation_mounts", return_value=()), patch("aka.agent.execution.bubblewrap.SYSTEM_PATHS", ()), patch("aka.agent.execution.bubblewrap.SandboxLaunch.with_bwrap_environment", side_effect=lambda argv, command, values: (argv, values)):
                    argv, values = BubblewrapExecution(proc_mode=mode).launch(["python"], workspace, {PREPARED_ENV: str(workspace / "home")}, backend, workspace / "host-home", None)
                self.assertEqual("--unshare-pid" in argv, mode == "private")
                self.assertEqual("--proc" in argv, mode == "private")
                if mode == "host":
                    index = argv.index("/proc")
                    self.assertEqual(argv[index - 1:index + 2], ["--bind", "/proc", "/proc"])
                self.assertNotIn("LD_PRELOAD", values)
                self.assertFalse(any("shim" in value for value in argv))

    def test_packages_and_profiles_have_no_legacy_owner(self):
        installer = (ROOT / "skills/atrex-aka-light/scripts/install_light.sh").read_text()
        self.assertNotIn("atrex_aka_legacy", installer)
        self.assertNotIn("atrex-aka-legacy", installer)
        light = tomllib.loads((ROOT / "packages/atrex-aka-light/pyproject.toml").read_text())
        self.assertIn("atrex-aka-framework==0.2.0", light["project"]["dependencies"])
        self.assertFalse(any("legacy" in dep for dep in light["project"]["dependencies"]))
        self.assertEqual(light["project"]["entry-points"]["aka.commands"]["optimize"], "aka.bootstrap.workflow_cli:main")
        self.assertFalse((ROOT / "src/aka/workflow/legacy").exists())
        self.assertFalse((ROOT / "packages/atrex-aka-legacy").exists())
        for name in ("light", "light-sandbox"):
            profile = json.loads(profile_path(name).read_text())
            self.assertNotIn("extends", profile)
            self.assertEqual(profile["target"], "startup")
            self.assertNotIn("canonical_measurement_reader", profile["tokens"])
            self.assertNotIn("legacy", json.dumps(profile))
        for name in ("workflow", "observed", "sandbox"):
            with self.assertRaisesRegex(ValueError, "Retired launch profile"):
                profile_path(name)
        spec = importlib.util.spec_from_file_location("build_distributions", ROOT / "tools/build_distributions.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertNotIn("atrex-aka-legacy", builder.OWNERS)
        self.assertTrue((ROOT / "src/aka/experiment/measurement_store.py") in builder.owned("atrex-aka-framework"))
        self.assertFalse(any("aka.workflow.legacy" in name for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
