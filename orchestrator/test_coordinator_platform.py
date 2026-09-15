from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from . import agent_sandbox, optimize


class CoordinatorPlatformTests(unittest.TestCase):
    def test_linux_auto_and_explicit_bwrap_resolve_executable(self) -> None:
        for mode in ("auto", "bwrap"):
            with (
                self.subTest(mode=mode),
                patch.object(agent_sandbox.platform, "system", return_value="Linux"),
                patch.object(agent_sandbox, "_bwrap_path", return_value="/opt/bin/bwrap") as find,
            ):
                self.assertEqual(
                    agent_sandbox.require_campaign_sandbox(mode, "custom-bwrap"),
                    "/opt/bin/bwrap",
                )
                find.assert_called_once_with("custom-bwrap")

    def test_non_linux_and_missing_bwrap_have_actionable_errors(self) -> None:
        for system in ("Darwin", "Windows", "Linux"):
            for mode in ("auto", "bwrap"):
                with (
                    self.subTest(system=system, mode=mode),
                    patch.object(agent_sandbox.platform, "system", return_value=system),
                    patch.object(agent_sandbox, "_bwrap_path", return_value=None),
                    self.assertRaises(RuntimeError) as error,
                ):
                    agent_sandbox.require_campaign_sandbox(mode, "missing-bwrap")
                text = str(error.exception)
                self.assertIn("docs/platforms.md", text)
                self.assertIn("Supervisor and coding Agent together", text)
                self.assertIn("remote Gateway/SSH GPU", text)
                if system == "Linux":
                    self.assertIn("Install bubblewrap", text)
                else:
                    self.assertIn(system, text)
                    self.assertIn("Lima Ubuntu", text)

    def test_none_is_not_a_campaign_escape_hatch(self) -> None:
        for system in ("Linux", "Darwin"):
            with (
                self.subTest(system=system),
                patch.object(agent_sandbox.platform, "system", return_value=system),
                self.assertRaisesRegex(RuntimeError, "trusted non-Git tool tests"),
            ):
                agent_sandbox.require_campaign_sandbox("none", "bwrap")

    def test_launch_guard_still_rejects_git_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
            for system, mode in (("Darwin", "auto"), ("Darwin", "bwrap"),
                                 ("Darwin", "none"), ("Linux", "none")):
                with (
                    self.subTest(system=system, mode=mode),
                    patch.object(agent_sandbox.platform, "system", return_value=system),
                    patch.object(agent_sandbox, "_bwrap_path", return_value="/usr/bin/bwrap"),
                    self.assertRaisesRegex(RuntimeError, "docs/platforms.md"),
                ):
                    agent_sandbox.wrap_agent_command(
                        ["python3", "--version"], workspace=workspace, environment={},
                        repository_root=workspace, provider_homes=workspace / "provider-homes",
                        hidden_host_paths=(), mode=mode, bwrap_executable="bwrap",
                    )
            self.assertFalse((workspace / "provider-homes").exists())

    def test_trusted_non_git_tools_remain_usable_without_bwrap(self) -> None:
        for system, mode in (("Darwin", "auto"), ("Darwin", "none"),
                             ("Linux", "auto"), ("Linux", "none")):
            with (
                self.subTest(system=system, mode=mode),
                tempfile.TemporaryDirectory() as directory,
                patch.object(agent_sandbox.platform, "system", return_value=system),
                patch.object(agent_sandbox, "_bwrap_path", return_value=None),
            ):
                workspace = Path(directory)
                launch = agent_sandbox.wrap_agent_command(
                    ["python3", "--version"], workspace=workspace, environment={"TEST": "value"},
                    repository_root=workspace, provider_homes=workspace / "provider-homes",
                    hidden_host_paths=(), mode=mode, bwrap_executable="bwrap",
                )
                self.addCleanup(launch.close)
                self.assertEqual(launch.command, ["python3", "--version"])
                self.assertEqual(launch.environment, {"TEST": "value"})

    def test_cli_fails_before_workspace_gpu_submodules_or_agent(self) -> None:
        for system, mode in (("Darwin", "auto"), ("Darwin", "none"),
                             ("Darwin", "bwrap"), ("Linux", "auto"),
                             ("Linux", "bwrap"), ("Linux", "none")):
            with (
                self.subTest(system=system, mode=mode),
                tempfile.TemporaryDirectory() as directory,
                patch.object(agent_sandbox.platform, "system", return_value=system),
                patch.object(agent_sandbox, "_bwrap_path", return_value=None),
                patch.object(optimize, "_resolve_op") as resolve,
                patch.object(optimize, "detect_arch") as gpu,
                patch.object(optimize, "ensure_submodules") as submodules,
                patch.object(optimize, "Campaign") as campaign,
                patch.object(optimize, "SupervisorRuntime") as runtime,
                patch.object(optimize.shutil, "which") as find_cli,
                redirect_stderr(io.StringIO()) as error,
            ):
                workspace = Path(directory) / "runs"
                with self.assertRaises(SystemExit) as exit_status:
                    optimize._run_main([
                        "--op-dir", str(Path(directory) / "operator"),
                        "--platform", "L20N", "--framework", "Triton",
                        "--sandbox-url", "https://gateway.example.test",
                        "--workspace", str(workspace), "--agent-sandbox", mode,
                    ])
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn("docs/platforms.md", error.getvalue())
                self.assertFalse(workspace.exists())
                for mock in (resolve, gpu, submodules, campaign, runtime, find_cli):
                    mock.assert_not_called()

    def test_supported_coordinator_reaches_operator_resolution(self) -> None:
        for mode in ("auto", "bwrap"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as directory,
                patch.object(agent_sandbox.platform, "system", return_value="Linux"),
                patch.object(agent_sandbox, "_bwrap_path", return_value="/usr/bin/bwrap"),
                patch.object(optimize.shutil, "which", return_value="/usr/bin/claude"),
                patch.object(optimize, "_resolve_op", side_effect=LookupError("operator probe")) as resolve,
            ):
                workspace = Path(directory) / "runs"
                with self.assertRaisesRegex(LookupError, "operator probe"):
                    optimize._run_main([
                        "--op-dir", "/operator", "--platform", "L20N",
                        "--sandbox-url", "https://gateway.example.test",
                        "--agent-cli", "claude", "--agent-sandbox", mode,
                        "--workspace", str(workspace),
                    ])
                resolve.assert_called_once()
                self.assertTrue(workspace.is_dir())

    def test_cli_help_remains_available_on_macos(self) -> None:
        with (
            patch.object(agent_sandbox.platform, "system", return_value="Darwin"),
            patch.object(optimize, "require_campaign_sandbox") as check,
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as exit_status,
        ):
            optimize._run_main(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        check.assert_not_called()
        self.assertIn("Linux + Bubblewrap", output.getvalue())
        self.assertIn("Lima Ubuntu", output.getvalue())


if __name__ == "__main__":
    unittest.main()
