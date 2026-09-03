from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import environment_recovery
from orchestrator import optimize
from orchestrator.agent_runtime.process import run_bounded
from tools import monitor_optimize_tasks, sandbox


class SandboxSSHTest(unittest.TestCase):
    def test_default_health_probe_matches_recovery_monitor(self) -> None:
        self.assertEqual(
            sandbox.DEFAULT_SSH_HEALTH_COMMAND,
            environment_recovery.DEFAULT_SSH_HEALTH_COMMAND,
        )

    def test_ssh_target_rejects_option_injection(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-option"):
            sandbox._validate_ssh_target("-oProxyCommand=bad")
        self.assertEqual(sandbox._validate_ssh_target("user@gpu-alias"), "user@gpu-alias")

    def test_candidate_failure_with_healthy_gpu_is_not_environment_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state = workspace / "failure.json"
            environment = {"ATREX_ENVIRONMENT_STATE_FILE": str(state)}
            command_result = subprocess.CompletedProcess(
                args=["ssh"], returncode=2, stdout="candidate failed\n", stderr=""
            )
            health_result = subprocess.CompletedProcess(
                args=["ssh"], returncode=0, stdout="healthy\n", stderr=""
            )
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                sandbox, "_run_ssh_job", return_value=command_result
            ), mock.patch.object(
                sandbox, "_run_ssh_health", return_value=health_result
            ):
                result = sandbox._main(
                    [
                        "--hardware",
                        "H20",
                        "--kind",
                        "run",
                        "--ssh",
                        "gpu",
                        "--workspace",
                        str(workspace),
                        "--no-sync",
                        "--",
                        "false",
                    ]
                )
            self.assertEqual(result, 2)
            self.assertFalse(state.exists())

    def test_command_and_health_failure_writes_private_environment_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state = workspace / "failure.json"
            command_result = subprocess.CompletedProcess(
                args=["ssh"], returncode=1, stdout="", stderr="command failed"
            )
            health_result = subprocess.CompletedProcess(
                args=["ssh"], returncode=1, stdout="", stderr="driver unavailable"
            )
            with mock.patch.dict(
                os.environ,
                {"ATREX_ENVIRONMENT_STATE_FILE": str(state)},
                clear=False,
            ), mock.patch.object(
                sandbox, "_run_ssh_job", return_value=command_result
            ), mock.patch.object(
                sandbox, "_run_ssh_health", return_value=health_result
            ):
                result = sandbox._main(
                    [
                        "--hardware",
                        "H20",
                        "--ssh",
                        "gpu",
                        "--workspace",
                        str(workspace),
                        "--no-sync",
                        "--",
                        "false",
                    ]
                )
            self.assertEqual(result, sandbox.ENVIRONMENT_TEMPFAIL)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(payload["stage"], "post-command-health")
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o600)

    def test_ssh_transport_exit_requests_recovery_without_health_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state = workspace / "failure.json"
            transport_failure = subprocess.CompletedProcess(
                args=["ssh"], returncode=255, stdout="", stderr="connection reset"
            )
            with mock.patch.dict(
                os.environ,
                {"ATREX_ENVIRONMENT_STATE_FILE": str(state)},
                clear=False,
            ), mock.patch.object(
                sandbox, "_run_ssh_job", return_value=transport_failure
            ), mock.patch.object(sandbox, "_run_ssh_health") as health:
                result = sandbox._main(
                    [
                        "--hardware",
                        "H20",
                        "--ssh",
                        "gpu",
                        "--workspace",
                        str(workspace),
                        "--no-sync",
                        "--",
                        "false",
                    ]
                )
            self.assertEqual(result, sandbox.ENVIRONMENT_TEMPFAIL)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["stage"],
                "execution-transport",
            )
            health.assert_not_called()

    def test_check_health_requires_ssh(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires --ssh"):
            sandbox._main(["--hardware", "H20", "--check-health"])

    def test_ssh_backend_executes_and_synchronizes_with_openssh_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            workspace = root / "workspace"
            bin_dir.mkdir()
            workspace.mkdir()
            ssh = bin_dir / "ssh"
            ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess, sys\n"
                "command = sys.argv[-1]\n"
                "raise SystemExit(subprocess.run(['bash', '-c', command]).returncode)\n",
                encoding="utf-8",
            )
            scp = bin_dir / "scp"
            scp.write_text(
                "#!/usr/bin/env python3\n"
                "import shutil, sys\n"
                "args = [value for value in sys.argv[1:] if value != '-q']\n"
                "if ':' in args[0]:\n"
                "    source = args[0].split(':', 1)[1]\n"
                "    shutil.copy2(source, args[1])\n"
                "else:\n"
                "    destination = args[-1].split(':', 1)[1]\n"
                "    for source in args[:-1]: shutil.copy2(source, destination)\n",
                encoding="utf-8",
            )
            base64_tool = bin_dir / "base64"
            base64_tool.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $# -eq 2 && $1 == -d ]]; then "
                "exec /usr/bin/base64 -d -i \"$2\"; fi\n"
                "exec /usr/bin/base64 \"$@\"\n",
                encoding="utf-8",
            )
            ssh.chmod(0o700)
            scp.chmod(0o700)
            base64_tool.chmod(0o700)
            (workspace / "run.py").write_text(
                "import os\n"
                "assert os.environ['INIT_VALUE'] == 'ready'\n"
                "open('result.txt', 'w').write('gpu-result\\n')\n"
                "print('remote-ok')\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", "")},
                clear=False,
            ):
                result = sandbox._main(
                    [
                        "--hardware",
                        "H20",
                        "--kind",
                        "run",
                        "--ssh",
                        "gpu",
                        "--ssh-init",
                        "export INIT_VALUE=ready",
                        "--workspace",
                        str(workspace),
                        "--input",
                        "run.py",
                        "--sync",
                        "result.txt",
                        "--",
                        "python3",
                        "run.py",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                (workspace / "result.txt").read_text(encoding="utf-8"),
                "gpu-result\n",
            )


class ProcessGuardTest(unittest.TestCase):
    def test_environment_marker_terminates_coding_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "failure.json"
            state.write_text("{}\n", encoding="utf-8")
            stdout, stderr, returncode, timed_out = run_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                Path(directory),
                5,
                {**os.environ, "ATREX_ENVIRONMENT_STATE_FILE": str(state)},
            )
        self.assertEqual(stdout, "")
        self.assertIn("environment became unavailable", stderr)
        self.assertEqual(returncode, environment_recovery.ENVIRONMENT_TEMPFAIL)
        self.assertFalse(timed_out)


class RecoveryMetadataTest(unittest.TestCase):
    def test_configure_recovery_preserves_exact_restart_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(directory)
            script = root / "orchestrator" / "optimize.py"
            script.parent.mkdir()
            script.write_text("", encoding="utf-8")
            context = environment_recovery.configure_recovery(
                workspace_base=root,
                raw_argv=["--op-dir", "path with spaces", "--framework", "Triton"],
                optimize_script=script,
                sandbox_hardware="H20",
                ssh_target="user@gpu",
                ssh_init="source /opt/env.sh",
                health_command="gpu-health",
                poll_interval=7,
            )
            payload = json.loads(
                (context.directory / "restart.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["command"][-4:],
                ["--op-dir", "path with spaces", "--framework", "Triton"],
            )
            self.assertEqual(payload["poll_interval"], 7)
            self.assertTrue((context.directory / "recover.sh").stat().st_mode & stat.S_IXUSR)
            self.assertEqual(
                os.environ["ATREX_ENVIRONMENT_STATE_FILE"], str(context.state_file)
            )

    def test_monitor_archives_failure_after_successful_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            failure = state_dir / "failure.json"
            failure.write_text("{}\n", encoding="utf-8")
            restart = {
                "cwd": directory,
                "command": [sys.executable, "-c", "pass"],
                "environment_state_file": str(failure),
                "sandbox_hardware": "H20",
                "ssh_target": "gpu",
                "ssh_init": "",
                "health_command": "gpu-health",
                "poll_interval": 1,
            }
            (state_dir / "restart.json").write_text(
                json.dumps(restart), encoding="utf-8"
            )
            healthy = subprocess.CompletedProcess(
                args=["health"], returncode=0, stdout="ok", stderr=""
            )
            with mock.patch.object(
                monitor_optimize_tasks.subprocess, "run", return_value=healthy
            ):
                result = monitor_optimize_tasks.run_monitor(
                    state_dir, once=True, no_restart=True
                )
            self.assertEqual(result, 0)
            self.assertFalse(failure.exists())
            self.assertEqual(len(list(state_dir.glob("recovered-*.json"))), 1)
            self.assertFalse((state_dir / "monitor.lock").exists())

    def test_monitor_lock_prevents_duplicate_probe_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            (state_dir / "monitor.lock").write_text(
                str(os.getpid()) + "\n", encoding="utf-8"
            )
            failure = state_dir / "failure.json"
            metadata = {
                "cwd": directory,
                "command": [sys.executable, "-c", "pass"],
                "environment_state_file": str(failure),
                "sandbox_hardware": "H20",
                "ssh_target": "gpu",
                "ssh_init": "",
                "health_command": "gpu-health",
                "poll_interval": 1,
            }
            (state_dir / "restart.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with mock.patch.object(monitor_optimize_tasks.subprocess, "run") as probe:
                result = monitor_optimize_tasks.run_monitor(state_dir, once=True)
            self.assertEqual(result, 0)
            probe.assert_not_called()

    def test_top_level_launches_monitor_for_environment_tempfail(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(directory)
            script = root / "orchestrator" / "optimize.py"
            script.parent.mkdir()
            script.write_text("", encoding="utf-8")
            context = environment_recovery.configure_recovery(
                workspace_base=root,
                raw_argv=["--op-dir", "op"],
                optimize_script=script,
                sandbox_hardware="H20",
                ssh_target="gpu",
                ssh_init="",
                health_command="gpu-health",
                poll_interval=60,
            )
            context.state_file.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                optimize, "_run_main", return_value=environment_recovery.ENVIRONMENT_TEMPFAIL
            ), mock.patch.object(
                optimize, "launch_recovery_monitor", return_value=123
            ) as launch:
                result = optimize.main([])
            self.assertEqual(result, environment_recovery.ENVIRONMENT_TEMPFAIL)
            launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
