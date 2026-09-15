"""Credentials must reach the CLI without appearing in any launcher's argv."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import select
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orchestrator.agent_runtime.process import descendant_process_commands, run_bounded
from orchestrator.recovery_processes import (
    HANDOFF_ID_ENV, HANDOFF_LOCK_FD_ENV, STATE_FILE_ENV, spawn_owned_session,
)
from orchestrator.sandbox_launch import SandboxLaunch
from orchestrator.supervisor_runtime import SupervisorRuntime, SupervisorRuntimeConfig, activate_supervisor_runtime


class SandboxEnvironmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "kernel.py").write_text("pass\n")
        self.home = self.root / "home"
        self.home.mkdir()
        self.environment = {
            "HOME": str(self.home), "PATH": os.environ.get("PATH", os.defpath),
            "ATREX_AGENT_CLI": "claude", "ANTHROPIC_AUTH_TOKEN": "dummy-provider-secret-123",
        }

    def runtime(self, executable="/usr/bin/true"):
        runtime = SupervisorRuntime(SupervisorRuntimeConfig(
            repository_root=Path(__file__).resolve().parent.parent,
            hardware="test", sandbox_timeout=10, agent_sandbox="bwrap", bwrap_executable=executable,
        ), self.environment)
        runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))

        def cleanup():
            runtime._server = None
            runtime.close()

        self.addCleanup(cleanup)
        return runtime

    def assert_closed(self, descriptor):
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_anonymous_arguments_round_trip_large_and_special_environment(self):
        environment = {**self.environment, "LARGE": "x" * 100_000, "EMPTY": "", "TEXT": "a=b\n中文 ' \" --args"}
        launch = SandboxLaunch.with_bwrap_environment(["bwrap"], ["claude", "--version"], environment)
        self.addCleanup(launch.close)
        descriptor, = launch.pass_fds
        self.assertFalse(os.get_inheritable(descriptor))
        self.assertEqual(os.fstat(descriptor).st_nlink, 0)
        self.assertEqual(launch.command, ["bwrap", "--args", str(descriptor), "--", "claude", "--version"])
        self.assertNotIn(environment["ANTHROPIC_AUTH_TOKEN"], repr(launch))
        options = os.pread(descriptor, 200_000, 0).split(b"\0")
        self.assertEqual(options.pop(), b"")
        self.assertEqual(options.pop(0), b"--clearenv")
        decoded = {}
        for index in range(0, len(options), 3):
            self.assertEqual(options[index], b"--setenv")
            decoded[os.fsdecode(options[index + 1])] = os.fsdecode(options[index + 2])
        self.assertEqual(decoded, environment)
        launch.close()
        launch.close()
        self.assert_closed(descriptor)

    def test_invalid_environment_is_rejected_before_opening_file(self):
        for environment in ({"": "value"}, {"A=B": "value"}, {"A\0": "value"}, {"A": "secret\0value"}):
            with self.subTest(environment=environment), patch("orchestrator.sandbox_launch.tempfile.TemporaryFile") as create:
                with self.assertRaisesRegex(ValueError, "invalid name or NUL"):
                    SandboxLaunch.with_bwrap_environment(["bwrap"], ["true"], environment)
                create.assert_not_called()

    def test_argument_write_failure_closes_file(self):
        arguments = tempfile.TemporaryFile(mode="w+b")
        self.addCleanup(arguments.close)
        descriptor = arguments.fileno()
        broken = Mock(wraps=arguments)
        broken.write.side_effect = OSError("disk full")
        with patch("orchestrator.sandbox_launch.tempfile.TemporaryFile", return_value=broken):
            with self.assertRaisesRegex(OSError, "disk full"):
                SandboxLaunch.with_bwrap_environment(["bwrap"], ["true"], self.environment)
        self.assert_closed(descriptor)

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_lease_revoke_and_runtime_close_release_files(self, _system):
        runtime = self.runtime()
        for cleanup in ("lease", "runtime"):
            lease = runtime.prepare_session(["claude", "--version"], self.workspace, self.environment)
            descriptor, = lease.pass_fds
            self.assertNotIn(lease.token, "\0".join(lease.command))
            self.assertNotIn(self.environment["ANTHROPIC_AUTH_TOKEN"], "\0".join(lease.command))
            self.assertNotIn("/usr/bin/env", lease.command)
            if cleanup == "lease":
                lease.close()
            else:
                runtime._server = None
                runtime.close()
            self.assert_closed(descriptor)
            self.assertIsNone(runtime.authorize(lease.token))

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_capture_setup_failure_revokes_and_closes_arguments(self, _system):
        runtime = self.runtime()
        descriptors = []

        def capture_failure(*args, **kwargs):
            descriptors.extend(next(iter(runtime._launches.values())).pass_fds)
            raise OSError("capture setup failed")

        with patch("orchestrator.session_capture.SessionCapture", side_effect=capture_failure):
            with self.assertRaisesRegex(OSError, "capture setup failed"):
                runtime.prepare_session(["claude"], self.workspace, self.environment)
        self.assertEqual(len(descriptors), 1)
        self.assert_closed(descriptors[0])
        self.assertFalse(runtime._capabilities)
        self.assertFalse(runtime._launches)

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_spawn_failure_revokes_and_closes_arguments(self, _system):
        runtime = self.runtime()
        descriptors = []

        def fail_spawn(*args, **kwargs):
            descriptors.extend(kwargs["inherited_fds"])
            raise OSError("spawn failed")

        with activate_supervisor_runtime(runtime), patch(
            "orchestrator.agent_runtime.process.spawn_owned_session", side_effect=fail_spawn,
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                run_bounded(["claude"], self.workspace, 5, self.environment)
        self.assertEqual(len(descriptors), 1)
        self.assert_closed(descriptors[0])
        self.assertFalse(runtime._capabilities)

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_successful_spawn_closes_parent_arguments_before_capture(self, _system):
        runtime = self.runtime()
        descriptors = []

        def spawn(command, **kwargs):
            descriptors.extend(kwargs["inherited_fds"])
            return subprocess.Popen(command, env=kwargs["environment"], pass_fds=kwargs["inherited_fds"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def communicate(capture, process, timeout=None):
            self.assertEqual(len(descriptors), 1)
            self.assert_closed(descriptors[0])
            return process.communicate(timeout=timeout)

        with activate_supervisor_runtime(runtime), patch(
            "orchestrator.agent_runtime.process.spawn_owned_session", side_effect=spawn,
        ), patch("orchestrator.session_capture.SessionCapture.communicate", autospec=True, side_effect=communicate):
            self.assertEqual(run_bounded(["true"], self.workspace, 5, self.environment)[2], 0)
        self.assertFalse(runtime._capabilities)

    def test_owned_session_forwards_requested_fd_with_and_without_recovery(self):
        for recovery in (False, True):
            with self.subTest(recovery=recovery), tempfile.TemporaryFile() as payload, tempfile.TemporaryFile() as lock:
                payload.write(b"private test payload")
                payload.seek(0)
                environment = dict(self.environment)
                if recovery:
                    environment.update({
                        HANDOFF_ID_ENV: "a" * 32, HANDOFF_LOCK_FD_ENV: str(lock.fileno()),
                        STATE_FILE_ENV: str(self.root / "environment.json"),
                    })
                process = spawn_owned_session(
                    [sys.executable, "-c", "import os,sys; print(os.read(int(sys.argv[1]), 100).decode())", str(payload.fileno())],
                    role="fd-test", environment=environment, inherited_fds=(payload.fileno(),),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=15)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout.strip(), "private test payload")
                self.assertFalse(os.get_inheritable(payload.fileno()))

    @unittest.skipUnless(platform.system() == "Linux" and shutil.which("bwrap"), "requires Linux Bubblewrap")
    def test_real_bwrap_hides_secrets_from_host_cmdline_and_closes_argument_fd(self):
        runtime = self.runtime(shutil.which("bwrap"))
        # Values never occur in command text, including this child script.
        child_code = "import os,hashlib; print(hashlib.sha256(os.environ['ANTHROPIC_AUTH_TOKEN'].encode()).hexdigest())"
        code = (
            "import hashlib,json,os,subprocess,sys\n"
            "from pathlib import Path\n"
            "keys=['ANTHROPIC_AUTH_TOKEN','ATREX_AKA_RUNTIME_TOKEN']\n"
            f"child=subprocess.check_output([sys.executable,'-c',{child_code!r}],text=True).strip()\n"
            "files=[]\n"
            "for path in Path('/proc/self/fd').iterdir():\n"
            "    try: s=path.stat(); files.append([s.st_dev,s.st_ino])\n"
            "    except OSError: pass\n"
            "print(json.dumps({'hashes':[hashlib.sha256(os.environ[k].encode()).hexdigest() for k in keys], 'child':child, 'files':files}),flush=True)\n"
            "input()\n"
        )
        for recovery in (False, True):
            with self.subTest(recovery=recovery), tempfile.TemporaryFile() as lock:
                environment = dict(self.environment)
                if recovery:
                    environment.update({HANDOFF_ID_ENV: "b" * 32, HANDOFF_LOCK_FD_ENV: str(lock.fileno()),
                                        STATE_FILE_ENV: str(self.root / "bwrap-environment.json")})
                lease = runtime.prepare_session([sys.executable, "-c", code], self.workspace, environment)
                descriptor, = lease.pass_fds
                stat = os.fstat(descriptor)
                process = spawn_owned_session(
                    lease.command, role="bwrap-fd-test", environment=lease.environment, inherited_fds=lease.pass_fds,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                lease.close_launch_fds()
                try:
                    self.assertTrue(select.select([process.stdout], [], [], 15)[0], "sandbox did not become ready")
                    line = process.stdout.readline()
                    self.assertTrue(line, "sandbox exited without an environment report")
                    result = json.loads(line)
                    secrets = [environment["ANTHROPIC_AUTH_TOKEN"], lease.token]
                    hashes = [hashlib.sha256(value.encode()).hexdigest() for value in secrets]
                    self.assertEqual(result["hashes"], hashes)
                    self.assertEqual(result["child"], hashes[0])
                    self.assertNotIn([stat.st_dev, stat.st_ino], result["files"])
                    commands = [(process.pid, Path(f"/proc/{process.pid}/cmdline").read_text()), *descendant_process_commands(process.pid)]
                    for pid, command in commands:
                        for secret in secrets:
                            self.assertNotIn(secret, command, f"secret in process {pid} argv")
                        for path in Path(f"/proc/{pid}/fd").glob("*"):
                            try:
                                value = path.stat()
                            except OSError:
                                continue
                            self.assertNotEqual((stat.st_dev, stat.st_ino), (value.st_dev, value.st_ino))
                    stdout, stderr = process.communicate("\n", timeout=15)
                    self.assertEqual(process.returncode, 0, stderr + stdout)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                    lease.close()
