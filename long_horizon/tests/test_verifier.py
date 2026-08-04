from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from long_horizon.git_episode import git_head
from long_horizon.remote_abba import run as run_remote
from long_horizon.store import CampaignStore
from long_horizon.tests.helpers import init_repo, run_git
from long_horizon.verifier import (
    ABBA_RESULT_PREFIX,
    GatewayABBAValidator,
    _payload_from_stdout,
    score_verification_payload,
    verification_schedule,
)


def row(revision: str, repeat: int, latency: float) -> dict:
    return {
        "revision": revision,
        "repeat": repeat,
        "exit_code": 0,
        "result": {"all_pass": True, "latency_us_geomean": latency},
    }


class VerificationScoringTests(unittest.TestCase):
    def test_abba_winner_passes(self) -> None:
        schedule = verification_schedule(2)
        payload = {
            "schema_version": 1,
            "error": None,
            "runs": [
                row("incumbent", 0, 10.0),
                row("candidate", 0, 8.0),
                row("candidate", 1, 8.2),
                row("incumbent", 1, 10.2),
            ],
        }
        result = score_verification_payload(
            payload, schedule=schedule, repeats=2, min_improvement_pct=0.0
        )
        self.assertTrue(result.passed)
        self.assertGreater(result.improvement_pct, 15.0)

    def test_wrong_schedule_fails_closed(self) -> None:
        schedule = verification_schedule(2)
        payload = {
            "schema_version": 1,
            "error": None,
            "runs": [row("candidate", 0, 8.0), row("incumbent", 0, 10.0)],
        }
        result = score_verification_payload(
            payload, schedule=schedule, repeats=2, min_improvement_pct=0.0
        )
        self.assertEqual(result.gate, "ERROR")

    def test_regression_is_not_promoted(self) -> None:
        schedule = verification_schedule(1)
        payload = {
            "schema_version": 1,
            "error": None,
            "runs": [row("incumbent", 0, 10.0), row("candidate", 0, 11.0)],
        }
        result = score_verification_payload(
            payload, schedule=schedule, repeats=1, min_improvement_pct=0.0
        )
        self.assertEqual(result.gate, "FAIL")


class StdoutPayloadTests(unittest.TestCase):
    def test_parses_sentinel_amid_other_output(self) -> None:
        payload = {"schema_version": 1, "runs": [], "error": None}
        stdout = "gateway prelude\n" + ABBA_RESULT_PREFIX + json.dumps(payload) + "\n"
        self.assertEqual(_payload_from_stdout(stdout), payload)

    def test_missing_sentinel_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing ABBA result sentinel"):
            _payload_from_stdout("ordinary output only\n")

    def test_malformed_sentinel_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "malformed ABBA result sentinel"):
            _payload_from_stdout(ABBA_RESULT_PREFIX + "{bad json}\n")


class RemoteDriverTests(unittest.TestCase):
    def test_revisions_are_applied_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_dir = root / "request"
            (request_dir / "snapshots/incumbent").mkdir(parents=True)
            (request_dir / "snapshots/candidate").mkdir(parents=True)
            (request_dir / "snapshots/incumbent/0000.bin").write_text("10\n", encoding="utf-8")
            (request_dir / "snapshots/candidate/0000.bin").write_text("8\n", encoding="utf-8")
            harness = root / "fake_eval.py"
            harness.write_text(
                "from pathlib import Path\n"
                "value=float(Path('value.txt').read_text())\n"
                "print('[test_kernel] RESULT_JSON=' + "
                "__import__('json').dumps({'all_pass': True, 'latency_us_geomean': value}))\n",
                encoding="utf-8",
            )
            request = {
                "schema_version": 1,
                "schedule": verification_schedule(2),
                "manifests": {
                    "incumbent": {"value.txt": "snapshots/incumbent/0000.bin"},
                    "candidate": {"value.txt": "snapshots/candidate/0000.bin"},
                },
                "command": ["python3", "fake_eval.py"],
                "run_timeout_seconds": 10,
            }
            request_path = request_dir / "request.json"
            result_path = request_dir / "result.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            previous = Path.cwd()
            try:
                __import__("os").chdir(root)
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(run_remote(request_path, result_path), 0)
            finally:
                __import__("os").chdir(previous)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(_payload_from_stdout(stdout.getvalue()), payload)
            latencies = [item["result"]["latency_us_geomean"] for item in payload["runs"]]
            self.assertEqual(latencies, [10.0, 8.0, 8.0, 10.0])


class GatewayCommandTests(unittest.TestCase):
    def test_abba_driver_uses_existing_evaluator_payload_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "campaign"
            base = init_repo(workspace)
            CampaignStore(workspace)
            (workspace / "kernel.py").write_text("VALUE = 8\n", encoding="utf-8")
            run_git(workspace, "add", "kernel.py")
            run_git(workspace, "commit", "-m", "candidate")
            candidate = git_head(workspace)
            captured: dict = {}
            payload = {
                "schema_version": 1,
                "error": None,
                "runs": [row("incumbent", 0, 10.0), row("candidate", 0, 8.0)],
            }

            def fake_run(
                workspace_arg,
                hardware,
                profile,
                url,
                timeout,
                command,
                **kwargs,
            ):
                captured.update(
                    workspace=workspace_arg,
                    hardware=hardware,
                    profile=profile,
                    url=url,
                    timeout=timeout,
                    command=command,
                    kwargs=kwargs,
                )
                stdout = ABBA_RESULT_PREFIX + json.dumps(payload) + "\n"
                return __import__("subprocess").CompletedProcess(command, 0, stdout, "")

            validator = GatewayABBAValidator(
                hardware="REMOTE_GPU", timeout=300, repeats=1, per_run_timeout=100
            )
            with mock.patch("long_horizon.main_adapter.run_sandbox", side_effect=fake_run):
                result = validator.verify(
                    workspace,
                    base_commit=base,
                    candidate_commit=candidate,
                    changed_paths=["kernel.py"],
                )
            self.assertTrue(result.passed)
            self.assertEqual(captured["hardware"], "REMOTE_GPU")
            self.assertEqual(captured["kwargs"]["gateway_kind"], "dev")
            self.assertEqual(captured["kwargs"]["sync"], ())
            self.assertNotIn("--input", captured["command"])
            remote_command = captured["command"][1]
            self.assertTrue(remote_command.endswith("/test_kernel.py"))
            self.assertIn("aggregate_kernels/.atrex_long_horizon_verify", remote_command)
            artifact = Path(result.artifact)
            self.assertTrue(artifact.is_file())
            self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), payload)
            self.assertEqual(artifact.name, "result.json")
            self.assertFalse(result.artifact.startswith("remote:"))
            self.assertEqual(run_git(workspace, "status", "--porcelain"), "")

    def test_current_sandbox_dry_run_packages_abba_driver_as_evaluator_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            verify_dir = workspace / "aggregate_kernels/.atrex_long_horizon_verify/test"
            verify_dir.mkdir(parents=True)
            repo_root = Path(__file__).resolve().parents[2]
            (workspace / "tools").symlink_to(repo_root / "tools", target_is_directory=True)
            for name, content in {
                "kernel.py": "def run(): pass\n",
                "test_kernel.py": "print('evaluator')\n",
                "reference.py": "def reference(): pass\n",
                "definition.json": "{}\n",
                "workload.jsonl": "{}\n",
                "solution.json": '{"sources": ["kernel.py"]}\n',
            }.items():
                (workspace / name).write_text(content, encoding="utf-8")
            (verify_dir / "test_kernel.py").write_text("print('driver')\n", encoding="utf-8")
            (verify_dir / "request.json").write_text("{}\n", encoding="utf-8")
            result_relative = "aggregate_kernels/.atrex_long_horizon_verify/test/result.json"
            command = [
                "python3", str(repo_root / "tools/sandbox.py"),
                "--dry-run", "--kind", "dev", "--hardware", "REMOTE_GPU",
                "--workspace", str(workspace), "--timeout", "600",
                "--no-sync",
                "--", "python3",
                "aggregate_kernels/.atrex_long_horizon_verify/test/test_kernel.py",
                "aggregate_kernels/.atrex_long_horizon_verify/test/request.json",
                result_relative,
            ]
            process = __import__("subprocess").run(
                command, capture_output=True, text=True, check=True
            )
            payload = json.loads(process.stdout)
            self.assertEqual(payload["kind"], "dev")
            self.assertGreaterEqual(payload["files"], 8)
            self.assertIn(".atrex_long_horizon_verify/test/test_kernel.py", payload["command"])


if __name__ == "__main__":
    unittest.main()
