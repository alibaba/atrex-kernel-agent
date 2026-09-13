from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from long_horizon.verifier import _verification_shape_batches
from orchestrator.campaign import Campaign
from orchestrator.workspace_runtime import link_runtime
from reference import sol_seed

from supervisor import gateway
from supervisor.runner_assets import PROFILE_DRIVER, RUNNERS_ROOT, evaluator_path
from supervisor.runners import profile_entry


class PrivateRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for name, content in {
            "kernel.py": "class Model: pass\n",
            "reference.py": "class Model: pass\n",
            "input.py": "def _make_inputs(**kwargs): return {}\n",
            "shapes.json": json.dumps(
                {str(i): {"init_kwargs": {}, "input_kwargs": {}} for i in range(9)}
            ),
        }.items():
            (self.workspace / name).write_text(content)

    def _fallback(self, *arguments: str) -> tuple[dict, dict[str, bytes]]:
        bundles: list[dict[str, bytes]] = []
        original = gateway._make_input_bundle

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result[0]))) as archive:
                bundles.append(
                    {
                        item.name: archive.extractfile(item).read()
                        for item in archive
                        if item.isfile()
                    }
                )
            return result

        output = io.StringIO()
        with (
            patch.object(gateway, "_run_typed_gateway", return_value=None),
            patch.object(gateway, "_find_agate", return_value=None),
            patch.object(gateway, "_private_atrex_bench_runtime", return_value=None),
            patch.object(gateway, "_make_input_bundle", side_effect=capture),
            patch("sys.stdout", output),
            patch("sys.stderr", io.StringIO()),
        ):
            status = gateway._main(
                [
                    "--workspace",
                    str(self.workspace),
                    "--hardware",
                    "L20N",
                    "--url",
                    "https://gateway.example.test",
                    "--dry-run",
                    *arguments,
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(len(bundles), 1)
        return json.loads(output.getvalue()), bundles[0]

    def test_linked_workspace_has_no_execution_drivers(self) -> None:
        link_runtime(self.workspace)
        for name in ("test_kernel.py", "profile_driver.py", "atrex_bench_test_kernel.py"):
            self.assertFalse((self.workspace / name).exists())
            self.assertFalse((self.workspace / "reference" / name).exists())
        self.assertTrue(PROFILE_DRIVER.is_file())
        self.assertTrue(evaluator_path(self.workspace).is_file())

    def test_profile_cli_does_not_receive_evaluate_only_flags(self) -> None:
        for kind in ("run", "profile"):
            with self.subTest(kind=kind):
                args = gateway.build_parser().parse_args([
                    "--workspace", str(self.workspace), "--hardware", "L20N", "--kind", kind,
                ])
                request = gateway._typed_request(self.workspace, "L20N", 120, [], [], kind)
                command = gateway._typed_agate_command(
                    "agate", args, self.workspace, kind, request, 0,
                )
                if kind == "run":
                    self.assertIn("--mode", command)
                    self.assertIn("warmup_iters=5", command)
                else:
                    for option in ("--mode", "--set", "--num-gpus"):
                        self.assertNotIn(option, command)
                    self.assertIn("--level", command)

    def test_missing_job_response_reports_cli_failure_without_private_shape_leak(self) -> None:
        args = gateway.build_parser().parse_args([
            "--workspace", str(self.workspace), "--hardware", "L20N", "--kind", "profile",
        ])
        request = gateway._typed_request(self.workspace, "L20N", 120, [], [], "profile")
        process = subprocess.CompletedProcess(
            ["agate", "profile"], 2, "", "agate: unrecognized arguments: --mode full; private-case-123",
        )
        for generalized in (False, True):
            with (
                self.subTest(generalized=generalized),
                patch.object(gateway, "_is_generalized_workspace", return_value=generalized),
                patch.object(gateway, "_typed_request", return_value=request),
                patch.object(gateway, "_execute_typed_processes", return_value=[process]),
                patch.object(gateway, "_find_agate", return_value="agate"),
                patch("sys.stderr", io.StringIO()) as stderr,
            ):
                status = gateway._run_typed_gateway(args, self.workspace, [], "profile", [], 0)
            self.assertEqual(status, 2)
            result = json.loads(stderr.getvalue())
            self.assertEqual(result["error"]["code"], "gateway_client_error")
            self.assertFalse(result["repairable"])
            self.assertEqual("private-case-123" in result["error"]["message"], not generalized)
            self.assertIn("operator", result["error"]["next_action"])

    def test_typed_run_accepts_top_level_options_without_driver(self) -> None:
        requests = []

        def typed(args, workspace, command, kind, *_):
            requests.append(
                gateway._typed_request(workspace, args.hardware, args.timeout, [], command, kind)
            )
            return 0

        with patch.object(gateway, "_run_typed_gateway", side_effect=typed):
            status = gateway._main(
                [
                    "--workspace",
                    str(self.workspace),
                    "--hardware",
                    "L20N",
                    "--kind",
                    "run",
                    "--version",
                    "v3",
                    "--shape-id",
                    "2",
                    "--multi-seed",
                    "5",
                    "--timed-runs",
                    "1",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(set(requests[0]["reference"]["shapes"]), {"2"})
        self.assertEqual(requests[0]["options"]["bench_iters"], 1)
        self.assertEqual(requests[0]["options"]["num_correctness_cases"], 6)
        self.assertFalse((self.workspace / "test_kernel.py").exists())

    def test_profile_has_no_default_download_or_local_directories(self) -> None:
        sync = []

        def typed(args, workspace, command, kind, sync_paths, *_):
            sync.append(sync_paths)
            gateway._record_profile_job(
                {"status": "succeeded", "result": {}}, workspace, sync_paths,
                {"kernels": [{"name": "identity"}]},
            )
            return 0

        with patch.object(gateway, "_run_typed_gateway", side_effect=typed):
            result = gateway._main([
                "--workspace", str(self.workspace), "--hardware", "L20N", "--kind", "profile",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(sync, [[]])
        self.assertFalse((self.workspace / "profiles").exists())
        self.assertFalse((self.workspace / "scratch").exists())

    def test_profile_download_is_opt_in_under_scratch(self) -> None:
        public = {"kernels": [{"name": "identity"}]}
        gateway._record_profile_job(
            {"status": "succeeded", "result": {}}, self.workspace,
            ["scratch/profile-1"], public,
        )
        stored = json.loads((self.workspace / "scratch/profile-1/gateway_profile.json").read_text())
        self.assertEqual(stored, {"status": "succeeded", "result": public})
        self.assertFalse((self.workspace / "profiles").exists())

    def test_fallback_profile_uses_remote_scratch_by_default(self) -> None:
        args = gateway.build_parser().parse_args(["--kind", "profile"])
        command = gateway._profile_fallback_command(args, [])
        self.assertEqual(
            command[:4], ["python3", "profile_entry.py", "--output-dir", "scratch/profile"]
        )

    def test_native_fallback_injects_private_evaluator(self) -> None:
        result, files = self._fallback("--kind", "run", "--version", "v3", "--no-sync")
        self.assertIn("python3 test_kernel.py", result["command"])
        self.assertEqual(files["test_kernel.py"], evaluator_path(self.workspace).read_bytes())
        self.assertNotIn("profile_driver.py", files)
        self.assertFalse((self.workspace / "test_kernel.py").exists())

    def test_workspace_cannot_shadow_private_evaluator(self) -> None:
        (self.workspace / "test_kernel.py").write_text("raise RuntimeError('forged')")
        _, files = self._fallback("--kind", "run", "--no-sync")
        self.assertEqual(files["test_kernel.py"], evaluator_path(self.workspace).read_bytes())

    def test_profile_fallback_injects_driver_and_preserves_options(self) -> None:
        result, files = self._fallback(
            "--kind",
            "profile",
            "--sync",
            "profiles/v3",
            "--profile-source",
            "--kernel-name",
            "vector_add",
            "--launch-skip",
            "2",
            "--launch-count",
            "4",
        )
        self.assertIn("profile_entry.py", result["command"])
        self.assertIn("--kernel-name vector_add", result["command"])
        self.assertIn("--launch-skip 2 --launch-count 4 --source", result["command"])
        self.assertEqual(files["profile_driver.py"], PROFILE_DRIVER.read_bytes())
        self.assertEqual(
            files["profile_entry.py"], (RUNNERS_ROOT / "profile_entry.py").read_bytes()
        )
        self.assertIn("tools/profile_nvidia.sh", files)
        self.assertIn("tools/profile_kernel.sh", files)
        self.assertNotIn("test_kernel.py", files)
        self.assertFalse((self.workspace / "profile_driver.py").exists())

    def test_private_profile_injects_only_selected_case(self) -> None:
        private = self.root / "reference"
        private.mkdir()
        (private / "shapes.json").write_bytes((self.workspace / "shapes.json").read_bytes())
        (private / "metadata.json").write_text("{}")
        (self.workspace / "shapes.json").unlink()
        (self.workspace / "agent_problem.json").write_text("{}")
        from orchestrator.optimization_policy import install_workspace_policy

        install_workspace_policy(self.workspace, "production", "Triton")
        with patch.dict(os.environ, {gateway.PRIVATE_REFERENCE_ENV: str(private)}):
            _, files = self._fallback("--kind", "profile", "--profile-shape-id", "3", "--no-sync")
        self.assertNotIn("shapes.json", files)
        self.assertNotIn("metadata.json", files)
        self.assertEqual(json.loads(files[gateway.PRIVATE_PROFILE_CASE_FILENAME])["shape_id"], "3")

    def test_abba_staging_and_batching_do_not_require_agent_driver(self) -> None:
        target = self.root / "staged"
        target.mkdir()
        gateway._copy_evaluation_workspace(self.workspace, target)
        self.assertEqual(
            (target / "test_kernel.py").read_bytes(), evaluator_path(self.workspace).read_bytes()
        )
        batches, ids = _verification_shape_batches(self.workspace, None, 4)
        self.assertEqual(batches, [["0", "1", "2", "3"], ["4", "5", "6", "7"], ["8"]])
        self.assertEqual(len(ids), 9)
        self.assertFalse((self.workspace / "test_kernel.py").exists())

    def test_sol_fallback_selects_private_sol_runner_and_config(self) -> None:
        (self.workspace / "workload.jsonl").write_text("{}\n")
        (self.workspace / "config.json").write_text('{"seed":200}')
        _, files = self._fallback("--kind", "run", "--no-sync")
        self.assertEqual(
            files["test_kernel.py"], (RUNNERS_ROOT / "sol_test_kernel.py").read_bytes()
        )
        self.assertEqual(json.loads(files["config.json"]), {"seed": 200})
        self.assertEqual(_verification_shape_batches(self.workspace, None, 4), ([None], []))

    def test_arbitrary_dev_does_not_receive_private_runners(self) -> None:
        _, files = self._fallback(
            "--kind", "dev", "--no-sync", "--", "python3", "-c", "print('ok')"
        )
        self.assertEqual(files, {})

    def test_framework_smoke_prompt_has_only_public_cli(self) -> None:
        campaign = object.__new__(Campaign)
        campaign.sandbox_hardware = "L20N"
        campaign.sandbox_ssh = ""
        campaign.sandbox_url = ""
        campaign.sandbox_profile = ""
        campaign.atrex_bench_root = self.root
        with patch.object(campaign, "_framework_baseline_smoke_shape_ids", return_value=["1", "3"]):
            command, _ = campaign._framework_baseline_smoke_command(1)
        self.assertIn("python3 tools/sandbox.py --kind run", command)
        self.assertIn("--shape-id 1 --shape-id 3 --timed-runs 1", command)
        self.assertNotIn("test_kernel", command)
        self.assertNotIn("--no-memory", command)

    def test_sol_bootstrap_does_not_create_or_commit_drivers(self) -> None:
        operator = self.root / "operator"
        operator.mkdir()
        (operator / "definition.json").write_text(
            json.dumps(
                {
                    "name": "identity",
                    "inputs": {"x": {}},
                    "outputs": {"out": {}},
                }
            )
        )
        (operator / "reference.py").write_text("def run(x): return x\n")
        (operator / "workload.jsonl").write_text("{}\n")
        with (
            patch.object(
                sol_seed.subprocess, "run", wraps=subprocess.run
            ) as run,
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(
                sol_seed.main(
                    [
                        "--op-dir",
                        str(operator),
                        "--name",
                        "identity",
                        "--workspace",
                        str(self.workspace),
                        "--no-bench",
                    ]
                ),
                0,
            )
        self.assertTrue((self.workspace / "kernel.py").is_file())
        for name in ("test_kernel.py", "profile_driver.py", ".gitignore"):
            self.assertFalse((self.workspace / name).exists())
            self.assertTrue(all(name not in call.args[0] for call in run.call_args_list))

    def test_remote_profile_entry_maps_nvidia_and_amd_options(self) -> None:
        cases = [
            (
                [
                    "--profiler",
                    "ncu",
                    "--kernel-regex",
                    "^fused$",
                    "--launch-skip",
                    "2",
                    "--launch-count",
                    "3",
                    "--source",
                ],
                [
                    "bash",
                    "tools/profile_nvidia.sh",
                    "profile_driver.py",
                    "--output-dir",
                    "profiles/v3",
                    "--kernel-name",
                    "regex:^fused$",
                    "--launch-skip",
                    "2",
                    "--launch-count",
                    "3",
                    "--source",
                ],
            ),
            (
                [
                    "--profiler",
                    "rocprofv3",
                    "--kernel-name",
                    "fused(x)",
                    "--launch-skip",
                    "2",
                    "--launch-count",
                    "3",
                ],
                [
                    "bash",
                    "tools/profile_kernel.sh",
                    "profile_driver.py",
                    "--output-dir",
                    "profiles/v3",
                    "--kernel-regex",
                    r"^fused\(x\)$",
                    "--iteration-range",
                    "[2,3,4]",
                ],
            ),
        ]
        for arguments, expected in cases:
            with (
                self.subTest(arguments=arguments),
                patch.object(
                    sys, "argv", ["profile_entry.py", "--output-dir", "profiles/v3", *arguments]
                ),
                patch.object(profile_entry.subprocess, "call", return_value=7) as call,
            ):
                self.assertEqual(profile_entry.main(), 7)
                call.assert_called_once_with(expected)

    def test_remote_profile_entry_auto_detects_gpu_worker_profiler(self) -> None:
        with (
            patch.object(sys, "argv", ["profile_entry.py", "--output-dir", "profiles/v3"]),
            patch.object(
                profile_entry.shutil,
                "which",
                side_effect=lambda name: name if name == "rocprofv3" else None,
            ),
            patch.object(profile_entry.subprocess, "call", return_value=0) as call,
        ):
            self.assertEqual(profile_entry.main(), 0)
        self.assertEqual(call.call_args.args[0][1], "tools/profile_kernel.sh")


if __name__ == "__main__":
    unittest.main()
