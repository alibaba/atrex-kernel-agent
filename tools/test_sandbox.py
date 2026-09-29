"""Regression tests for the sandbox's typed agate commands.

Run with: python3 -m unittest tools.test_sandbox
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import sandbox


class AgateProfileTests(unittest.TestCase):
    def test_profile_cli_submits_without_evaluation_flags_and_saves_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            for filename, content in {
                "kernel.py": "class Model: pass\n",
                "reference.py": "class Model: pass\n",
                "input.py": "def _make_inputs(): return {}\n",
                "shapes.json": '{"0":{"init_kwargs":{},"input_kwargs":{}}}',
                "metadata.json": "{}",
            }.items():
                (workspace / filename).write_text(content)
            for level in ("sol", "deep"):
                with self.subTest(level=level):
                    job = {
                        "job_id": "pf_fixture",
                        "status": "succeeded",
                        "result": {"profiler": "ncu", "level": level, "kernels": []},
                    }
                    response = subprocess.CompletedProcess([], 0, json.dumps(job), "")
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch.object(sandbox, "_find_agate", return_value="/fixture/agate"),
                        patch.object(sandbox, "_run_agate_with_cancel_retry",
                                     return_value=response) as submit,
                        redirect_stdout(io.StringIO()),
                        redirect_stderr(io.StringIO()),
                    ):
                        status = sandbox.main([
                            "--kind", "profile", "--hardware", "L20N",
                            "--workspace", str(workspace), "--profiler", "ncu",
                            "--profile-level", level, "--kernel-regex", "fixture.*",
                            "--sync", "profiles", "--", "bash",
                            "tools/profile_nvidia.sh", "driver.py",
                        ])
                    self.assertEqual(status, 0)
                    submit.assert_called_once()
                    command = submit.call_args.kwargs["agate"]
                    self.assertEqual(command[:2], ["/fixture/agate", "profile"])
                    self.assertNotIn("--mode", command)
                    self.assertNotIn("--set", command)
                    self.assertEqual(command[command.index("--level") + 1], level)
                    self.assertEqual(command[command.index("--profiler") + 1], "ncu")
                    self.assertEqual(command[command.index("--kernel-regex") + 1], "fixture.*")
                    report = workspace / "profiles/gateway_profile.json"
                    self.assertEqual(json.loads(report.read_text())["job_id"], "pf_fixture")

    def test_evaluation_options_are_only_forwarded_to_run(self):
        args = SimpleNamespace(
            url="", gateway_profile=None, hardware="L20N", timeout=180,
            env=[], profile_level="sol", profiler="ncu", profile_counter=[],
            kernel_regex=None, top_kernels=None,
        )
        for mode in ("full", "correctness_only"):
            for tolerance in (None, 0.2):
                for kind in ("run", "profile"):
                    with self.subTest(mode=mode, tolerance=tolerance, kind=kind):
                        request = {
                            "mode": mode,
                            "options": {
                                "num_correctness_cases": 2, "bench_iters": 5,
                                "correctness_max_rel_l2": tolerance,
                            },
                            "reference": {"operator": "fixture"},
                        }
                        command = sandbox._typed_agate_command(
                            "agate", args, Path("/fixture/workspace"), kind,
                            request, 60, reference_dir=Path("/fixture/reference"),
                        )
                        settings = [command[i + 1] for i, flag in enumerate(command)
                                    if flag == "--set"]
                        if kind == "run":
                            self.assertEqual(command[command.index("--mode") + 1], mode)
                            expected = ["warmup_iters=5"]
                            if tolerance is not None:
                                expected.append("correctness_max_rel_l2=0.2")
                            self.assertEqual(settings, expected)
                        else:
                            self.assertNotIn("--mode", command)
                            self.assertEqual(settings, [])


if __name__ == "__main__":
    unittest.main()
