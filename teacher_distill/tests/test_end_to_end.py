from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from teacher_distill.tests import test_campaign as campaign_fixtures


class TeacherDistillCliEndToEndTest(unittest.TestCase):
    def test_cli_reaches_real_supervisor_and_writes_public_and_private_state(self) -> None:
        helper = campaign_fixtures.TeacherCampaignTest()
        helper.setUp()
        with tempfile.TemporaryDirectory(prefix="teacher-e2e-") as temp_dir:
            root = Path(temp_dir)
            request = helper._request(root)
            patches = helper._patch_dependencies(root)
            op = {
                "name": request.name,
                "reference": str(request.kernel_demo),
                "roofline_py": "",
                "op_dir": str(request.op_dir),
                "atrex_bench_root": "",
            }
            argv = [
                "--campaign-mode",
                "teacher-distill",
                "--op-dir",
                str(request.op_dir),
                "--teacher-solution",
                str(request.teacher_solution),
                "--platform",
                request.platform,
                "--arch",
                request.architecture,
                "--framework",
                request.framework,
                "--no-workload-bucketing",
                "--sandbox-hardware",
                request.sandbox_hardware,
                "--sandbox-url",
                request.sandbox_url,
                "--agent-cli",
                request.agent_cli,
                "--workspace",
                str(request.workspace_root),
                "--teacher-private-root",
                str(request.private_root),
                "--max-iters",
                "10",
                "--max-stall",
                "5",
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/pi"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                exit_code = optimize.main(argv)

            candidate = campaign_fixtures.FakeCandidate.instances[-1]
            private_results = list(request.private_root.glob("campaign_*/result.json"))
            public_target_exists = (candidate.workspace / "teacher_target.json").is_file()
            public_lock_exists = (candidate.workspace / "campaign_lock.json").is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(public_target_exists)
        self.assertTrue(public_lock_exists)
        self.assertEqual(len(private_results), 1)


if __name__ == "__main__":
    unittest.main()
