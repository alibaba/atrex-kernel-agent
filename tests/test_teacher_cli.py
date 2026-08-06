from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize


class TeacherCliTest(unittest.TestCase):
    def _op(self) -> dict[str, str]:
        return {
            "name": "gdn",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
            "atrex_bench_root": "",
        }

    def _base(self, workspace: str, teacher: str) -> list[str]:
        return [
            "--op-dir",
            "/tmp/op",
            "--platform",
            "H20",
            "--arch",
            "sm_90",
            "--framework",
            "CuteDSL",
            "--campaign-mode",
            "teacher-distill",
            "--teacher-solution",
            teacher,
            "--no-workload-bucketing",
            "--workspace",
            workspace,
        ]

    def _run_valid(self, argv: list[str]):
        with (
            mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
            mock.patch.object(optimize, "_resolve_op", return_value=self._op()),
            mock.patch.object(optimize, "ensure_submodules"),
            mock.patch("teacher_distill.cli.run_teacher_distill", return_value=0) as run,
        ):
            result = optimize.main(argv)
        self.assertEqual(result, 0)
        return run.call_args.args[0]

    def test_valid_teacher_mode_builds_a_production_single_workspace_request(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-cli-") as temp_dir:
            teacher = Path(temp_dir) / "teacher"
            teacher.mkdir()
            request = self._run_valid(self._base(temp_dir, str(teacher)))

        self.assertEqual(request.campaign_mode, "teacher-distill")
        self.assertEqual(request.optimization_mode, "production")
        self.assertEqual(request.framework, "CuteDSL")
        self.assertEqual(request.architecture, "sm_90")
        self.assertEqual(request.teacher_solution, teacher.resolve())
        self.assertEqual(request.geomean_ratio, 1.05)
        self.assertEqual(request.shape_ratio, 1.10)
        self.assertEqual(request.framework_baseline, "always")
        self.assertEqual(request.convert_after, 0)
        self.assertEqual(request.max_stall, 5)
        self.assertFalse(request.workload_bucketing)
        self.assertFalse(request.layer)

    def test_explicit_production_and_custom_teacher_controls_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-cli-options-") as temp_dir:
            teacher = Path(temp_dir) / "teacher"
            teacher.mkdir()
            argv = [
                *self._base(temp_dir, str(teacher)),
                "--optimization-mode",
                "production",
                "--teacher-geomean-ratio",
                "1.02",
                "--teacher-shape-ratio",
                "1.07",
                "--teacher-stall-before-episode",
                "4",
                "--teacher-partial-restarts",
                "2",
                "--teacher-private-root",
                str(Path(temp_dir) / "private"),
            ]
            request = self._run_valid(argv)

        self.assertEqual(request.geomean_ratio, 1.02)
        self.assertEqual(request.shape_ratio, 1.07)
        self.assertEqual(request.stall_before_episode, 4)
        self.assertEqual(request.partial_restarts, 2)
        self.assertEqual(request.private_root, (Path(temp_dir) / "private").resolve())

    def test_teacher_mode_rejects_unsupported_combinations_before_setup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-cli-invalid-") as temp_dir:
            teacher = Path(temp_dir) / "teacher"
            teacher.mkdir()
            base = self._base(temp_dir, str(teacher))
            cases = {
                "missing framework": [
                    value
                    for index, value in enumerate(base)
                    if index not in {base.index("--framework"), base.index("--framework") + 1}
                ],
                "explicit leaderboard": [*base, "--optimization-mode", "leaderboard"],
                "layer": [*base, "--layer"],
                "bucketing": [value for value in base if value != "--no-workload-bucketing"],
                "framework baseline never": [*base, "--framework-baseline", "never"],
            }
            for label, argv in cases.items():
                with self.subTest(label=label):
                    with (
                        mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                        mock.patch.object(optimize, "detect_arch") as detect,
                        mock.patch.object(optimize, "ensure_submodules") as submodules,
                    ):
                        with self.assertRaises(SystemExit):
                            optimize.main(argv)
                    detect.assert_not_called()
                    submodules.assert_not_called()

    def test_teacher_mode_rejects_invalid_ratios_counts_and_missing_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-cli-values-") as temp_dir:
            teacher = Path(temp_dir) / "teacher"
            teacher.mkdir()
            base = self._base(temp_dir, str(teacher))
            cases = (
                [*base, "--teacher-geomean-ratio", "0.99"],
                [*base, "--teacher-shape-ratio", str(math.inf)],
                [*base, "--teacher-stall-before-episode", "0"],
                [*base, "--teacher-partial-restarts", "-1"],
                self._base(temp_dir, str(Path(temp_dir) / "missing")),
            )
            for argv in cases:
                with self.subTest(argv=argv[-2:]):
                    with (
                        mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                        mock.patch.object(optimize, "detect_arch") as detect,
                        mock.patch.object(optimize, "ensure_submodules") as submodules,
                    ):
                        with self.assertRaises(SystemExit):
                            optimize.main(argv)
                    detect.assert_not_called()
                    submodules.assert_not_called()

    def test_teacher_options_are_rejected_in_standard_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-cli-standard-") as temp_dir:
            teacher = Path(temp_dir) / "teacher"
            teacher.mkdir()
            argv = [
                "--op-dir",
                "/tmp/op",
                "--platform",
                "H20",
                "--teacher-solution",
                str(teacher),
            ]
            with self.assertRaises(SystemExit):
                optimize.main(argv)


if __name__ == "__main__":
    unittest.main()
