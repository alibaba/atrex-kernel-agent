from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools import memory_manager


class TeacherMemoryTest(unittest.TestCase):
    def _write(self, workspace: Path, version: str, data: dict) -> None:
        memory = workspace / "memory"
        memory.mkdir(parents=True, exist_ok=True)
        (memory / f"{version}.json").write_text(
            json.dumps({"version": version, "masked": False, **data}),
            encoding="utf-8",
        )

    def _summary(self, workspace: Path) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            memory_manager.cmd_summary(argparse.Namespace(workspace=str(workspace)))
        return output.getvalue()

    def test_create_initializes_optional_teacher_progress_to_null(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-memory-create-") as temp_dir:
            workspace = Path(temp_dir)
            memory_manager.cmd_create(
                argparse.Namespace(workspace=str(workspace), version="v0", force=False)
            )
            created = json.loads((workspace / "memory/v0.json").read_text(encoding="utf-8"))

        self.assertIn("teacher_progress", created)
        self.assertIsNone(created["teacher_progress"])

    def test_standard_summary_remains_on_the_existing_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-memory-standard-") as temp_dir:
            workspace = Path(temp_dir)
            self._write(
                workspace,
                "v0",
                {
                    "performance": {"tflops": 10.0, "bandwidth_gbps": 20.0},
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                    "optimization": {"action_category": "baseline"},
                },
            )
            summary = self._summary(workspace)

        header = summary.splitlines()[0]
        self.assertIn("TFLOPS", header)
        self.assertIn("BW(GB/s)", header)
        self.assertNotIn("Teacher", header)
        self.assertNotIn("ABBA", header)

    def test_teacher_summary_adds_ratio_worst_shape_and_abba_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-memory-progress-") as temp_dir:
            workspace = Path(temp_dir)
            self._write(
                workspace,
                "v1",
                {
                    "performance": {"latency_us": 118.0},
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                    "optimization": {"action_category": "vectorized_load"},
                    "teacher_progress": {
                        "target_id": "teacher_01234567",
                        "candidate_to_teacher_geomean_ratio": 1.18,
                        "worst_shape_ratio": 1.27,
                        "worst_shape_key": "shape_4",
                        "geomean_gate_met": False,
                        "shape_gate_met": False,
                        "provisional_target_met": False,
                        "abba_status": "NOT_RUN",
                        "final_candidate_to_teacher_ratio": None,
                    },
                },
            )
            summary = self._summary(workspace)

        header, _separator, row = summary.splitlines()
        self.assertIn("Teacher×", header)
        self.assertIn("Worst×", header)
        self.assertIn("WorstShape", header)
        self.assertIn("ABBA", header)
        self.assertIn("1.180", row)
        self.assertIn("1.270", row)
        self.assertIn("shape_4", row)
        self.assertIn("NOT_RUN", row)

    def test_mixed_history_uses_placeholders_for_non_teacher_versions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-memory-mixed-") as temp_dir:
            workspace = Path(temp_dir)
            self._write(workspace, "v0", {"performance": {}, "teacher_progress": None})
            self._write(
                workspace,
                "v1",
                {
                    "performance": {},
                    "teacher_progress": {
                        "candidate_to_teacher_geomean_ratio": 1.05,
                        "worst_shape_ratio": 1.08,
                        "worst_shape_key": "0",
                        "abba_status": "PASS",
                    },
                },
            )
            lines = self._summary(workspace).splitlines()

        self.assertIn("v0", lines[2])
        self.assertIn("-", lines[2])
        self.assertIn("PASS", lines[3])


if __name__ == "__main__":
    unittest.main()
