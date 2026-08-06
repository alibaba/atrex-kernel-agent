from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from teacher_distill.evidence import build_evidence_bundle


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


class EvidenceBuilderTest(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Path, Path]:
        workspace = root / "candidate"
        private = root / "private"
        for directory in (
            workspace / "memory",
            workspace / "plans",
            workspace / "profiles/v1",
            private / "teacher_workspace/aggregate_kernels/.atrex_teacher_verify/run1",
            private / "teacher_workspace/aggregate_kernels/.atrex_teacher_verify/run2",
            workspace / ".atrex_long_horizon/episodes/e0001/archive",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (workspace / "kernel.py").write_text("# v0\n", encoding="utf-8")
        _git(workspace, "init", "-q")
        _git(workspace, "config", "user.email", "test@local")
        _git(workspace, "config", "user.name", "test")
        _git(workspace, "add", "kernel.py")
        _git(workspace, "commit", "-q", "-m", "v0")
        v0_commit = _git(workspace, "rev-parse", "HEAD")

        (workspace / "kernel.py").write_text("# v1 faster\n", encoding="utf-8")
        _git(workspace, "add", "kernel.py")
        _git(workspace, "commit", "-q", "-m", "v1")
        v1_commit = _git(workspace, "rev-parse", "HEAD")

        records = {
            0: {
                "version": "v0",
                "masked": False,
                "performance": {"latency_us": 200.0},
                "quality_gate": {"result": "PASS"},
                "correctness": {"status": "PASS"},
                "optimization": {"action_category": "baseline"},
                "git_commit_hash": v0_commit,
            },
            1: {
                "version": "v1",
                "masked": False,
                "performance": {
                    "latency_us": 150.0,
                    "latency_us_by_shape": {"a": 140.0, "b": 160.0},
                },
                "quality_gate": {"result": "PASS"},
                "correctness": {"status": "PASS"},
                "optimization": {"action_category": "vectorized_load"},
                "teacher_progress": {
                    "candidate_to_teacher_geomean_ratio": 1.20,
                    "worst_shape_ratio": 1.25,
                    "worst_shape_key": "a",
                    "abba_status": "NOT_RUN",
                },
                "git_commit_hash": v1_commit,
            },
            2: {
                "version": "v2",
                "masked": False,
                "performance": {"latency_us": 170.0},
                "quality_gate": {"result": "FAIL"},
                "correctness": {"status": "PASS"},
                "optimization": {"action_category": "double_buffering"},
                "git_commit_hash": None,
            },
            3: {
                "version": "v3",
                "masked": True,
                "performance": {"latency_us": 145.0},
                "quality_gate": {"result": "PASS"},
                "correctness": {"status": "PASS"},
                "optimization": {"action_category": "swizzle"},
                "git_commit_hash": v1_commit,
            },
        }
        for version, record in records.items():
            (workspace / "memory" / f"v{version}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        (workspace / "plans/v1_plan.md").write_text("# V1 plan\n", encoding="utf-8")
        (workspace / "profiles/v1/REPORT.md").write_text("# V1 profile\n", encoding="utf-8")
        abba = private / "teacher_workspace/aggregate_kernels/.atrex_teacher_verify/run1/result.json"
        abba.write_text(
            json.dumps({"schema_version": 1, "runs": [], "error": None}),
            encoding="utf-8",
        )
        verified_abba = private / "teacher_workspace/aggregate_kernels/.atrex_teacher_verify/run2/result.json"
        verified_abba.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification_status": "PASS",
                    "candidate_to_teacher_ratio": 1.03,
                    "payload": {"schema_version": 1, "runs": [], "error": None},
                }
            ),
            encoding="utf-8",
        )
        (workspace / ".atrex_long_horizon/episodes/e0001/archive/candidate.patch").write_text(
            "private checkpoint patch\n", encoding="utf-8"
        )
        return workspace, private

    def test_builder_is_deterministic_and_classifies_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-evidence-") as temp_dir:
            workspace, private = self._workspace(Path(temp_dir))
            first = build_evidence_bundle(workspace, private)
            manifest_first = first.manifest_path.read_bytes()
            trajectory_first = first.trajectory_path.read_bytes()
            second = build_evidence_bundle(workspace, private)
            manifest_second = second.manifest_path.read_bytes()
            trajectory_second = second.trajectory_path.read_bytes()
            manifest = json.loads(manifest_second)
            trajectory = json.loads(trajectory_second)

        self.assertEqual(manifest_first, manifest_second)
        self.assertEqual(trajectory_first, trajectory_second)
        by_id = {entry["evidence_id"]: entry for entry in manifest["evidence"]}
        self.assertEqual(by_id["E-V1-MEMORY"]["classification"], "accepted")
        self.assertTrue(by_id["E-V1-MEMORY"]["citable_as_verified"])
        self.assertEqual(by_id["E-V2-MEMORY"]["classification"], "reverted")
        self.assertFalse(by_id["E-V2-MEMORY"]["citable_as_verified"])
        self.assertEqual(by_id["E-V3-MEMORY"]["classification"], "masked")
        self.assertFalse(by_id["E-V3-MEMORY"]["citable_as_verified"])
        self.assertIn("E-V1-PLAN", by_id)
        self.assertIn("E-V1-PROFILE", by_id)
        abba_rows = [entry for key, entry in by_id.items() if key.startswith("E-ABBA-")]
        self.assertEqual(len(abba_rows), 2)
        self.assertEqual(
            sorted(row["citable_as_verified"] for row in abba_rows),
            [False, True],
        )
        checkpoint_rows = [
            entry for key, entry in by_id.items() if key.startswith("E-EPISODE-CHECKPOINT-")
        ]
        self.assertTrue(checkpoint_rows)
        self.assertTrue(all(not row["citable_as_verified"] for row in checkpoint_rows))
        self.assertEqual([row["version"] for row in trajectory["versions"]], ["v0", "v1", "v2", "v3"])
        self.assertEqual(trajectory["versions"][1]["latency_us"], 150.0)
        self.assertEqual(trajectory["versions"][1]["teacher_ratio"], 1.20)

    def test_every_performance_row_references_its_memory_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-evidence-links-") as temp_dir:
            workspace, private = self._workspace(Path(temp_dir))
            bundle = build_evidence_bundle(workspace, private)
            trajectory = json.loads(bundle.trajectory_path.read_text(encoding="utf-8"))

        for row in trajectory["versions"]:
            self.assertEqual(row["evidence_id"], f"E-{row['version'].upper()}-MEMORY")

    def test_schema_is_valid_json(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "teacher_distill/schemas/evidence.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
