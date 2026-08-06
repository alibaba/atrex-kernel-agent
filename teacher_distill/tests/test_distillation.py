from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from teacher_distill.distillation import generate_distillation
from teacher_distill.models import CampaignTerminalStatus, TeacherCampaignResult
from teacher_distill.state import PRIVATE_STATE_FILE


class FakeAgentRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, workspace: Path, prompt: str) -> None:
        self.calls.append((workspace, prompt))
        if "TEACHER_GAP_ANALYSIS" in prompt:
            self.assert_gap_inputs(workspace)
            (workspace / "teacher_gap_analysis.md").write_text(
                "# Gap hypotheses\n\nPossible pipeline difference.\n", encoding="utf-8"
            )
            (workspace / "teacher_gap_analysis.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "hypothesis",
                        "promotion_eligible": False,
                        "findings": [
                            {
                                "claim": "Possible pipeline difference",
                                "status": "hypothesis",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        else:
            self.assert_distill_has_no_teacher_source(workspace)
            (workspace / "optimization_cards").mkdir(exist_ok=True)
            (workspace / "journey.md").write_text(
                "# Journey\n\nV1 improved latency. Evidence: [E-V1-MEMORY]\n",
                encoding="utf-8",
            )
            (workspace / "pitfalls.md").write_text(
                "# Pitfalls\n\nA reverted attempt. Evidence: [E-V2-MEMORY]\n",
                encoding="utf-8",
            )
            (workspace / "optimization_cards/vectorized-load.md").write_text(
                "# Vectorized load\n\n"
                "Architecture: sm90\n\nFramework: CuteDSL\n\nScope: tested workloads\n\n"
                "Verified improvement. Evidence: [E-V1-MEMORY]\n",
                encoding="utf-8",
            )
            (workspace / "promotion_checklist.md").write_text(
                "# Promotion checklist\n", encoding="utf-8"
            )
            (workspace / "draft_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "evidence_level": "single-campaign",
                        "documents": [
                            "journey.md",
                            "pitfalls.md",
                            "optimization_cards/vectorized-load.md",
                            "promotion_checklist.md",
                        ],
                    }
                ),
                encoding="utf-8",
            )

    @staticmethod
    def assert_gap_inputs(workspace: Path) -> None:
        assert (workspace / "teacher/kernel.py").is_file()
        assert (workspace / "candidate/kernel.py").is_file()

    @staticmethod
    def assert_distill_has_no_teacher_source(workspace: Path) -> None:
        assert not (workspace / "teacher").exists()
        for path in workspace.rglob("*"):
            if path.is_file():
                assert "UNIQUE_TEACHER_SOURCE" not in path.read_text(
                    encoding="utf-8", errors="ignore"
                )


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


class DistillationGenerationTest(unittest.TestCase):
    def _state(self, root: Path) -> tuple[Path, Path, str]:
        candidate = root / "candidate"
        private = root / "private"
        teacher = root / "teacher-bundle"
        for path in (
            candidate / "memory",
            candidate / "plans",
            candidate / "profiles/v1",
            private,
            teacher,
        ):
            path.mkdir(parents=True, exist_ok=True)
        teacher_source = "# UNIQUE_TEACHER_SOURCE\ndef run(): pass\n"
        (teacher / "kernel.py").write_text(teacher_source, encoding="utf-8")
        (teacher / "solution.json").write_text(
            json.dumps({"sources": [{"path": "kernel.py"}]}), encoding="utf-8"
        )
        (candidate / "kernel.py").write_text("# final candidate\ndef run(): pass\n", encoding="utf-8")
        (candidate / "solution.json").write_text(
            json.dumps({"sources": [{"path": "kernel.py"}]}), encoding="utf-8"
        )
        _git(candidate, "init", "-q")
        _git(candidate, "config", "user.email", "test@local")
        _git(candidate, "config", "user.name", "test")
        _git(candidate, "add", "kernel.py", "solution.json")
        _git(candidate, "commit", "-q", "-m", "candidate")
        commit = _git(candidate, "rev-parse", "HEAD")
        for version, latency, gate, masked in (
            (0, 200.0, "PASS", False),
            (1, 150.0, "PASS", False),
            (2, 170.0, "FAIL", False),
        ):
            (candidate / "memory" / f"v{version}.json").write_text(
                json.dumps(
                    {
                        "version": f"v{version}",
                        "masked": masked,
                        "performance": {"latency_us": latency},
                        "correctness": {"status": "PASS"},
                        "quality_gate": {"result": gate},
                        "optimization": {"action_category": "baseline" if version == 0 else "test"},
                        "git_commit_hash": commit if gate == "PASS" else None,
                    }
                ),
                encoding="utf-8",
            )
        (candidate / "plans/v1_plan.md").write_text("# plan\n", encoding="utf-8")
        (candidate / "profiles/v1/REPORT.md").write_text("# profile\n", encoding="utf-8")
        (private / PRIVATE_STATE_FILE).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "campaign_01234567",
                    "bundle_path": str(teacher),
                }
            ),
            encoding="utf-8",
        )
        return candidate, private, teacher_source

    @staticmethod
    def _result(status: CampaignTerminalStatus) -> TeacherCampaignResult:
        return TeacherCampaignResult(
            schema_version=1,
            campaign_id="campaign_01234567",
            status=status,
            reason=status.value,
            final_version="v2",
            final_candidate_to_teacher_ratio=1.03,
        )

    def test_success_runs_gap_then_distillation_and_archives_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-distill-generate-") as temp_dir:
            candidate, private, teacher_source = self._state(Path(temp_dir))
            runner = FakeAgentRunner()
            artifacts = generate_distillation(
                candidate,
                private,
                self._result(CampaignTerminalStatus.SUCCESS),
                agent_runner=runner,
            )

            gap = json.loads(artifacts.gap_json.read_text(encoding="utf-8"))
            archived = (artifacts.root / "reference_kernel/kernel.py").read_text(encoding="utf-8")
            draft_files = [path.relative_to(artifacts.root).as_posix() for path in artifacts.root.rglob("*.md")]
            agent_workspaces = [workspace.resolve() for workspace, _prompt in runner.calls]
            private_root = private.resolve()
            candidate_root = candidate.resolve()
            workspaces_were_isolated = all(
                private_root != workspace
                and private_root not in workspace.parents
                and candidate_root != workspace
                and candidate_root not in workspace.parents
                for workspace in agent_workspaces
            )
            workspaces_were_cleaned = all(not workspace.exists() for workspace in agent_workspaces)

        self.assertEqual(len(runner.calls), 2)
        self.assertTrue(workspaces_were_isolated)
        self.assertTrue(workspaces_were_cleaned)
        self.assertEqual(gap["status"], "hypothesis")
        self.assertFalse(gap["promotion_eligible"])
        self.assertIn("final candidate", archived)
        self.assertNotIn(teacher_source, archived)
        self.assertIn("journey.md", draft_files)
        self.assertFalse(artifacts.audit_only)

    def test_plateau_generates_drafts_but_not_reference_kernel(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-distill-plateau-") as temp_dir:
            candidate, private, _teacher_source = self._state(Path(temp_dir))
            runner = FakeAgentRunner()
            artifacts = generate_distillation(
                candidate,
                private,
                self._result(CampaignTerminalStatus.PLATEAU),
                agent_runner=runner,
            )
            has_reference = (artifacts.root / "reference_kernel").exists()

        self.assertEqual(len(runner.calls), 2)
        self.assertFalse(has_reference)

    def test_leakage_violation_is_audit_only_and_runs_no_agent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-distill-leakage-") as temp_dir:
            candidate, private, _teacher_source = self._state(Path(temp_dir))
            runner = mock.Mock()
            artifacts = generate_distillation(
                candidate,
                private,
                self._result(CampaignTerminalStatus.TEACHER_LEAKAGE_VIOLATION),
                agent_runner=runner,
            )
            audit_file_exists = (artifacts.root / "AUDIT_ONLY.md").is_file()

        runner.assert_not_called()
        self.assertTrue(artifacts.audit_only)
        self.assertTrue(audit_file_exists)


if __name__ == "__main__":
    unittest.main()
