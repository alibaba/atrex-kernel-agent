from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize


def _killed(exit_status: int = 1, timed_out: bool = False) -> optimize.SessionResult:
    return optimize.SessionResult(
        exit_status=exit_status,
        timed_out=timed_out,
        tokens=1234,
        stdout_tail="Let me check the profile",
        stderr_tail="API Error: Request rejected (429) Throttling",
        session_id="11111111-2222-3333-4444-555555555555",
    )


def _campaign(work_dir: Path, **kwargs) -> optimize.Campaign:
    campaign = optimize.Campaign(
        name="flash_attention",
        kernel_demo=str(work_dir / "reference.py"),
        platform="pro5000",
        framework="CuteDSL",
        work_dir=str(work_dir),
        **kwargs,
    )
    campaign.workspace.mkdir(parents=True, exist_ok=True)
    return campaign


class InterruptedIterationRecordTest(unittest.TestCase):
    def test_timeout_and_crash_get_distinct_correctness_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-record-") as temp_dir:
            workspace = Path(temp_dir)
            timeout_record = json.loads(optimize.record_interrupted_iteration(
                workspace, 7, kind="iter", exit_status=-9, timed_out=True,
                timeout_s=5400, stderr_tail="",
            ).read_text(encoding="utf-8"))
            crash_record = json.loads(optimize.record_interrupted_iteration(
                workspace, 8, kind="iter", exit_status=1, timed_out=False,
                timeout_s=5400, stderr_tail="429 Throttling",
            ).read_text(encoding="utf-8"))

        self.assertEqual(timeout_record["correctness"]["status"], "TIMEOUT_FAIL")
        self.assertIn("5400s", timeout_record["quality_gate"]["failure_reason"])
        self.assertEqual(crash_record["correctness"]["status"], "FAIL")
        self.assertIn("429 Throttling", crash_record["quality_gate"]["failure_reason"])
        for record in (timeout_record, crash_record):
            self.assertEqual(record["quality_gate"]["result"], "FAIL")
            self.assertIsNone(record["git_commit_hash"])
            self.assertFalse(record["masked"])
            self.assertEqual(record["optimization"]["action_category"], optimize.INTERRUPTED_CATEGORY)
            self.assertEqual(record["pitfalls_and_fixes"][-1]["error_type"], "infra")

    def test_partial_record_written_by_the_killed_session_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-merge-") as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v3.json").write_text(json.dumps({
                "version": "v3",
                "masked": False,
                "performance": {"latency_us": 4321.0, "latency_us_by_shape": {"2": 4321.0}},
                "profile_evidence": {"bottleneck_type": "latency_stall_bound"},
                "pitfalls_and_fixes": [{"error_type": "performance", "error_message": "2 CTAs regressed"}],
            }), encoding="utf-8")

            record = json.loads(optimize.record_interrupted_iteration(
                workspace, 3, kind="iter", exit_status=1, timed_out=False,
                timeout_s=5400, stderr_tail="",
            ).read_text(encoding="utf-8"))

        self.assertEqual(record["performance"]["latency_us"], 4321.0)
        self.assertEqual(record["profile_evidence"]["bottleneck_type"], "latency_stall_bound")
        self.assertEqual(len(record["pitfalls_and_fixes"]), 2)
        self.assertEqual(record["pitfalls_and_fixes"][0]["error_message"], "2 CTAs regressed")
        self.assertEqual(record["quality_gate"]["result"], "FAIL")


class EnsureIterationMemoryTest(unittest.TestCase):
    def test_mechanical_record_lands_when_the_salvage_session_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-fallback-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            with mock.patch.object(optimize, "run_session", return_value=_killed()) as run:
                campaign._ensure_iteration_memory(5, _killed(), "iter")
            self.assertEqual(run.call_count, 1)
            record = optimize.read_memory(campaign.workspace, 5)

        self.assertIsNotNone(record)
        self.assertEqual(record["optimization"]["action_category"], optimize.INTERRUPTED_CATEGORY)

    def test_a_salvage_written_record_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-agent-") as temp_dir:
            campaign = _campaign(Path(temp_dir))

            def write_record(workspace, prompt, **kwargs):
                (workspace / "memory").mkdir(parents=True, exist_ok=True)
                (workspace / "memory" / "v5.json").write_text(json.dumps({
                    "version": "v5",
                    "optimization": {"action_category": "smem_tiling"},
                    "open_directions": [{"direction": "clamp the paged tail read", "rationale": "OOB NaN"}],
                }), encoding="utf-8")
                return _killed(exit_status=0)

            with mock.patch.object(optimize, "run_session", side_effect=write_record):
                campaign._ensure_iteration_memory(5, _killed(), "iter")
            record = optimize.read_memory(campaign.workspace, 5)

        self.assertEqual(record["optimization"]["action_category"], "smem_tiling")
        self.assertEqual(record["open_directions"][0]["direction"], "clamp the paged tail read")

    def test_record_with_findings_skips_the_salvage_session_entirely(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-skip-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            (campaign.workspace / "memory").mkdir(parents=True)
            (campaign.workspace / "memory" / "v5.json").write_text(
                json.dumps({"version": "v5", "quality_gate": {"result": "FAIL"}}), encoding="utf-8"
            )
            with mock.patch.object(optimize, "run_session") as run:
                campaign._ensure_iteration_memory(5, _killed(), "iter")
            run.assert_not_called()

    def test_null_template_left_by_a_killed_session_still_triggers_recovery(self) -> None:
        template = {
            "version": "v5",
            "masked": False,
            "performance": {"latency_us": None, "tflops": None},
            "optimization": {"action_category": None, "action_description": None},
            "correctness": {"rel_err": None, "status": None},
            "quality_gate": {"result": None, "failure_reason": None},
            "pitfalls_and_fixes": [],
            "search_log": [],
            "open_directions": [],
        }
        with tempfile.TemporaryDirectory(prefix="salvage-template-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            (campaign.workspace / "memory").mkdir(parents=True, exist_ok=True)
            (campaign.workspace / "memory" / "v5.json").write_text(
                json.dumps(template), encoding="utf-8"
            )
            with mock.patch.object(optimize, "run_session", return_value=_killed()) as run:
                campaign._ensure_iteration_memory(5, _killed(timed_out=True, exit_status=-9), "iter")
            self.assertEqual(run.call_count, 1)  # the salvage session must be given a chance
            record = optimize.read_memory(campaign.workspace, 5)

        self.assertEqual(record["correctness"]["status"], "TIMEOUT_FAIL")
        self.assertEqual(record["optimization"]["action_category"], optimize.INTERRUPTED_CATEGORY)

    def test_empty_record_predicate_distinguishes_findings_from_a_template(self) -> None:
        self.assertTrue(optimize.memory_record_is_empty(None))
        self.assertTrue(optimize.memory_record_is_empty({"version": "v5"}))
        self.assertTrue(optimize.memory_record_is_empty({
            "quality_gate": {"result": None}, "correctness": {"status": None},
            "performance": {"latency_us": None}, "pitfalls_and_fixes": [],
        }))
        self.assertFalse(optimize.memory_record_is_empty({"quality_gate": {"result": "FAIL"}}))
        self.assertFalse(optimize.memory_record_is_empty({"performance": {"latency_us": 12.5}}))
        self.assertFalse(optimize.memory_record_is_empty({
            "pitfalls_and_fixes": [{"error_type": "infra", "error_message": "x"}]
        }))

    def test_zero_salvage_timeout_records_mechanically_without_a_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="salvage-disabled-") as temp_dir:
            campaign = _campaign(Path(temp_dir), salvage_timeout=0)
            with mock.patch.object(optimize, "run_session") as run:
                campaign._ensure_iteration_memory(9, _killed(timed_out=True, exit_status=-9), "iter")
            run.assert_not_called()
            record = optimize.read_memory(campaign.workspace, 9)

        self.assertEqual(record["correctness"]["status"], "TIMEOUT_FAIL")


class SalvagePromptTest(unittest.TestCase):
    def test_prompt_renders_without_leftover_placeholders(self) -> None:
        rendered = optimize._render(
            optimize.PROMPTS_DIR / "salvage.md",
            WORKSPACE="/tmp/ws", N=5, PREV=4, KIND="iter",
            KILL_REASON="it hit the 5400s hang backstop and was killed",
            TRANSCRIPT="/root/.claude/projects/x/y.jsonl",
            STDOUT_TAIL="Let me check",
            MODE_POLICY="## Optimization mode: production",
        )
        self.assertNotIn("{{", rendered)
        self.assertTrue(rendered.startswith("## Optimization mode: production"))
        self.assertIn("memory/v5.json", rendered)

    def test_transcript_is_resolved_only_for_session_persisting_clis(self) -> None:
        self.assertIsNone(optimize.session_transcript_path("codex", "abc"))
        self.assertIsNone(optimize.session_transcript_path("claude", ""))


if __name__ == "__main__":
    unittest.main()
