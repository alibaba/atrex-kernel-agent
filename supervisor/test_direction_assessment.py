"""Match Runtime closure/evidence semantics while retaining AKA's Gateway ID list API."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supervisor import gateway
from supervisor.errors import AgentRequestError
from supervisor.journal import SupervisorJournalService, initialize_journal, load_journal


class DirectionAssessmentTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "kernel.py").write_text("def run(x): return x\n")
        self.service = self.service_for(2)

    def service_for(self, episode: int) -> SupervisorJournalService:
        evidence = self.root / ".atrex_long_horizon/episodes" / f"e{episode:04d}/supervisor_runtime"
        initialize_journal(
            evidence / "journal.json", episode=episode, base_commit="a" * 40, branch="test"
        )
        return SupervisorJournalService(
            workspace=self.root, campaign_root=self.root, evidence_root=evidence
        )

    def propose(self, *, start: bool = True) -> str:
        direction = self.service.execute(
            {
                "operation": "direction_update",
                "request": {
                    "action": "propose",
                    "name": "fusion",
                    "hypothesis": "fusion is faster",
                    "rationale": "avoid intermediate writes",
                    "plan": ["fuse and measure"],
                    "success_criteria": ["correct and faster"],
                    "stop_conditions": ["no progress"],
                },
            }
        )["direction_id"]
        if start:
            self.update(direction, "start")
        return direction

    def update(self, direction: str, action: str, **fields: object) -> dict:
        return self.service.execute(
            {
                "operation": "direction_update",
                "request": {
                    "action": action,
                    "direction_id": direction,
                    "analysis": "investigation update",
                    **fields,
                },
            }
        )

    def record(
        self, *, kind: str = "check", status: str = "succeeded", result: dict | None = None
    ) -> str:
        with patch.dict(
            os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.service.evidence_root)}
        ):
            return gateway._record_episode_evaluation(
                self.root,
                result if result is not None else {"passed": False},
                gateway_kind=kind,
                private_result={"status": status},
            )["record_id"]

    def experiment(
        self, direction: str, records: list[str], *, action: str = "abandon_direction"
    ) -> str:
        return self.service.execute(
            {
                "operation": "experiment_record",
                "request": {
                    "direction_id": direction,
                    "name": "diagnostic",
                    "hypothesis": "fusion compiles",
                    "change": "checked the implementation",
                    "gateway_record_ids": records,
                    "evidence": "the referenced observation",
                    "analysis": "interpret the result",
                    "action": action,
                },
            }
        )["experiment_id"]

    def view(self, direction: str) -> dict:
        return self.service.execute({"operation": "direction_load", "direction_id": direction})

    def close(
        self,
        direction: str,
        experiments: list[str],
        action: str = "defer",
        assessment: str = "unresolved",
    ) -> dict:
        return self.update(
            direction, action, hypothesis_status=assessment, supporting_experiment_ids=experiments
        )

    def test_every_closure_needs_explicit_same_direction_evidence_without_partial_writes(
        self,
    ) -> None:
        direction = self.propose()
        unrelated = self.propose(start=False)
        # A completed, unrelated historical experiment must not establish this Direction.
        value = load_journal(self.service.path)
        foreign = "experiment_" + "e" * 32
        value["experiments"].append(
            {
                "experiment_id": foreign,
                "direction_id": unrelated,
                "gateway_record_ids": [self.record()],
            }
        )
        self.service.path.write_text(json.dumps(value))
        own = self.experiment(direction, [self.record()])
        for action in ("complete", "abandon", "block", "defer"):
            for fields in (
                {},
                {"hypothesis_status": "unresolved"},
                {"hypothesis_status": "unresolved", "supporting_experiment_ids": []},
                {"hypothesis_status": "unresolved", "supporting_experiment_ids": [own, own]},
                {"hypothesis_status": "unresolved", "supporting_experiment_ids": [foreign]},
                {
                    "hypothesis_status": "unresolved",
                    "supporting_experiment_ids": ["experiment_" + "f" * 32],
                },
                {"hypothesis_status": "PASS", "supporting_experiment_ids": [own]},
                {"hypothesis_status": "unresolved", "supporting_experiment_ids": "not an array"},
                {"hypothesis_status": "unresolved", "supporting_experiment_ids": [own] * 33},
            ):
                before = self.service.path.read_bytes()
                with self.subTest(action=action, fields=fields), self.assertRaises(ValueError):
                    self.update(direction, action, **fields)
                self.assertEqual(self.service.path.read_bytes(), before)
            self.close(direction, [own], action)
            self.assertEqual(self.view(direction)["supporting_experiment_ids"], [own])
            self.update(direction, "start")
        self.assertEqual(self.view(direction)["hypothesis_status"], "unresolved")

    def test_execution_status_not_correctness_controls_whether_evidence_is_complete(self) -> None:
        direction = self.propose()
        for status in ("succeeded", "completed", "failed", "cancelled", "running"):
            experiment = self.experiment(direction, [self.record(status=status)])
            for assessment in ("unresolved", "supported", "refuted"):
                with self.subTest(status=status, assessment=assessment):
                    before = self.service.path.read_bytes()
                    if status not in {"succeeded", "completed"} and assessment != "unresolved":
                        with self.assertRaises(AgentRequestError) as rejected:
                            self.close(direction, [experiment], assessment=assessment)
                        self.assertEqual(
                            rejected.exception.response["error"]["code"],
                            "direction_assessment_requires_completed_result",
                        )
                        self.assertIn(
                            "unresolved", rejected.exception.response["error"]["next_action"]
                        )
                        self.assertEqual(self.service.path.read_bytes(), before)
                        continue
                    self.close(direction, [experiment], assessment=assessment)
                    self.assertEqual(self.view(direction)["hypothesis_status"], assessment)
                    self.update(direction, "start")
                    loaded = self.view(direction)
                    self.assertEqual(loaded["hypothesis_status"], "unresolved")
                    self.assertEqual(loaded["supporting_experiment_ids"], [])
                    self.assertIn(experiment, loaded["associated_experiment_ids"])

    def test_every_selected_experiment_needs_completed_evidence_but_not_every_cited_record(
        self,
    ) -> None:
        direction = self.propose()
        failed = self.record(status="failed")
        good = self.experiment(direction, [failed, self.record()])
        bad = self.experiment(direction, [failed])
        with self.assertRaisesRegex(ValueError, "every selected Experiment"):
            self.close(direction, [good, bad], assessment="supported")
        self.close(direction, [good], assessment="supported")

    def test_every_experiment_needs_bound_evidence_even_for_abandon(self) -> None:
        direction = self.propose()
        for action in ("baseline", "keep_after", "restore_before", "adopt", "abandon_direction"):
            before = self.service.path.read_bytes()
            with self.subTest(action=action), self.assertRaises(AgentRequestError) as rejected:
                self.experiment(direction, [], action=action)
            self.assertEqual(
                rejected.exception.response["error"]["code"], "experiment_evidence_required"
            )
            self.assertIn("never invent", rejected.exception.response["error"]["next_action"])
            self.assertEqual(self.service.path.read_bytes(), before)
        for kind in ("env", "health", "wiki", "dev"):
            record = self.record(kind=kind)
            path = self.service.evidence_root / "gateway-records" / record / "result.json"
            if kind == "dev":
                value = json.loads(path.read_bytes())
                value.pop("kernel_artifact_digest")
                path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "real Kernel"):
                self.experiment(direction, [record])

    def test_all_gpu_record_types_can_support_an_assessment(self) -> None:
        direction = self.propose()
        for kind in ("run", "same_allocation_abba", "profile", "dev", "check", "disassemble"):
            with self.subTest(kind=kind):
                experiment = self.experiment(direction, [self.record(kind=kind, result={})])
                self.close(direction, [experiment], assessment="refuted")
                self.update(direction, "start")

    def test_late_experiments_do_not_change_closure_or_interfere_with_another_direction(
        self,
    ) -> None:
        first = self.propose()
        original = self.experiment(first, [self.record()])
        for action, status in (
            ("complete", "completed"),
            ("abandon", "abandoned"),
            ("block", "blocked"),
            ("defer", "deferred"),
        ):
            self.close(first, [original], action, "supported")
            second = self.propose(start=False)
            # Only start the second on the last pass, to remain inside the three-Direction limit.
            if action == "defer":
                self.update(second, "start")
            before = load_journal(self.service.path)["direction_events"]
            late = self.experiment(first, [self.record(kind="profile")])
            self.assertEqual(load_journal(self.service.path)["direction_events"], before)
            view = self.view(first)
            self.assertEqual(view["status"], status)
            self.assertEqual(view["hypothesis_status"], "supported")
            self.assertEqual(view["supporting_experiment_ids"], [original])
            self.assertIn(late, view["associated_experiment_ids"])
            self.assertEqual(
                self.view(second)["status"], "in_progress" if action == "defer" else "proposed"
            )
            if action != "defer":
                with self.assertRaisesRegex(ValueError, "proposed"):
                    self.experiment(second, [self.record()])
            else:
                with self.assertRaises(AgentRequestError):
                    self.update(first, "start")

    def test_legacy_history_is_readable_and_new_support_survives_report_and_memory(self) -> None:
        self.service = self.service_for(1)
        direction = self.propose()
        evidence = self.experiment(direction, [self.record()])
        unmeasured = "experiment_" + "b" * 32
        legacy = load_journal(self.service.path)
        legacy["experiments"].append(
            {**legacy["experiments"][0], "experiment_id": unmeasured, "gateway_record_ids": []}
        )
        legacy["direction_events"].append(
            {
                "direction_event_id": "directionevent_" + "a" * 32,
                "direction_id": direction,
                "action": "defer",
                "analysis": "old automatic support",
                "supporting_experiment_ids": [evidence, unmeasured],
            }
        )
        self.service.path.write_text(json.dumps(legacy))
        frozen = self.service.path.read_bytes()
        old_path = self.service.path
        self.service = self.service_for(2)
        loaded = self.view(direction)
        self.assertEqual(loaded["hypothesis_status"], "unresolved")
        self.assertEqual(loaded["supporting_experiment_ids"], [])
        self.assertEqual(loaded["associated_experiment_ids"], [evidence, unmeasured])
        self.assertEqual(
            self.service.execute({"operation": "experiment_load", "experiment_id": unmeasured})[
                "gateway_record_ids"
            ],
            [],
        )
        self.update(direction, "start")
        with self.assertRaisesRegex(ValueError, "historical unmeasured notes"):
            self.close(direction, [unmeasured])
        self.close(direction, [evidence], assessment="refuted")
        self.service.execute({"operation": "directions_list", "file": "scratch/directions.json"})
        listed = json.loads((self.root / "scratch/directions.json").read_bytes())["directions"][0]
        self.assertEqual(listed["hypothesis_status"], "refuted")
        self.service.execute(
            {
                "operation": "episode_report",
                "request": {"status": "pivot", "summary": "historical evidence reinterpreted"},
            }
        )
        current = load_journal(self.service.path)
        from long_horizon.campaign import _memory_experience

        self.assertEqual(
            _memory_experience(current)["direction_events"][-1]["supporting_experiment_ids"],
            [evidence],
        )
        self.assertEqual(
            _memory_experience(current)["direction_events"][-1]["hypothesis_status"], "refuted"
        )
        self.assertEqual(old_path.read_bytes(), frozen)
        with self.assertRaises(AgentRequestError) as finalized:
            self.experiment(direction, [self.record()])
        self.assertFalse(finalized.exception.response["repairable"])

    def test_future_gateway_records_are_not_visible_to_current_episode(self) -> None:
        current = self.service
        self.service = self.service_for(3)
        future = self.record()
        self.service = current
        direction = self.propose()
        with self.assertRaisesRegex(ValueError, "not visible"):
            self.experiment(direction, [future])

    def test_no_result_means_no_direction_closure_or_terminal_bypass(self) -> None:
        direction = self.propose()
        for action in ("complete", "abandon", "block", "defer"):
            with self.assertRaises(AgentRequestError):
                self.close(direction, [], action)
        for status in ("pivot", "blocked"):
            request = {"status": status, "summary": "no result"}
            if status == "blocked":
                request["blocker"] = "transport failed before a record existed"
            before = self.service.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "record its real Gateway evidence"):
                self.service.execute({"operation": "episode_report", "request": request})
            self.assertEqual(self.service.path.read_bytes(), before)

    def test_legacy_completed_payloads_and_raw_status_remain_usable_without_rewriting(self) -> None:
        direction = self.propose()
        for kind, result in (
            ("run", {"all_pass": False}),
            ("profile", {"kernels": []}),
            ("check", {"passed": False}),
            ("dev", {"exit_code": 1}),
        ):
            record = self.record(kind=kind, result=result)
            directory = self.service.evidence_root / "gateway-records" / record
            path = directory / "result.json"
            value = json.loads(path.read_bytes())
            value.pop("execution_status")
            path.write_text(json.dumps(value))
            if kind != "profile":
                (directory / "raw-result.json").unlink()
            frozen = path.read_bytes()
            experiment = self.experiment(direction, [record])
            self.close(direction, [experiment], assessment="refuted")
            self.update(direction, "start")
            self.assertEqual(path.read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
