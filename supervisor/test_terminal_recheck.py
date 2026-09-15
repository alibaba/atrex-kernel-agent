"""Submission and recovery must reject the same broken evidence and Direction histories."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import EpisodeWorktree
from long_horizon.journal import validate_terminal
from long_horizon.models import EpisodeHandoff
from supervisor.errors import RuntimeStateError
from supervisor.journal import SupervisorJournalService, initialize_journal, load_journal
from supervisor.test_journal import _record_kernel, _write_json


class TerminalRecheckTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.source = b"def run(x): return x\n"
        (self.workspace / "kernel.py").write_bytes(self.source)
        self.service = self.new_episode(1)
        _, self.gateway_id = _record_kernel(self.service.evidence_root, self.source)
        self.direction_id = self.propose(self.service)
        self.start(self.service, self.direction_id)
        self.experiment_id = self.record(self.service, self.direction_id)
        self.close(self.service, self.direction_id, self.experiment_id)
        self.report(self.service, self.experiment_id)
        self.original = load_journal(self.service.path)

    def new_episode(self, episode: int) -> SupervisorJournalService:
        evidence = self.workspace / f"private-{episode}"
        initialize_journal(evidence / "journal.json", episode=episode, base_commit="a" * 40, branch=f"episode-{episode}")
        return SupervisorJournalService(workspace=self.workspace, campaign_root=self.workspace, evidence_root=evidence)

    def propose(self, service: SupervisorJournalService) -> str:
        return service.execute({"operation": "direction_update", "request": {
            "action": "propose", "name": "fusion", "hypothesis": "fewer launches",
            "rationale": "overhead", "plan": ["measure"], "success_criteria": ["faster"],
            "stop_conditions": ["no gain"],
        }})["direction_id"]

    def start(self, service: SupervisorJournalService, direction_id: str) -> None:
        service.execute({"operation": "direction_update", "request": {
            "action": "start", "direction_id": direction_id, "analysis": "explore",
        }})

    def record(self, service: SupervisorJournalService, direction_id: str) -> str:
        return service.execute({"operation": "experiment_record", "request": {
            "direction_id": direction_id, "name": "measurement", "hypothesis": "faster",
            "change": "fusion", "gateway_record_ids": [self.gateway_id],
            "evidence": "passed", "analysis": "useful", "action": "keep_after",
        }})["experiment_id"]

    def close(self, service: SupervisorJournalService, direction_id: str, experiment_id: str) -> None:
        service.execute({"operation": "direction_update", "request": {
            "action": "complete", "direction_id": direction_id, "analysis": "measured",
            "hypothesis_status": "supported", "supporting_experiment_ids": [experiment_id],
        }})

    def report(self, service: SupervisorJournalService, selected_id: str) -> None:
        with patch("long_horizon.git_episode.EpisodeWorktree.commit_candidate", return_value="b" * 40):
            self.assertEqual(service.execute({"operation": "episode_report", "request": {
                "status": "candidate_ready", "summary": "ready", "selected_experiment_id": selected_id,
            }})["status"], "accepted")

    def recheck(self, service: SupervisorJournalService | None = None) -> str:
        service = service or self.service
        value = load_journal(service.path)
        before = service.path.read_bytes()
        result = validate_terminal(
            service.path, expected_episode=value["episode"], base_commit=value["base_commit"],
            branch=value["episode_branch"], state="candidate_ready", candidate_commit="b" * 40,
            campaign_root=self.workspace, workspace=self.workspace,
        )
        self.assertEqual(service.path.read_bytes(), before)
        return result

    def assert_rejected_by_both(self, *, history: bool = False) -> None:
        self.assertTrue(self.recheck())
        handoff = self.workspace / ".atrex_long_horizon/handoff.json"
        before = self.service.path.read_bytes(), handoff.read_bytes()
        with patch("long_horizon.git_episode.EpisodeWorktree.commit_candidate") as commit:
            with self.assertRaises(RuntimeStateError if history else (ValueError, RuntimeStateError)):
                self.service.execute({"operation": "episode_report", "request": {
                    "status": "candidate_ready", "summary": "ready",
                    "selected_experiment_id": self.experiment_id,
                }})
        commit.assert_not_called()
        self.assertEqual((self.service.path.read_bytes(), handoff.read_bytes()), before)

    def test_selected_experiment_must_keep_valid_gateway_references(self) -> None:
        for references in (None, [], "not a list", ["invalid"], ["gateway-" + "f" * 32], [self.gateway_id] * 2):
            with self.subTest(references=references):
                value = copy.deepcopy(self.original)
                if references is None:
                    value["experiments"][0].pop("gateway_record_ids")
                else:
                    value["experiments"][0]["gateway_record_ids"] = references
                _write_json(self.service.path, value)
                self.assert_rejected_by_both()
        _write_json(self.service.path, self.original)
        self.assertEqual(self.recheck(), "")

    def test_recheck_requires_completed_passing_evaluate_for_current_kernel(self) -> None:
        record_path = self.service.evidence_root / "gateway-records" / self.gateway_id / "result.json"
        original = json.loads(record_path.read_bytes())
        for change in (
            {"execution_status": "cancelled"}, {"execution_status": "running"},
            {"error": {"error_class": "infra"}}, {"gateway_kind": "profile"},
            {"gateway_kind": "env"}, {"kernel_artifact_digest": "sha256:" + "f" * 64},
            {"result": {"all_pass": False}},
            {"execution_status": "completed", "result": {"all_pass": True, "error": "failed"}},
            {"gateway_kind": []}, {"execution_status": []},
        ):
            with self.subTest(change=change):
                _write_json(record_path, {**original, **change})
                self.assert_rejected_by_both()
        _write_json(record_path, original)
        record_path.unlink()
        self.assert_rejected_by_both()
        _write_json(record_path, original)
        (self.workspace / "kernel.py").write_bytes(self.source + b"# unmeasured\n")
        self.assert_rejected_by_both()

    def test_selected_identity_action_and_direction_are_rechecked(self) -> None:
        cases = []
        for field, replacement in (("action", "restore_before"), ("action", []), ("direction_id", "direction_" + "f" * 32)):
            value = copy.deepcopy(self.original)
            value["experiments"][0][field] = replacement
            cases.append(value)
        duplicate = copy.deepcopy(self.original)
        duplicate["experiments"] *= 2
        cases.append(duplicate)
        for value in cases:
            with self.subTest(value=value):
                _write_json(self.service.path, value)
                self.assert_rejected_by_both()
        value = copy.deepcopy(self.original)
        value["outcome"].pop("selected_experiment_id")
        _write_json(self.service.path, value)
        self.assertIn("selected_experiment_id", self.recheck())

    def test_impossible_direction_transitions_are_not_just_folded_to_the_last_status(self) -> None:
        events = self.original["direction_events"]
        for order in ((0, 2), (2,), (1, 2), (0, 0, 1, 2), (0, 1, 1, 2), (0, 1)):
            with self.subTest(order=order):
                value = copy.deepcopy(self.original)
                value["direction_events"] = [copy.deepcopy(events[index]) for index in order]
                _write_json(self.service.path, value)
                self.assert_rejected_by_both(history=order != (0, 1))
        for bad in (None, {"direction_id": "bad", "action": "complete"}, {"direction_id": self.direction_id, "action": []}):
            with self.subTest(event=bad):
                value = copy.deepcopy(self.original)
                value["direction_events"] = [bad]
                _write_json(self.service.path, value)
                self.assert_rejected_by_both(history=True)

    def test_overlapping_exploration_and_four_directions_are_rejected_even_if_all_closed(self) -> None:
        def events_for(number: int) -> list[dict]:
            return [{**event, "direction_id": f"direction_{number:032x}"} for event in self.original["direction_events"]]

        first, second = events_for(1), events_for(2)
        for events in (
            [first[0], second[0], first[1], second[1], first[2], second[2]],
            [event for number in range(1, 5) for event in events_for(number)],
        ):
            value = copy.deepcopy(self.original)
            value["direction_events"] = events
            _write_json(self.service.path, value)
            self.assert_rejected_by_both(history=True)

    def archive_first_episode(self) -> None:
        archive = self.workspace / ".atrex_long_horizon/episodes/e0001/supervisor_runtime"
        shutil.copytree(self.service.evidence_root, archive)

    def test_historical_closed_direction_can_be_reassessed_or_restarted(self) -> None:
        self.archive_first_episode()
        for restart in (False, True):
            with self.subTest(restart=restart):
                service = self.new_episode(2)
                if restart:
                    self.start(service, self.direction_id)
                # Late evidence and reassessment need not reopen a historical closed Direction.
                experiment_id = self.record(service, self.direction_id)
                self.close(service, self.direction_id, experiment_id)
                self.report(service, experiment_id)
                self.assertEqual(self.recheck(service), "")
                self.assertNotIn("propose", [event["action"] for event in load_journal(service.path)["direction_events"]])

    def test_direction_budget_resets_each_episode_but_is_not_reset_by_closure(self) -> None:
        self.archive_first_episode()
        service = self.new_episode(2)
        for _ in range(3):
            direction = self.propose(service)
            self.start(service, direction)
            experiment = self.record(service, direction)
            self.close(service, direction, experiment)
        fourth = self.propose(service)
        before = service.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "at most three"):
            self.start(service, fourth)
        self.assertEqual(service.path.read_bytes(), before)
        self.report(service, experiment)
        self.assertEqual(self.recheck(service), "")

    def test_historical_transitions_are_validated_and_future_records_are_not_visible(self) -> None:
        self.archive_first_episode()
        current = self.new_episode(2)
        experiment_id = self.record(current, self.direction_id)
        self.close(current, self.direction_id, experiment_id)
        self.report(current, experiment_id)
        history_path = self.workspace / ".atrex_long_horizon/episodes/e0001/supervisor_runtime/journal.json"
        historical = json.loads(history_path.read_bytes())
        broken = copy.deepcopy(historical)
        del broken["direction_events"][1]  # No start in the earlier Episode.
        _write_json(history_path, broken)
        self.assertIn("Invalid Direction history in Episode 1", self.recheck(current))
        _write_json(history_path, historical)
        future_root = self.workspace / ".atrex_long_horizon/episodes/e0003/supervisor_runtime"
        initialize_journal(future_root / "journal.json", episode=3, base_commit="a" * 40, branch="episode-3")
        _, future_id = _record_kernel(future_root, b"# future source\n")
        self.assertEqual(self.recheck(current), "")
        value = load_journal(current.path)
        value["experiments"][0]["gateway_record_ids"] = [future_id]
        _write_json(current.path, value)
        self.assertIn("not visible", self.recheck(current))

    def test_request_transition_errors_do_not_mutate_journal(self) -> None:
        self.archive_first_episode()
        service = self.new_episode(2)
        direction = self.propose(service)
        before = service.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "cannot complete from status proposed"):
            self.close(service, direction, self.experiment_id)
        self.assertEqual(service.path.read_bytes(), before)
        self.start(service, direction)
        other = self.propose(service)
        before = service.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "only one Direction"):
            self.start(service, other)
        self.assertEqual(service.path.read_bytes(), before)

    def test_recovery_uses_campaign_history_and_rejects_missing_measurement_before_git_checks(self) -> None:
        value = copy.deepcopy(self.original)
        value["experiments"][0].pop("gateway_record_ids")
        _write_json(self.service.path, value)
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        worktree = EpisodeWorktree(1, "a" * 40, "episode-1", self.workspace)
        with patch.object(EpisodeWorktree, "validate_candidate") as git_validation:
            diagnosis = runner._completion_check(worktree, self.service.path, EpisodeHandoff("candidate_ready", "b" * 40))
        self.assertIn("gateway_record_ids", diagnosis)
        git_validation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
