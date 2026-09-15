from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from long_horizon.journal import validate_terminal
from supervisor.errors import AgentRequestError
from supervisor.identifiers import kernel_id_for_digest
from supervisor.journal import (
    SupervisorJournalService,
    _has_passing_evaluate,
    _validate_record_ids,
    initialize_journal,
    load_journal,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _record_kernel(evidence: Path, source: bytes) -> tuple[str, str]:
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    kernel_id = kernel_id_for_digest(digest)
    artifact = evidence / "kernel-artifacts" / "sha256" / digest.removeprefix("sha256:")
    artifact.mkdir(parents=True)
    (artifact / "kernel.py").write_bytes(source)
    _write_json(
        artifact / "identity.json",
        {"kernel_id": kernel_id, "kernel_artifact_digest": digest},
    )
    gateway_id = "gateway-" + digest[-32:]
    _write_json(
        evidence / "gateway-records" / gateway_id / "result.json",
        {
            "gateway_kind": "run",
            "kernel_id": kernel_id,
            "kernel_artifact_digest": digest,
            "kernel_sha256": digest.removeprefix("sha256:"),
            "result": {"all_pass": True, "latency_us_geomean": 10.0},
        },
    )
    return kernel_id, gateway_id


class SupervisorJournalServiceTest(unittest.TestCase):
    def test_bad_report_can_be_repaired_without_publishing_a_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / ".private" / "evidence"
            path = evidence / "journal.json"
            handoff = workspace / ".atrex_long_horizon" / "handoff.json"
            initialize_journal(path, episode=3, base_commit="base", branch="episode-3")
            service = SupervisorJournalService(
                workspace=workspace, campaign_root=workspace, evidence_root=evidence,
            )
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "summary must be non-empty"):
                service.execute({
                    "operation": "episode_report",
                    "request": {"status": "pivot", "summary": " "},
                })
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(handoff.exists())
            self.assertEqual(service.execute({
                "operation": "episode_report",
                "request": {"status": "pivot", "summary": "Need a different direction"},
            }), {"status": "accepted", "message": "Report accepted and recorded"})
            self.assertEqual(json.loads(handoff.read_text()), {"status": "pivot"})
            self.assertEqual(validate_terminal(
                path, expected_episode=3, base_commit="base", branch="episode-3", state="pivot",
                campaign_root=workspace, workspace=workspace,
            ), "")

    def test_genealogy_validates_visible_history_and_remains_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def service_for(episode: int) -> SupervisorJournalService:
                evidence = (
                    root
                    / ".atrex_long_horizon"
                    / "episodes"
                    / f"e{episode:04d}"
                    / "supervisor_runtime"
                )
                initialize_journal(
                    evidence / "journal.json",
                    episode=episode,
                    base_commit="a" * 40,
                    branch=f"episode-{episode}",
                )
                return SupervisorJournalService(
                    workspace=root, campaign_root=root, evidence_root=evidence
                )

            def propose(service: SupervisorJournalService, **ancestry: object) -> str:
                result = service.execute(
                    {
                        "operation": "direction_update",
                        "request": {
                            "action": "propose",
                            "name": "test",
                            "hypothesis": "test mechanism",
                            "rationale": "prior evidence",
                            "plan": ["measure"],
                            "success_criteria": ["correct and faster"],
                            "stop_conditions": ["no gain"],
                            **ancestry,
                        },
                    }
                )
                return str(result["direction_id"])

            previous = service_for(1)
            parent_a = propose(previous)
            parent_b = propose(previous)
            future = propose(service_for(3))
            current = service_for(2)
            child = propose(
                current, relationship="combination", derived_from_direction_ids=[parent_a, parent_b]
            )
            loaded = current.execute({"operation": "direction_load", "direction_id": child})
            self.assertEqual(loaded["derived_from_direction_ids"], [parent_a, parent_b])
            before = current.path.read_bytes()
            for parents in ([future], [parent_a, parent_a]):
                with self.assertRaises(ValueError):
                    propose(current, relationship="retry", derived_from_direction_ids=parents)
                self.assertEqual(current.path.read_bytes(), before)
            with self.assertRaises(ValueError):
                current.execute(
                    {
                        "operation": "direction_update",
                        "request": {
                            "action": "start",
                            "direction_id": child,
                            "analysis": "rewrite",
                            "relationship": "retry",
                            "derived_from_direction_ids": [parent_a],
                        },
                    }
                )
            result = current.execute(
                {"operation": "directions_list", "file": "scratch/directions.json"}
            )
            self.assertEqual(result["count"], 3)
            index = json.loads((root / "scratch/directions.json").read_bytes())
            listed = next(item for item in index["directions"] if item["direction_id"] == child)
            self.assertEqual(listed["relationship"], "combination")
            self.assertEqual(listed["derived_from_direction_ids"], [parent_a, parent_b])
            self.assertNotIn(future, json.dumps(index))
            with self.assertRaisesRegex(ValueError, "unsupported Journal operation"):
                current.execute({"operation": "direction_graph", "file": "scratch/removed.json"})
            self.assertFalse((root / "scratch/removed.json").exists())

    def test_runtime_binds_facts_and_publishes_repairable_terminal_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "episode-1"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
            source = b"def run(x):\n    return x\n"
            (workspace / "kernel.py").write_bytes(b"def run(x):\n    return x + 0\n")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-qm", "candidate"], cwd=workspace, check=True)
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            (workspace / "kernel.py").write_bytes(source)
            from long_horizon.store import CampaignStore

            CampaignStore.ensure_excluded(workspace)

            evidence = workspace / ".private" / "evidence"
            kernel_id, gateway_id = _record_kernel(evidence, source)
            initialize_journal(
                evidence / "journal.json",
                episode=1,
                base_commit=commit,
                branch="episode-1",
            )
            service = SupervisorJournalService(
                workspace=workspace,
                campaign_root=workspace,
                evidence_root=evidence,
            )
            proposed = service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "propose",
                        "name": "reduce launch overhead",
                        "hypothesis": "one fused launch is faster",
                        "rationale": "the current path launches twice",
                        "plan": ["fuse", "measure"],
                        "success_criteria": ["correct and faster"],
                        "stop_conditions": ["compile failure after repair"],
                    },
                }
            )
            direction_id = str(proposed["direction_id"])
            service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "start",
                        "direction_id": direction_id,
                        "analysis": "begin exploration",
                    },
                }
            )
            recorded = service.execute(
                {
                    "operation": "experiment_record",
                    "request": {
                        "direction_id": direction_id,
                        "name": "candidate measurement",
                        "hypothesis": "fewer launches improve latency",
                        "change": "fused the two stages",
                        "gateway_record_ids": [gateway_id],
                        "evidence": "full evaluator passed",
                        "analysis": "retain the measured candidate",
                        "action": "baseline",
                    },
                }
            )
            experiment_id = str(recorded["experiment_id"])
            loaded = service.execute(
                {"operation": "experiment_load", "experiment_id": experiment_id}
            )
            self.assertEqual(loaded["gateway_record_ids"], [gateway_id])
            for removed in ("before", "after", "kernel_id", "kernel_artifact_digest", "sequence"):
                self.assertNotIn(removed, loaded)

            profile_id = "gateway-11111111111111111111111111111111"
            _write_json(
                evidence / "gateway-records" / profile_id / "result.json",
                {
                    "gateway_kind": "profile",
                    "kernel_id": kernel_id,
                    "kernel_artifact_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
                    "result": {},
                },
            )
            profile_experiment = service.execute(
                {
                    "operation": "experiment_record",
                    "request": {
                        "direction_id": direction_id,
                        "name": "profile interpretation",
                        "hypothesis": "check bottleneck",
                        "change": "inspect the measured candidate",
                        "gateway_record_ids": [profile_id],
                        "evidence": "profile only",
                        "analysis": "needs evaluation evidence",
                        "action": "keep_after",
                    },
                }
            )
            profile_loaded = service.execute(
                {
                    "operation": "experiment_load",
                    "experiment_id": profile_experiment["experiment_id"],
                }
            )
            self.assertEqual(profile_loaded["gateway_record_ids"], [profile_id])

            with self.assertRaisesRegex(ValueError, "in progress"):
                service.execute(
                    {
                        "operation": "episode_report",
                        "request": {"status": "pivot", "summary": "not yet closed"},
                    }
                )
            service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "complete",
                        "direction_id": direction_id,
                        "analysis": "measurement supports the hypothesis",
                        "hypothesis_status": "supported",
                        "supporting_experiment_ids": [experiment_id],
                    },
                }
            )
            with self.assertRaisesRegex(ValueError, "passing"):
                service.execute(
                    {
                        "operation": "episode_report",
                        "request": {
                            "status": "candidate_ready",
                            "summary": "profile does not suffice",
                            "selected_experiment_id": profile_experiment["experiment_id"],
                        },
                    }
                )
            self.assertEqual(subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
                capture_output=True, check=True,
            ).stdout.strip(), commit)
            valid_report = {
                "status": "candidate_ready", "summary": "measured candidate",
                "selected_experiment_id": experiment_id,
            }
            journal_before_report = service.path.read_bytes()
            for field, value in {
                "git_commit_hash": commit, "candidate_commit": commit,
                "last_trial_commit": commit, "selected_experiment_index": 1,
            }.items():
                with self.assertRaises(AgentRequestError) as rejected:
                    service.execute({"operation": "episode_report", "request": {
                        **valid_report, field: value,
                    }})
                self.assertEqual(rejected.exception.response["error"]["unexpected_fields"], [field])
                self.assertEqual(rejected.exception.response["error"]["missing_fields"], [])
                self.assertEqual(service.path.read_bytes(), journal_before_report)
                self.assertFalse((workspace / ".atrex_long_horizon" / "handoff.json").exists())
            (workspace / "kernel.py").write_bytes(source + b"# not measured\n")
            with self.assertRaisesRegex(ValueError, "exactly matches current kernel.py"):
                service.execute({"operation": "episode_report", "request": valid_report})
            (workspace / "kernel.py").write_bytes(source)
            result = service.execute(
                {
                    "operation": "episode_report",
                    "request": {
                        "status": "candidate_ready",
                        "summary": "candidate is ready for Supervisor comparison",
                        "selected_experiment_id": experiment_id,
                    },
                }
            )
            self.assertEqual(result["status"], "accepted")
            candidate = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
                capture_output=True, check=True,
            ).stdout.strip()
            self.assertNotEqual(candidate, commit)
            self.assertEqual(service.execute({
                "operation": "episode_report", "request": valid_report,
            })["status"], "accepted")
            self.assertEqual(subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
                capture_output=True, check=True,
            ).stdout.strip(), candidate)
            private = load_journal(evidence / "journal.json")
            self.assertTrue(private["runtime_managed"])
            self.assertEqual(private["outcome"]["selected_experiment_id"], experiment_id)
            self.assertNotIn("selected_experiment_index", private["outcome"])
            self.assertNotIn("evaluation", private["experiments"][0])
            self.assertEqual(validate_terminal(
                service.path, expected_episode=1, base_commit=commit, branch="episode-1",
                state="candidate_ready", candidate_commit=candidate,
                campaign_root=workspace, workspace=workspace,
            ), "")
            self.assertEqual(
                private["experiments"][0]["gateway_record_ids"],
                [gateway_id],
            )
            self.assertEqual(
                json.loads((workspace / ".atrex_long_horizon" / "handoff.json").read_text()),
                {"status": "candidate_ready", "candidate_commit": candidate},
            )

    def test_history_is_queryable_and_can_supply_gateway_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            history = (
                workspace / ".atrex_long_horizon" / "episodes" / "e0001" / "supervisor_runtime"
            )
            _, gateway_id = _record_kernel(history, b"def run(x):\n    return x\n")
            direction_id = "direction_11111111111111111111111111111111"
            experiment_id = "experiment_11111111111111111111111111111111"
            _write_json(
                history / "journal.json",
                {
                    "schema_version": 2,
                    "runtime_managed": True,
                    "episode": 1,
                    "memory_version": 1,
                    "base_commit": "1" * 40,
                    "episode_branch": "episode-1",
                    "state": "pivot",
                    "direction_events": [
                        {
                            "direction_event_id": "directionevent_" + "1" * 32,
                            "direction_id": direction_id,
                            "action": "propose",
                            "name": "historical direction",
                            "hypothesis": "historical hypothesis",
                            "rationale": "historical rationale",
                            "plan": ["measure"],
                            "success_criteria": ["faster"],
                            "stop_conditions": ["slower"],
                            "recorded_at": "2026-01-01T00:00:00+00:00",
                        },
                        {
                            "direction_event_id": "directionevent_" + "2" * 32,
                            "direction_id": direction_id,
                            "action": "start",
                            "analysis": "started",
                            "recorded_at": "2026-01-01T00:00:01+00:00",
                        },
                        {
                            "direction_event_id": "directionevent_" + "3" * 32,
                            "direction_id": direction_id,
                            "action": "complete",
                            "analysis": "completed",
                            "recorded_at": "2026-01-01T00:00:02+00:00",
                        },
                    ],
                    "experiments": [
                        {
                            "experiment_id": experiment_id,
                            "direction_id": direction_id,
                            "sequence": 1,
                            "recorded_at": "2026-01-01T00:00:01+00:00",
                            "name": "historical experiment",
                            "hypothesis": "historical hypothesis",
                            "change": "historical change",
                            "gateway_record_ids": [gateway_id],
                            "evidence": "passed",
                            "analysis": "useful",
                            "action": "baseline",
                        }
                    ],
                    "outcome": {"summary": "done", "next_directions": []},
                    "candidate_commit": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "finalized_at": "2026-01-01T00:00:03+00:00",
                },
            )
            current = workspace / ".private" / "evidence"
            initialize_journal(
                current / "journal.json",
                episode=2,
                base_commit="2" * 40,
                branch="episode-2",
            )
            service = SupervisorJournalService(
                workspace=workspace,
                campaign_root=workspace,
                evidence_root=current,
            )
            loaded = service.execute(
                {"operation": "experiment_load", "experiment_id": experiment_id}
            )
            self.assertEqual(loaded["gateway_record_ids"], [gateway_id])
            self.assertNotIn("kernel_artifact_digest", loaded)

            proposed = service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "propose",
                        "name": "reuse historical kernel",
                        "hypothesis": "the old kernel remains a useful anchor",
                        "rationale": "its measurement is authoritative",
                        "plan": ["compare a new implementation"],
                        "success_criteria": ["new result is faster"],
                        "stop_conditions": ["new result regresses"],
                    },
                }
            )
            new_direction = str(proposed["direction_id"])
            service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "start",
                        "direction_id": new_direction,
                        "analysis": "reuse the old measured anchor",
                    },
                }
            )
            recorded = service.execute(
                {
                    "operation": "experiment_record",
                    "request": {
                        "direction_id": new_direction,
                        "name": "historical adoption",
                        "hypothesis": "reuse avoids redundant evaluation",
                        "change": "adopted the old measured Kernel",
                        "gateway_record_ids": [gateway_id],
                        "evidence": "the archived Gateway record remains available",
                        "analysis": "no new Gateway call was required",
                        "action": "adopt",
                    },
                }
            )
            private = load_journal(current / "journal.json")
            stored = next(
                item
                for item in private["experiments"]
                if item["experiment_id"] == recorded["experiment_id"]
            )
            self.assertEqual(stored["gateway_record_ids"], [gateway_id])
            service.execute({"operation": "experiments_list", "file": "scratch/index.json"})
            index = json.loads((workspace / "scratch/index.json").read_text())["experiments"]
            self.assertEqual(
                [item["gateway_record_ids"] for item in index], [[gateway_id], [gateway_id]]
            )
            from long_horizon.campaign import _memory_experience

            memory = _memory_experience(private)
            self.assertEqual(memory["experiments"][0]["gateway_record_ids"], [gateway_id])
            self.assertNotIn("before", memory["experiments"][0])
            self.assertNotIn("after", memory["experiments"][0])

    def test_references_accept_multiple_kernels_and_gateway_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / "evidence"
            kernel_id, evaluate_id = _record_kernel(evidence, b"def run(x): return x\n")
            digest = "sha256:" + hashlib.sha256(b"def run(x): return x\n").hexdigest()
            profile_id = "gateway-11111111111111111111111111111111"
            abba_id = "gateway-22222222222222222222222222222222"
            _write_json(
                evidence / "gateway-records" / profile_id / "result.json",
                {
                    "gateway_kind": "profile",
                    "kernel_id": kernel_id,
                    "kernel_artifact_digest": digest,
                    "result": {},
                },
            )
            _write_json(
                evidence / "gateway-records" / abba_id / "result.json",
                {
                    "gateway_kind": "same_allocation_abba",
                    "kernel_id": "kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "kernel_artifact_digest": digest,
                    "kernel_subject_ids": {
                        "incumbent": kernel_id,
                        "candidate": "kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    },
                    "result": {},
                },
            )
            _, other_id = _record_kernel(evidence, b"def run(x): return x + 1\n")
            others = []
            for offset, kind in enumerate(("dev", "check", "disassemble"), start=103):
                record_id = f"gateway-{offset:032x}"
                _write_json(
                    evidence / "gateway-records" / record_id / "result.json",
                    {
                        "gateway_kind": kind,
                        "kernel_id": kernel_id,
                        "kernel_artifact_digest": digest,
                        "result": {},
                    },
                )
                others.append(record_id)
            for selected in (
                [profile_id],
                [evaluate_id, profile_id],
                [abba_id],
                [evaluate_id, other_id, *others],
            ):
                with self.subTest(selected=selected):
                    self.assertEqual(_validate_record_ids(evidence, workspace, selected), selected)

    def test_references_reject_invalid_duplicate_and_invisible_result_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / "evidence"
            _, gateway_id = _record_kernel(evidence, b"def run(x): return x\n")
            symlink_id = "gateway-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            (evidence / "gateway-records" / symlink_id).symlink_to(
                evidence / "gateway-records" / gateway_id,
                target_is_directory=True,
            )
            broken_id = "gateway-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            _write_json(evidence / "gateway-records" / broken_id / "result.json", {})
            cases = [
                (None, "array"),
                (gateway_id, "array"),
                (["invalid"], "valid gateway"),
                ([gateway_id, gateway_id], "duplicates"),
                (["gateway-ffffffffffffffffffffffffffffffff"], "not visible"),
                ([symlink_id], "not visible"),
                ([broken_id], "invalid"),
            ]
            for raw, message in cases:
                with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                    _validate_record_ids(evidence, workspace, raw)

    def test_candidate_requires_a_cited_passing_evaluate_for_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / "evidence"
            source = b"def run(x): return x\n"
            digest = "sha256:" + hashlib.sha256(source).hexdigest()
            _, matching = _record_kernel(evidence, source)
            _, other = _record_kernel(evidence, b"def run(x): return x + 1\n")
            failed = "gateway-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            profile = "gateway-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            for record_id, kind, passed in ((failed, "run", False), (profile, "profile", True)):
                _write_json(
                    evidence / "gateway-records" / record_id / "result.json",
                    {
                        "gateway_kind": kind,
                        "kernel_artifact_digest": digest,
                        "result": {"all_pass": passed},
                    },
                )
            for ids in ([other], [profile], [failed], [other, profile, failed]):
                with self.subTest(ids=ids):
                    self.assertFalse(_has_passing_evaluate(evidence, workspace, ids, digest))
            self.assertTrue(
                _has_passing_evaluate(evidence, workspace, [other, matching, profile], digest)
            )


if __name__ == "__main__":
    unittest.main()
