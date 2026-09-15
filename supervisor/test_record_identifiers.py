"""Object IDs must not depend on host clocks, PIDs, or Episode-local counters."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from supervisor import gateway, journal
from supervisor import test_kernel_identity as fixtures


class GlobalRecordIdentifiersTest(unittest.TestCase):
    setUp = fixtures.GlobalKernelIdentityTest.setUp
    record = fixtures.GlobalKernelIdentityTest.record

    def test_gateway_records_are_unique_even_when_source_and_time_match(self):
        with patch.object(gateway.time, "time_ns", return_value=123):
            records = [self.record(self.evidence) for _ in range(8)]
        self.assertEqual(len({item["record_id"] for item in records}), 8)
        self.assertEqual({item["kernel_id"] for item in records}, {self.kernel_id})
        for item in records:
            self.assertRegex(item["record_id"], r"^gateway-[0-9a-f]{32}$")
            self.assertEqual(uuid.UUID(hex=item["record_id"][8:]).version, 4)

    def test_concurrent_processes_do_not_overwrite_records(self):
        code = (
            "import json,sys; from pathlib import Path; from supervisor import gateway; "
            "gateway.time.time_ns=lambda:123; "
            "print(json.dumps(gateway._record_episode_evaluation("
            "Path(sys.argv[1]), {'passed':True}, gateway_kind='check')))"
        )

        def record(_ordinal):
            result = subprocess.run(
                [sys.executable, "-c", code, str(self.workspace)],
                cwd=gateway.REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            return json.loads(result.stdout)

        with ThreadPoolExecutor(max_workers=4) as executor:
            records = list(executor.map(record, range(4)))
        self.assertEqual(len({item["record_id"] for item in records}), 4)
        self.assertEqual(len(gateway._visible_gateway_records(self.workspace)), 4)
        self.assertEqual({item["kernel_id"] for item in records}, {self.kernel_id})

    def test_record_collision_fails_closed_without_overwriting(self):
        generated = SimpleNamespace(uuid4=lambda: uuid.UUID(int=1))
        with patch.object(gateway, "uuid", generated):
            first = self.record(self.evidence)
            path = self.evidence / "gateway-records" / first["record_id"] / "result.json"
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.record(self.evidence, kind="profile")
        self.assertEqual(path.read_bytes(), original)

    def test_history_orders_by_record_time_not_uuid(self):
        generated = iter([uuid.UUID(int=2**128 - 1), uuid.UUID(int=1)])
        with patch.object(gateway, "uuid", SimpleNamespace(uuid4=lambda: next(generated))):
            first = self.record(self.evidence)
            second = self.record(self.evidence, kind="profile")
        self.assertGreater(first["record_id"], second["record_id"])
        self.assertEqual(
            [item["record_id"] for item in gateway._visible_gateway_records(self.workspace)],
            [first["record_id"], second["record_id"]],
        )

    def test_direction_experiment_and_direction_events_use_independent_uuid4_ids(self):
        direction_ids, experiment_ids, event_ids = set(), set(), set()
        for ordinal in range(2):
            evidence = self.root / f"campaign-{ordinal}"
            path = evidence / "journal.json"
            journal.initialize_journal(path, episode=1, base_commit="a" * 40, branch="episode-1")
            record = self.record(evidence)
            service = journal.SupervisorJournalService(
                workspace=self.workspace, campaign_root=self.root, evidence_root=evidence
            )
            direction = service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "propose",
                        "name": "fusion",
                        "hypothesis": "less traffic",
                        "rationale": "extra writes",
                        "plan": ["fuse"],
                        "success_criteria": ["faster"],
                        "stop_conditions": ["no improvement"],
                    },
                }
            )["direction_id"]
            direction_ids.add(direction)
            service.execute(
                {
                    "operation": "direction_update",
                    "request": {
                        "action": "start",
                        "direction_id": direction,
                        "analysis": "explore",
                    },
                }
            )
            for _ in range(2):
                experiment = service.execute(
                    {
                        "operation": "experiment_record",
                        "request": {
                            "direction_id": direction,
                            "name": "probe",
                            "hypothesis": "faster",
                            "change": "fused code",
                            "gateway_record_ids": [record["record_id"]],
                            "evidence": "measured",
                            "analysis": "needs analysis",
                            "action": "keep_after",
                        },
                    }
                )["experiment_id"]
                experiment_ids.add(experiment)
            event_ids.update(
                event["direction_event_id"]
                for event in journal.load_journal(path)["direction_events"]
            )
        self.assertEqual(len(direction_ids), 2)
        self.assertEqual(len(experiment_ids), 4)
        self.assertEqual(len(event_ids), 4)
        for identifier in direction_ids | experiment_ids | event_ids:
            self.assertEqual(uuid.UUID(hex=identifier.split("_", 1)[1]).version, 4)

    def test_old_gateway_ids_are_rejected(self):
        old_id = "gateway-100-0123456789ab"
        with self.assertRaisesRegex(ValueError, "invalid format"):
            gateway._load_gateway_record(self.workspace, old_id)
        with self.assertRaisesRegex(ValueError, "valid gateway"):
            journal._gateway_record(self.evidence, self.root, old_id)
