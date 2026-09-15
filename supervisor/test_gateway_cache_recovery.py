"""Invalid cached evidence must not poison reservation or weaken strict reads."""

from __future__ import annotations

import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from unittest.mock import patch

from supervisor import gateway
from supervisor import test_gateway_deduplication as fixtures


class GatewayCacheRecoveryTest(unittest.TestCase):
    setUp = fixtures.GatewayDeduplicationTest.setUp
    run_task = fixtures.GatewayDeduplicationTest.run_task
    assert_duplicate = fixtures.GatewayDeduplicationTest.assert_duplicate
    typed_response = staticmethod(fixtures.GatewayDeduplicationTest.typed_response)
    dev_response = staticmethod(fixtures.GatewayDeduplicationTest.dev_response)

    def completed(self, root):
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(root)}):
            self.assertEqual(self.run_task("--kind", "check")[0], 0)
            record = gateway._visible_gateway_records(self.workspace)[-1]
        marker = root / "gateway-tasks" / f"{record['gateway_task_digest']}.json"
        return record, marker

    def damage(self, record, marker, case):
        record_id = record["record_id"]
        preserved = record["record_dir"]
        result_path = preserved / "result.json"
        if case == "legacy_gateway_id":
            record_id = "gateway-20260815-0001"
            marker.write_text(json.dumps({"status": "completed", "gateway_record_id": record_id}))
        elif case == "missing_record":
            moved = preserved.with_name(f"removed-{record_id}")
            preserved.rename(moved)
            preserved = moved
        elif case == "missing_result":
            result_path.rename(preserved / "result.missing")
        elif case == "missing_kernel":
            (preserved / "kernel.py").rename(preserved / "kernel.missing")
        elif case == "source_mismatch":
            (preserved / "kernel.py").write_bytes(b"# a different Kernel\n")
        else:
            value = json.loads(result_path.read_text())
            if case == "legacy_kernel_id":
                value["kernel_id"] = "kernel-20260815-0001"
            elif case == "task_mismatch":
                value["gateway_task_digest"] = "f" * 64
            elif case == "identity_mismatch":
                value["kernel_sha256"] = "f" * 64
                value["kernel_artifact_digest"] = "sha256:" + "f" * 64
            else:
                self.fail(f"unknown damage case {case}")
            result_path.write_text(json.dumps(value))
        snapshot = {p.name: p.read_bytes() for p in preserved.iterdir() if p.is_file()}
        return record_id, preserved, snapshot

    def test_invalid_current_and_historical_records_allow_fresh_execution(self):
        cases = (
            "legacy_gateway_id", "legacy_kernel_id", "missing_record", "missing_result",
            "missing_kernel", "task_mismatch", "identity_mismatch", "source_mismatch",
        )
        for historical in (False, True):
            for case in cases:
                with self.subTest(historical=historical, case=case):
                    location = self.root / f"{historical}-{case}"
                    current, history = location / "current", location / "history"
                    old_root = history / "e0001/supervisor_runtime" if historical else current
                    self.typed.reset_mock()
                    record, marker = self.completed(old_root)
                    digest = record["gateway_task_digest"]
                    old_id, preserved, snapshot = self.damage(record, marker, case)
                    old_index = marker.read_bytes()
                    with patch.dict(os.environ, {
                        gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(current),
                        gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history),
                    }):
                        # Only the dedup index may recover. Direct validation/reuse stays strict.
                        for read in (gateway._validated_cached_gateway_record, gateway._reusable_gateway_record):
                            with self.assertRaises((ValueError, OSError, RuntimeError)):
                                read(self.workspace, digest, old_id)
                        self.assertEqual(self.run_task("--kind", "check")[0], 0)
                        new_index = json.loads((current / "gateway-tasks" / marker.name).read_text())
                        self.assertEqual(new_index["status"], "completed")
                        new_id = new_index["gateway_record_id"]
                        self.assertNotEqual(new_id, old_id)
                        self.assertIsNotNone(gateway._validated_cached_gateway_record(self.workspace, digest, new_id))
                        self.assert_duplicate(lambda: self.run_task("--kind", "check"), new_id)
                    self.assertEqual(self.typed.call_count, 2)
                    self.assertEqual({p.name: p.read_bytes() for p in preserved.iterdir() if p.is_file()}, snapshot)
                    if historical:
                        self.assertEqual(marker.read_bytes(), old_index)

    def test_validation_errors_during_completion_release_only_owned_reservations(self):
        for case in ("legacy_kernel_id", "missing_kernel", "task_mismatch"):
            with self.subTest(case=case):
                root = self.root / case
                record, marker = self.completed(root)
                digest = record["gateway_task_digest"]
                old_id, preserved, snapshot = self.damage(record, marker, case)
                with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(root)}):
                    owner, previous = gateway._reserve_gateway_task(self.workspace, digest)
                    self.assertIsNotNone(owner)
                    self.assertIsNone(previous)
                    original = marker.read_bytes()
                    gateway._complete_gateway_task(self.workspace, digest, "another-owner", old_id)
                    self.assertEqual(marker.read_bytes(), original)
                    gateway._complete_gateway_task(self.workspace, digest, owner, old_id)
                    self.assertFalse(marker.exists())
                    new_owner, previous = gateway._reserve_gateway_task(self.workspace, digest)
                    self.assertIsNotNone(new_owner)
                    self.assertIsNone(previous)
                    gateway._abandon_gateway_task(self.workspace, digest, new_owner)
                self.assertEqual({p.name: p.read_bytes() for p in preserved.iterdir() if p.is_file()}, snapshot)

    def test_valid_older_history_is_reused_after_stale_current_and_newer_indexes(self):
        valid, valid_marker = self.completed(self.history / "e0001/supervisor_runtime")
        invalid, invalid_marker = self.completed(self.history / "e0002/supervisor_runtime")
        self.damage(invalid, invalid_marker, "legacy_gateway_id")
        current, current_marker = self.completed(self.evidence)
        self.damage(current, current_marker, "legacy_kernel_id")
        valid_index, invalid_index = valid_marker.read_bytes(), invalid_marker.read_bytes()
        self.typed.reset_mock()
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history)}):
            self.assert_duplicate(lambda: self.run_task("--kind", "check"), valid["record_id"])
        self.typed.assert_not_called()
        self.assertFalse(current_marker.exists())
        self.assertEqual(valid_marker.read_bytes(), valid_index)
        self.assertEqual(invalid_marker.read_bytes(), invalid_index)

    def test_invalid_new_publication_is_not_marked_completed_or_emitted_as_a_receipt(self):
        record, marker = self.completed(self.evidence)
        digest = record["gateway_task_digest"]
        self.damage(record, marker, "legacy_gateway_id")
        source = (self.workspace / "kernel.py").read_bytes()
        with gateway._gateway_task(self.workspace, digest, source) as task:
            with patch.object(gateway, "_validated_cached_gateway_record", side_effect=ValueError("invalid record")):
                published = task.record({"passed": True}, gateway_kind="check")
            self.assertFalse(marker.exists())
        # The new immutable record still exists for audit; no invalid fact was cached.
        self.assertTrue((self.evidence / "gateway-records" / published["record_id"] / "result.json").is_file())
        self.assertFalse(marker.exists())
        with (
            patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}),
            patch.object(gateway, "_load_gateway_record", side_effect=ValueError("bad identity")),
            redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaisesRegex(ValueError, "bad identity"):
                gateway._emit_supervisor_measurement(self.workspace, published["record_id"], reused=True)
            self.assertEqual(output.getvalue(), "")

    def test_one_request_reserves_after_concurrent_stale_index_recovery(self):
        record, marker = self.completed(self.evidence)
        digest = record["gateway_task_digest"]
        self.damage(record, marker, "legacy_gateway_id")
        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(lambda _: gateway._reserve_gateway_task(self.workspace, digest), range(8)))
        owners = [owner for owner, _ in reservations if owner is not None]
        self.assertEqual(len(owners), 1)
        self.assertEqual(sum(item == (None, None) for item in reservations), 7)
        self.assertEqual(json.loads(marker.read_text())["owner"], owners[0])
        gateway._abandon_gateway_task(self.workspace, digest, "another-owner")
        self.assertTrue(marker.exists())
        gateway._abandon_gateway_task(self.workspace, digest, owners[0])
        self.assertFalse(marker.exists())

    def test_index_write_failures_and_unexpected_validator_errors_still_propagate(self):
        record, marker = self.completed(self.evidence)
        digest = record["gateway_task_digest"]
        self.damage(record, marker, "legacy_gateway_id")
        original = marker.read_bytes()
        with patch.object(gateway, "durable_write_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                gateway._reserve_gateway_task(self.workspace, digest)
        self.assertFalse(marker.exists())
        marker.write_bytes(original)
        with patch.object(gateway, "_validated_cached_gateway_record", side_effect=TypeError("programming error")):
            with self.assertRaisesRegex(TypeError, "programming error"):
                gateway._reserve_gateway_task(self.workspace, digest)
        self.assertEqual(marker.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
