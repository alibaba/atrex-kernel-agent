"""Storage read failures must not turn a recorded Gateway job into another submission."""

from __future__ import annotations

import errno
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from supervisor import gateway
from supervisor.errors import RuntimeStateError
from supervisor import test_gateway_deduplication as fixtures


class GatewayEvidenceIOTest(unittest.TestCase):
    setUp = fixtures.GatewayDeduplicationTest.setUp
    run_task = fixtures.GatewayDeduplicationTest.run_task
    assert_duplicate = fixtures.GatewayDeduplicationTest.assert_duplicate
    typed_response = staticmethod(fixtures.GatewayDeduplicationTest.typed_response)
    dev_response = staticmethod(fixtures.GatewayDeduplicationTest.dev_response)

    def pending(self):
        with (
            patch.object(gateway, "_validated_cached_gateway_record", side_effect=OSError(errno.EIO, "busy storage")) as read,
            patch.object(gateway.time, "sleep"),
        ):
            with self.assertRaises(RuntimeStateError) as failure:
                self.run_task("--kind", "check")
        self.assertEqual(read.call_count, 3)
        error = failure.exception.response["error"]
        self.assertEqual(error["code"], "gateway_evidence_unavailable")
        record_id = error["gateway_record_id"]
        record = gateway._load_gateway_record(self.workspace, record_id)
        marker = gateway._gateway_task_root(self.workspace) / f"{record['gateway_task_digest']}.json"
        state = json.loads(marker.read_text())
        self.assertEqual(state["status"], "validation_pending")
        self.assertEqual(state["gateway_record_id"], record_id)
        self.assertNotIn("owner", state)  # _gateway_task's finally must not discard it.
        return record, marker

    def test_transient_completion_read_retries_without_resubmission(self):
        validate = gateway._validated_cached_gateway_record
        failures = iter([True, True, False])

        def flaky(*args):
            if next(failures):
                raise OSError(errno.EIO, "busy storage")
            return validate(*args)

        with (
            patch.object(gateway, "_validated_cached_gateway_record", side_effect=flaky) as read,
            patch.object(gateway.time, "sleep") as sleep,
        ):
            self.assertEqual(self.run_task("--kind", "check")[0], 0)
        self.assertEqual(read.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.1])
        record = gateway._visible_gateway_records(self.workspace)[-1]
        self.assert_duplicate(lambda: self.run_task("--kind", "check"), record["record_id"])
        self.assertEqual(self.typed.call_count, 1)

    def test_pending_index_survives_cleanup_and_revalidates_without_resubmission(self):
        record, marker = self.pending()
        self.assert_duplicate(lambda: self.run_task("--kind", "check"), record["record_id"])
        self.assertEqual(json.loads(marker.read_text())["status"], "completed")
        with patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}):
            code, output = self.run_task("--kind", "check")
        self.assertEqual(code, 0)
        self.assertIn(record["record_id"], output)
        self.assertEqual(self.typed.call_count, 1)

    def test_pending_history_is_revalidated_without_mutating_archive(self):
        with patch.dict(os.environ, {
            gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.history / "e0001/supervisor_runtime"),
        }):
            record, marker = self.pending()
        saved = marker.read_bytes()
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history)}):
            self.assert_duplicate(lambda: self.run_task("--kind", "check"), record["record_id"])
        self.assertEqual(marker.read_bytes(), saved)
        self.assertEqual(self.typed.call_count, 1)

    def test_unavailable_current_and_historical_indexes_are_not_removed_or_bypassed(self):
        record, marker = self.pending()
        self.typed.reset_mock()
        for historical in (False, True):
            for status in ("completed", "validation_pending"):
                with self.subTest(historical=historical, status=status):
                    state = json.loads(marker.read_text())
                    state["status"] = status
                    marker.write_text(json.dumps(state))
                    saved = marker.read_bytes()
                    with (
                        patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.root / "new-evidence")}) if historical else patch.dict(os.environ, {}),
                        patch.object(gateway, "_historical_evidence_roots", return_value=[self.evidence] if historical else []),
                        patch.object(gateway, "_validated_cached_gateway_record", side_effect=PermissionError(errno.EACCES, "unreadable")),
                        patch.object(gateway.time, "sleep"),
                    ):
                        with self.assertRaises(RuntimeStateError):
                            self.run_task("--kind", "check")
                    self.assertEqual(marker.read_bytes(), saved)
        self.typed.assert_not_called()
        self.assertIsNotNone(gateway._validated_cached_gateway_record(
            self.workspace, record["gateway_task_digest"], record["record_id"],
        ))

    def test_pending_record_confirmed_missing_allows_fresh_reservation(self):
        record, marker = self.pending()
        (record["record_dir"] / "kernel.py").unlink()
        self.assertEqual(self.run_task("--kind", "check")[0], 0)
        state = json.loads(marker.read_text())
        self.assertEqual(state["status"], "completed")
        self.assertNotEqual(state["gateway_record_id"], record["record_id"])
        self.assertEqual(self.typed.call_count, 2)

    def test_index_and_history_discovery_io_errors_do_not_look_like_missing_tasks(self):
        history_root = self.history / "e0001/supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(history_root)}):
            _, marker = self.pending()
        saved = marker.read_bytes()
        original = Path.lstat
        for target in (self.history, history_root, history_root / "gateway-tasks" / marker.name):
            with self.subTest(target=target):
                def unavailable(path, *args, **kwargs):
                    if path == target:
                        raise OSError(errno.EIO, "storage stat failure")
                    return original(path, *args, **kwargs)

                with (
                    patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history)}),
                    patch.object(Path, "lstat", unavailable),
                ):
                    with self.assertRaises(OSError):
                        self.run_task("--kind", "check")
                self.assertEqual(marker.read_bytes(), saved)
        self.assertEqual(self.typed.call_count, 1)

    def test_record_read_and_stat_io_errors_reach_retry_policy(self):
        record, marker = self.pending()
        paths = [
            ("read_text", record["record_dir"] / "result.json"),
            ("read_bytes", record["record_dir"] / "kernel.py"),
            ("read_text", gateway._kernel_artifact_root(self.workspace) / record["kernel_sha256"] / "identity.json"),
            ("lstat", record["record_dir"]),
            ("lstat", record["record_dir"] / "result.json"),
        ]
        for method, target in paths:
            with self.subTest(method=method, target=target):
                original = getattr(Path, method)

                def unavailable(path, *args, **kwargs):
                    if path == target:
                        raise OSError(errno.EIO, "storage read failure")
                    return original(path, *args, **kwargs)

                saved = marker.read_bytes()
                with patch.object(Path, method, unavailable), patch.object(gateway.time, "sleep"):
                    with self.assertRaises(RuntimeStateError):
                        gateway._reserve_gateway_task(self.workspace, record["gateway_task_digest"])
                self.assertEqual(marker.read_bytes(), saved)

    def test_completion_validation_does_not_hold_lock_or_replace_another_owner(self):
        digest = "a" * 64
        owner, _ = gateway._reserve_gateway_task(self.workspace, digest)
        marker = gateway._gateway_task_root(self.workspace) / f"{digest}.json"
        reading, release = Event(), Event()

        def unavailable(*args):
            reading.set()
            if not release.wait(5):
                raise AssertionError("completion validation blocked task lock")
            raise OSError(errno.EIO, "unreadable")

        with (
            patch.object(gateway, "_validated_cached_gateway_record", side_effect=unavailable),
            patch.object(gateway.time, "sleep"),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            completion = pool.submit(gateway._complete_gateway_task, self.workspace, digest, owner, "record-placeholder")
            try:
                self.assertTrue(reading.wait(5))
                # This task's reservation still blocks a duplicate while I/O retries run.
                self.assertEqual(pool.submit(gateway._reserve_gateway_task, self.workspace, digest).result(2), (None, None))
                other_digest = "b" * 64
                other_owner, _ = pool.submit(gateway._reserve_gateway_task, self.workspace, other_digest).result(2)
                self.assertIsNotNone(other_owner)
                with gateway._locked_gateway_task_root(self.workspace):
                    marker.write_text(json.dumps({"status": "running", "owner": "replacement-owner"}))
                saved = marker.read_bytes()
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "owned by another request"):
                completion.result(5)
        self.assertEqual(marker.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
