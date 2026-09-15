"""Bounded discovery must not re-read stable headers or lose late child sessions."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator import session_native
from orchestrator.session_capture import SessionCapture, clear_capture
from orchestrator.session_native import HostSessionTranscripts
from orchestrator.test_unsandboxed_usage import CHILD_ID, ROOT_ID, codex_stream, count, lines, metadata


class SessionDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(clear_capture)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "codex"
        self.environment = {"CODEX_HOME": str(self.home)}
        clock = patch.object(session_native.time, "monotonic", return_value=0.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def write(self, name, text):
        path = self.home / "sessions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def discovery(self, **kwargs):
        return HostSessionTranscripts("codex", self.environment, ["codex", "resume", ROOT_ID], **kwargs)

    def paths(self, discovery, **kwargs):
        return {path for path, _ in discovery.selected(**kwargs).values()}

    def test_stable_headers_and_negative_results_are_cached_between_scans(self):
        root = self.write("main.jsonl", lines(metadata(ROOT_ID)))
        for name, content in (("empty", ""), ("partial", '{"type":'), ("no-meta", "{}\n")):
            self.write(name + ".jsonl", content)
        discovery = self.discovery()
        with (
            patch.object(discovery, "_paths", wraps=discovery._paths) as scans,
            patch.object(session_native, "_regular_bytes", wraps=session_native._regular_bytes) as reads,
        ):
            for timestamp in (0, 1, 2, 4):
                self.clock.return_value = timestamp
                self.assertEqual(self.paths(discovery), {root})
            self.assertEqual(scans.call_count, 0)
            self.assertEqual(reads.call_count, 4)
            self.clock.return_value = 5
            self.assertEqual(self.paths(discovery), {root})
            self.assertEqual(scans.call_count, 1)
            self.assertEqual(reads.call_count, 4)
            self.paths(discovery, force=True)
            self.assertEqual(scans.call_count, 2)
            self.assertEqual(reads.call_count, 4)

    def test_negative_header_is_retried_when_completed_or_rewritten(self):
        header = lines(metadata(ROOT_ID))
        path = self.write("main.jsonl", "")
        rewritten = self.write("rewritten.jsonl", " " * len(header))
        os.utime(rewritten, ns=(10, 10))
        discovery = self.discovery()
        self.assertEqual(self.paths(discovery), set())
        path.write_text(header[:8])
        self.assertEqual(self.paths(discovery), set())
        path.write_text(header)
        self.assertEqual(self.paths(discovery), {path})
        # A same-size rewrite must invalidate a negative entry as well.
        rewritten.write_text(header)
        os.utime(rewritten, ns=(20, 20))
        self.assertEqual(self.paths(discovery), {path, rewritten})

    def test_positive_headers_survive_appends_and_negative_ones_track_replacement(self):
        path = self.write("main.jsonl", lines(metadata(ROOT_ID)))
        child_header = lines(metadata(CHILD_ID, ROOT_ID))
        child = self.write("child.jsonl", " " * len(child_header))
        discovery = self.discovery()
        with patch.object(session_native, "_regular_bytes", wraps=session_native._regular_bytes) as reads:
            self.assertEqual(self.paths(discovery), {path})
            with path.open("a") as stream:
                stream.write("{}\n")
            self.assertEqual(self.paths(discovery), {path})
            self.assertEqual(reads.call_count, 2)
            previous = child.stat()
            replacement = child.with_suffix(".new")
            replacement.write_text(child_header)
            os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            replacement.replace(child)
            self.assertEqual(self.paths(discovery), {path, child})
            self.assertEqual(reads.call_count, 3)

    def test_new_children_are_discovered_periodically_or_by_forced_scan(self):
        root = self.write("main.jsonl", lines(metadata(ROOT_ID)))
        discovery = self.discovery()
        self.assertEqual(self.paths(discovery), {root})
        child = self.write("child.jsonl", lines(metadata(CHILD_ID, ROOT_ID)))
        self.clock.return_value = 4
        self.assertEqual(self.paths(discovery), {root})
        self.clock.return_value = 5
        self.assertEqual(self.paths(discovery), {root, child})
        grandchild = self.write("grandchild.jsonl", lines(metadata("grandchild", CHILD_ID)))
        self.assertEqual(self.paths(discovery, force=True), {root, child, grandchild})

    def test_announced_codex_thread_triggers_discovery_before_periodic_scan(self):
        discovery = HostSessionTranscripts("codex", self.environment, ["codex", "exec", "prompt"])
        root = self.write("main.jsonl", lines(metadata(ROOT_ID)))
        self.assertEqual(discovery.selected(), {})
        selected = discovery.selected(codex_stream())
        self.assertEqual({path for path, _ in selected.values()}, {root})

    def test_scan_prunes_negative_entries_and_keeps_inventory_bounded(self):
        for index in range(4):
            self.write(f"{index}.jsonl", "{}\n")
        limits = []
        discovery = self.discovery(max_files=2, on_limit=lambda: limits.append(True))
        self.paths(discovery)
        self.assertEqual(len(discovery._metadata), 2)
        removed = set(discovery._known_paths)
        for path in removed:
            path.unlink()
        self.paths(discovery, force=True)
        self.assertEqual(len(discovery._metadata), 2)
        self.assertFalse(removed & discovery._metadata.keys())
        self.assertTrue(limits)

    def test_cached_path_replaced_by_symlink_never_reads_the_target(self):
        path = self.write("main.jsonl", "{}\n")
        discovery = self.discovery()
        with patch.object(session_native, "_regular_bytes", wraps=session_native._regular_bytes) as reads:
            self.assertEqual(self.paths(discovery), set())
            secret = self.root / "secret.jsonl"
            secret.write_text(lines(metadata(ROOT_ID)))
            path.unlink()
            path.symlink_to(secret)
            self.assertEqual(self.paths(discovery), set())
            self.assertEqual(self.paths(discovery, force=True), set())
            self.assertEqual(reads.call_count, 1)

    def test_replacement_and_truncation_still_mark_capture_partial(self):
        for replace_file in (False, True):
            with self.subTest(replace_file=replace_file):
                path = self.write("main.jsonl", lines(metadata(ROOT_ID)))
                capture = SessionCapture(
                    self.root / "captures", backend="codex", command=["codex", "resume", ROOT_ID, "prompt"],
                    provider_home=None, native_environment=self.environment, context={},
                )
                with path.open("a") as stream:
                    stream.write(lines(count()))
                capture.sync_native()
                if replace_file:
                    replacement = path.with_suffix(".new")
                    replacement.write_text(lines(metadata(CHILD_ID), count(900)))
                    replacement.replace(path)
                else:
                    path.write_text("")
                capture._chunks["stdout"].append(codex_stream())
                capture.finish(exit_status=0)
                report = json.loads((capture.root / "token-usage.json").read_text())
                self.assertFalse(report["capture_complete"])
                self.assertEqual(report["total"]["measurement"], "partial")
                self.assertIn("replaced or truncated", str(report["capture_errors"]))

    def test_capture_keeps_live_tailing_and_force_discovers_late_children_at_finish(self):
        path = self.write("main.jsonl", lines(metadata(ROOT_ID), count()))
        capture = SessionCapture(
            self.root / "captures", backend="codex", command=["codex", "resume", ROOT_ID, "prompt"],
            provider_home=None, native_environment=self.environment, context={},
        )
        delta = count(10, 1, 2)["payload"]["info"]["total_token_usage"]
        addition = lines(count(110, 11, 22, last=delta))
        with path.open("a") as stream:
            stream.write(addition)
        capture.sync_native()
        self.assertIn(addition, b"".join(capture.native.values()).decode())
        self.write("late-child.jsonl", lines(metadata(CHILD_ID, ROOT_ID), count(20, 2, 4)))
        capture.sync_native()
        self.assertEqual(len(capture.native), 1)
        capture._chunks["stdout"].append(codex_stream(110, 11, 22))
        capture.finish(exit_status=0)
        self.assertEqual(len(capture.native), 2)
        report = json.loads((capture.root / "token-usage.json").read_text())
        self.assertTrue(report["capture_complete"])
        self.assertEqual(report["total"]["total_tokens"], 33)
        self.assertEqual(report["total"]["measurement"], "exact")


if __name__ == "__main__":
    unittest.main()
