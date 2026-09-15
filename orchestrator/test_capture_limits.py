from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from orchestrator.agent_workspace import _open_regular
from orchestrator.session_capture import SessionCapture
from orchestrator.session_tail import CaptureLimits
from orchestrator.session_usage import UsageAccumulator, summarize_usage
from orchestrator.test_session_capture import assistant, lines, usage


class IncrementalCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"

    def capture(self, **limits):
        return SessionCapture(self.root / "captures", backend="claude",
                              command=["claude", "--session-id", "main", "prompt"],
                              provider_home=self.home, context={}, limits=CaptureLimits(**limits))

    def write(self, text, name="main.jsonl"):
        path = self.home / ".claude/projects/p" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def finish(self, capture):
        capture._read_pipe("stdout", io.StringIO(lines({"type": "result", "usage": usage()})))
        capture.finish(exit_status=0)
        return json.loads((capture.root / "token-usage.json").read_text())

    def test_resume_and_live_poll_read_and_parse_each_byte_once(self):
        prior = lines(assistant("old"))
        path = self.write(prior)
        reads = []

        @contextmanager
        def counted(path):
            with _open_regular(path) as stream:
                class Reader:
                    def __getattr__(self, name):
                        return getattr(stream, name)

                    def read(self, size):
                        data = stream.read(size)
                        reads.append((stream.tell() - len(data), len(data), size))
                        return data

                yield Reader()

        with patch("orchestrator.session_tail._open_regular", counted):
            capture = self.capture()
            self.assertEqual(sum(row[1] for row in reads), len(prior))
            with patch.object(capture._usage, "feed", wraps=capture._usage.feed) as feed:
                for _ in range(20):
                    capture.sync_native()
                feed.assert_not_called()
                addition = lines(assistant("new"))
                with path.open("a") as stream:
                    stream.write(addition[:-2])
                capture.sync_native()
                self.assertFalse(capture.native)
                with path.open("a") as stream:
                    stream.write(addition[-2:])
                capture.sync_native()
                for _ in range(20):
                    capture.sync_native()
                self.assertEqual(feed.call_count, 1)
            self.assertEqual(sum(row[1] for row in reads), len(prior) + len(addition))
            self.assertTrue(all(0 < row[2] <= 256 * 1024 for row in reads))
        report = self.finish(capture)
        self.assertEqual(report["response_count"], 1)
        self.assertEqual(report["total"]["measurement"], "exact")

    def test_large_native_file_is_bounded_and_usage_is_partial(self):
        first = lines(assistant())
        capture = self.capture(file_bytes=len(first) + 100, line_bytes=2000)
        self.write(first + "X" * 100_000)
        capture.sync_native()
        consumed = capture._budget.used
        for _ in range(10):
            capture.sync_native()
        self.assertEqual(capture._budget.used, consumed)
        self.assertEqual(consumed, len(first) + 100)
        self.assertEqual(sum(map(len, capture.native.values())), len(first))
        report = self.finish(capture)
        self.assertEqual(report["total"]["measurement"], "partial")
        self.assertFalse(report["capture_complete"])
        self.assertIn("per_file_bytes_exceeded", str(report["capture_errors"]))
        footer = json.loads((capture.root / "conversation.jsonl").read_text().splitlines()[-1])
        self.assertFalse(footer["capture_complete"])

    def test_aggregate_budget_and_file_inventory_are_bounded(self):
        for kwargs in ({"total_bytes": 800}, {"files": 2}):
            with self.subTest(limits=kwargs):
                capture = self.capture(**kwargs)
                for n in range(12):
                    self.write(lines(assistant(f"child-{n}", agent=f"child-{n}")), f"main/subagents/{n}.jsonl")
                capture.sync_native()
                self.assertLessEqual(capture._budget.used, capture._budget.limits.total_bytes)
                self.assertLessEqual(len(capture._tails), capture._budget.limits.files)
                self.assertTrue(capture.errors)
                report = self.finish(capture)
                self.assertFalse(report["capture_complete"])

    def test_oversized_native_line_and_file_replacement_are_partial(self):
        capture = self.capture(line_bytes=512)
        path = self.write("X" * 50_000)
        capture.sync_native()
        self.assertFalse(capture.native)
        self.assertIn("native_line_bytes_exceeded", str(capture.errors))
        self.finish(capture)
        path.unlink()
        capture = self.capture()
        self.write(lines(assistant()))
        capture.sync_native()
        replacement = path.with_suffix(".new")
        replacement.write_text(lines(assistant("replacement")))
        replacement.replace(path)
        capture.sync_native()
        self.assertIn("replaced or truncated", str(capture.errors))
        self.assertEqual(self.finish(capture)["response_count"], 1)

    def test_output_limits_keep_draining_and_preserve_cli_exit_status(self):
        for limits in ({"line_bytes": 2000}, {"total_bytes": 2000}, {"records": 4}):
            with self.subTest(limits=limits):
                capture = self.capture(**limits)
                terminal = lines({"type": "result", "usage": usage()})
                code = (
                    "import sys\n"
                    "for i in range(64):\n"
                    " print('x'*4096, flush=True)\n"
                    " print('y'*4096, file=sys.stderr, flush=True)\n"
                    f"print({terminal!r}, end='', flush=True)\n"
                    "raise SystemExit(7)\n"
                )
                process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
                try:
                    capture.communicate(process, timeout=5)
                    self.assertEqual(process.returncode, 7)
                finally:
                    if process.poll() is None:
                        process.kill()
                        capture.communicate(process)
                capture.finish(exit_status=process.returncode)
                report = json.loads((capture.root / "token-usage.json").read_text())
                self.assertEqual(report["exit_status"], 7)
                self.assertFalse(report["capture_complete"])
                self.assertEqual(report["total"]["measurement"], "partial")
                self.assertLessEqual(capture._budget.used, capture._budget.limits.total_bytes)
                self.assertLessEqual(sum(map(len, capture._chunks.values())), capture._budget.limits.records)

    def test_malformed_counter_cannot_stop_pipe_drain(self):
        capture = self.capture(total_bytes=10)
        malformed = assistant()
        malformed["message"]["id"] = ["not-an-id"]
        capture._read_pipe("stdout", io.StringIO(lines(malformed, {"type": "result", "usage": usage()})))
        self.assertIn("usage_capture", str(capture.errors))
        capture.finish(exit_status=0)
        report = json.loads((capture.root / "token-usage.json").read_text())
        self.assertEqual(report["terminal"]["total_tokens"], 19)
        self.assertEqual(report["total"]["measurement"], "partial")


class IncrementalUsageTests(unittest.TestCase):
    def test_native_wins_even_when_stream_arrives_after_native(self):
        native = "provider/native/.claude/projects/p/main.jsonl"
        stream = "provider/stdout.stream-json"
        accumulator = UsageAccumulator("claude")
        accumulator.feed(native, lines(assistant()))
        accumulator.feed(stream, lines(assistant(counters=usage(output=0)), {"type": "result", "usage": usage()}))
        report = accumulator.report(finished=True)
        self.assertEqual(report["total"]["measurement"], "exact")
        self.assertEqual(report["total"]["total_tokens"], 19)
        self.assertEqual(report, summarize_usage("claude", lines({"type": "result", "usage": usage()}),
                                               {native: lines(assistant())}, finished=True))

    def test_usage_index_cap_marks_partial(self):
        accumulator = UsageAccumulator("claude", max_entries=2)
        for i in range(10):
            accumulator.feed("native", lines(assistant(str(i))))
        report = accumulator.report(finished=True)
        self.assertEqual(report["response_count"], 2)
        self.assertEqual(report["total"]["measurement"], "partial")
        self.assertIn("usage_index_limit_exceeded", report["missing_usage"])
