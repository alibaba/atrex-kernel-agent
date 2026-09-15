"""Capture sink failures must never inject EPIPE into a running CLI."""

from __future__ import annotations

import errno
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from orchestrator.agent_runtime import process as agent_process
from orchestrator.session_capture import SessionCapture
from orchestrator.session_tail import CaptureLimits
from orchestrator.test_session_capture import assistant, lines, usage


class FailingSink:
    def __init__(self, stream, failure):
        self.stream = stream
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, value):
        if self.failure in {"write", "write-close"}:
            raise OSError(errno.ENOSPC, "simulated full capture disk")
        return self.stream.write(value)

    def flush(self):
        if self.failure == "flush":
            raise OSError(errno.EIO, "simulated capture flush failure")
        self.stream.flush()

    def close(self):
        self.stream.close()
        if self.failure in {"close", "write-close"}:
            raise OSError(errno.EIO, "simulated capture close failure")


class CaptureFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def capture(self, *, backend="claude", limits=None):
        return SessionCapture(
            self.root / "captures",
            backend=backend,
            command=[backend, "--session-id", "main", "initial prompt"],
            provider_home=None,
            context={"attempt": "test"},
            limits=limits,
        )

    @contextmanager
    def fail_sinks(self, capture, failures):
        original = Path.open
        opens = {name: 0 for name in failures}

        def open_file(path, mode="r", *args, **kwargs):
            relative = (
                path.relative_to(capture.root).as_posix()
                if path.is_relative_to(capture.root)
                else ""
            )
            failure = failures.get(relative) if mode == "a" else None
            if failure:
                opens[relative] += 1
                if failure == "open":
                    raise PermissionError(errno.EACCES, "simulated capture open failure")
                return FailingSink(original(path, mode, *args, **kwargs), failure)
            return original(path, mode, *args, **kwargs)

        with patch.object(Path, "open", open_file):
            yield opens

    def command(self, exit_status=0):
        # Each pipe exceeds its OS buffer. A non-draining reader would block the
        # process; a prematurely closed reader would give it BrokenPipeError.
        terminal = lines(assistant(), {"type": "result", "usage": usage()})
        code = (
            "import sys\nfor i in range(32):\n"
            " print('O'*4096 + '-' + str(i), flush=True)\n"
            " print('E'*4096 + '-' + str(i), file=sys.stderr, flush=True)\n"
            f"print({terminal!r}, end='', flush=True)\nraise SystemExit({exit_status})\n"
        )
        stdout = "".join("O" * 4096 + f"-{i}\n" for i in range(32)) + terminal
        stderr = "".join("E" * 4096 + f"-{i}\n" for i in range(32))
        return [sys.executable, "-c", code], stdout, stderr

    def run_capture(self, capture, command):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            stdout, stderr = capture.communicate(process, timeout=5)
            return stdout, stderr, process.returncode
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
                capture.communicate(process)

    def assert_final_capture_is_marked(self, capture, exit_status):
        capture.finish(exit_status=exit_status)
        report = json.loads((capture.root / "token-usage.json").read_text())
        self.assertEqual(report["exit_status"], exit_status)
        self.assertEqual(report["state"], "completed" if exit_status == 0 else "failed")
        self.assertTrue(report["capture_errors"])
        self.assertEqual(report["total"]["measurement"], "partial")
        rows = [
            json.loads(line)
            for line in (capture.root / "conversation.jsonl").read_text().splitlines()
        ]
        self.assertEqual(rows[-1]["state"], report["state"])
        self.assertIn("E" * 4096 + "-31", (capture.root / "conversation.jsonl").read_text())

    def test_raw_open_write_flush_and_close_failures_keep_both_pipes_draining(self):
        command, expected_out, expected_err = self.command()
        for name in ("provider/stdout.stream-json", "provider/stderr.log"):
            for failure in ("open", "write", "flush", "close", "write-close"):
                with self.subTest(name=name, failure=failure):
                    capture = self.capture()
                    try:
                        with self.fail_sinks(capture, {name: failure}) as opens:
                            stdout, stderr, status = self.run_capture(capture, command)
                        self.assertEqual(status, 0, stderr[-1000:])
                        self.assertEqual((stdout, stderr), (expected_out, expected_err))
                        self.assertEqual(opens[name], 1)
                        other = (
                            "provider/stderr.log"
                            if "stdout" in name
                            else "provider/stdout.stream-json"
                        )
                        self.assertEqual(
                            (capture.root / other).read_text(),
                            expected_err if "stdout" in name else expected_out,
                        )
                        self.assert_final_capture_is_marked(capture, 0)
                    finally:
                        capture.finish(exit_status=0)

    def test_conversation_failure_does_not_disable_raw_capture(self):
        command, expected_out, expected_err = self.command()
        for failure in ("open", "write", "close"):
            with self.subTest(failure=failure):
                capture = self.capture()
                try:
                    with self.fail_sinks(capture, {"conversation.jsonl": failure}) as opens:
                        stdout, stderr, status = self.run_capture(capture, command)
                    self.assertEqual((stdout, stderr, status), (expected_out, expected_err, 0))
                    self.assertLessEqual(
                        opens["conversation.jsonl"], 2
                    )  # One failure per pipe reader.
                    self.assertEqual(
                        (capture.root / "provider/stdout.stream-json").read_text(), expected_out
                    )
                    self.assertEqual(
                        (capture.root / "provider/stderr.log").read_text(), expected_err
                    )
                    self.assert_final_capture_is_marked(capture, 0)
                finally:
                    capture.finish(exit_status=0)

    def test_all_sinks_failing_preserves_real_cli_exit_status(self):
        for status in (0, 7):
            with self.subTest(status=status):
                capture = self.capture()
                command, expected_out, expected_err = self.command(status)
                try:
                    with self.fail_sinks(
                        capture,
                        {
                            "provider/stdout.stream-json": "open",
                            "provider/stderr.log": "write-close",
                            "conversation.jsonl": "write",
                        },
                    ):
                        result = self.run_capture(capture, command)
                    self.assertEqual(result, (expected_out, expected_err, status))
                    self.assert_final_capture_is_marked(capture, status)
                finally:
                    capture.finish(exit_status=status)

    def test_timeout_still_uses_existing_process_lifecycle(self):
        capture = self.capture()
        command = [
            sys.executable,
            "-c",
            "import time; print('start', flush=True); time.sleep(.05); print('still running', flush=True); time.sleep(30)",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            with self.fail_sinks(
                capture, {"provider/stdout.stream-json": "write", "conversation.jsonl": "open"}
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    capture.communicate(process, timeout=0.3)
                self.assertIsNone(process.poll())
                process.kill()
                stdout, _ = capture.communicate(process)
            self.assertEqual(stdout, "start\nstill running\n")
            capture.finish(exit_status=process.returncode, timed_out=True)
            report = json.loads((capture.root / "token-usage.json").read_text())
            self.assertEqual(report["state"], "timed_out")
            self.assertTrue(report["capture_errors"])
        finally:
            if process.poll() is None:
                process.kill()
                capture.communicate(process)
            capture.finish(exit_status=process.returncode, timed_out=True)

    def test_codex_identity_recursion_after_capture_limit_preserves_exit_and_usage(self):
        terminal = lines(
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}
        )
        for limits in ({"file_bytes": 4096}, {"total_bytes": 4096}, {"records": 1}):
            for status in (0, 7):
                with self.subTest(limits=limits, status=status):
                    capture = self.capture(backend="codex", limits=CaptureLimits(**limits))
                    code = (
                        "import sys\nprint('preamble', flush=True)\n"
                        "for i in range(128):\n print('x'*4096, flush=True)\n"
                        f"print({terminal!r}, end='', flush=True)\nraise SystemExit({status})\n"
                    )
                    with (
                        patch(
                            "orchestrator.agent_runtime.codex_ledger.codex_thread_id_from_stream",
                            side_effect=RecursionError("deep provider JSON"),
                        ) as parse,
                        # Make the first identity parse happen in the pipe's
                        # overflow path, not opportunistically in the monitor.
                        patch.object(capture, "_poll"),
                    ):
                        _, _, actual = self.run_capture(capture, [sys.executable, "-c", code])
                        self.assertEqual(actual, status)
                        capture.finish(exit_status=actual)
                        self.assertEqual(
                            sum(call.args == ("preamble\n",) for call in parse.call_args_list), 1
                        )
                    report = json.loads((capture.root / "token-usage.json").read_text())
                    self.assertEqual(report["exit_status"], status)
                    self.assertEqual(report["total"]["measurement"], "partial")
                    self.assertEqual(report["terminal"]["total_tokens"], 12)
                    self.assertFalse(report["capture_complete"])
                    self.assertIn(
                        "codex_session_identity:RecursionError", str(report["capture_errors"])
                    )
                    self.assertNotIn("stdout_pipe_read", str(report["capture_errors"]))

    def test_unexpected_capture_error_switches_to_drain_only_for_both_pipes(self):
        for status in (0, 7):
            with self.subTest(status=status):
                capture = self.capture()
                command, _, _ = self.command(status)
                with patch.object(
                    capture._budget, "retain", side_effect=RuntimeError("accounting failure")
                ) as retain:
                    _, _, actual = self.run_capture(capture, command)
                    self.assertEqual(actual, status)
                    # Stop processing the affected stream after the first error;
                    # subsequent reads only discard bytes until EOF.
                    self.assertEqual(retain.call_count, 2)
                capture.finish(exit_status=actual)
                report = json.loads((capture.root / "token-usage.json").read_text())
                self.assertFalse(report["capture_complete"])
                for name in ("stdout", "stderr"):
                    self.assertIn(
                        f"{name}_capture_processing:RuntimeError", str(report["capture_errors"])
                    )
                    self.assertNotIn(f"{name}_pipe_read", str(report["capture_errors"]))

    def test_filter_and_projection_recursion_keep_draining_to_eof(self):
        for target in ("record_provider_line", "SessionCapture._append"):
            with self.subTest(target=target):
                capture = self.capture()
                command, stdout, stderr = self.command(7)
                with patch(
                    f"orchestrator.session_capture.{target}",
                    side_effect=RecursionError("deep provider JSON"),
                ):
                    self.assertEqual(self.run_capture(capture, command), (stdout, stderr, 7))
                self.assert_final_capture_is_marked(capture, 7)

    def test_codex_identity_recovers_from_bad_record_without_replaying_it_at_finish(self):
        from orchestrator.agent_runtime.codex_ledger import codex_thread_id_from_stream
        from orchestrator.test_unsandboxed_usage import ROOT_ID, codex_stream, count, metadata

        capture = SessionCapture(
            self.root / "captures",
            backend="codex",
            command=["codex", "exec", "prompt"],
            provider_home=None,
            native_environment={"HOME": str(self.root / "home")},
            context={},
        )
        native = self.root / "home/.codex/sessions/root.jsonl"
        native.parent.mkdir(parents=True)
        native.write_text(lines(metadata(ROOT_ID), count()))

        def parse(text):
            if text == "bad record\n":
                raise RecursionError("deep provider JSON")
            return codex_thread_id_from_stream(text)

        with patch(
            "orchestrator.agent_runtime.codex_ledger.codex_thread_id_from_stream", side_effect=parse
        ) as parser:
            capture._read_pipe("stdout", io.StringIO("bad record\n" + codex_stream()))
            capture.sync_native()
            self.assertEqual(capture.session_id, ROOT_ID)
            self.assertEqual(capture._host_transcripts.session_id, ROOT_ID)
            self.assertTrue(capture.native)
            capture.finish(exit_status=0)
            self.assertEqual(
                sum(call.args == ("bad record\n",) for call in parser.call_args_list), 1
            )
        report = json.loads((capture.root / "token-usage.json").read_text())
        self.assertEqual(report["total"]["total_tokens"], 110)
        self.assertEqual(report["total"]["measurement"], "partial")
        self.assertEqual(report["session_id"], ROOT_ID)

    def test_process_guard_preserves_outputs_even_if_final_capture_cannot_be_written(self):
        capture = self.capture()
        command, expected_out, expected_err = self.command(7)

        def spawn(argv, **kwargs):
            return subprocess.Popen(
                argv,
                cwd=kwargs["cwd"],
                env=kwargs["environment"],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                text=True,
            )

        with (
            patch("orchestrator.session_capture.start_session_capture", return_value=capture),
            patch.object(agent_process, "spawn_owned_session", side_effect=spawn),
            patch.object(agent_process, "dependency_guard"),
            self.fail_sinks(
                capture, {"provider/stdout.stream-json": "write", "conversation.jsonl": "open"}
            ),
            patch(
                "orchestrator.session_capture._atomic",
                side_effect=OSError(errno.ENOSPC, "disk full"),
            ),
            redirect_stdout(io.StringIO()) as warnings,
        ):
            result = agent_process.run_bounded(command, self.root, timeout=5)
        self.assertEqual(result, (expected_out, expected_err, 7, False))
        self.assertIn("session capture failed", warnings.getvalue())
        self.assertTrue(capture.errors)


if __name__ == "__main__":
    unittest.main()
