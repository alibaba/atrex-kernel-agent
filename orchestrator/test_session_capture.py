from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from orchestrator.agent_runtime.adapter import ClaudeAdapter
from orchestrator.agent_runtime.runtime import terminal_usage_from_stream
from orchestrator.session_capture import SessionCapture
from orchestrator.session_usage import summarize_usage


def usage(input=10, output=2, cached=3, write=4):
    return {
        "input_tokens": input,
        "output_tokens": output,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": write,
    }


def assistant(id="msg-main", *, agent=None, counters=None, text="full thinking and content"):
    event = {
        "type": "assistant",
        "message": {
            "id": id,
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": text},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "cat kernel.py"},
                },
            ],
            "usage": counters or usage(),
        },
    }
    if agent:
        event["agentId"] = agent
    return event


def lines(*events):
    return "".join(json.dumps(event) + "\n" for event in events)


class UsageTests(unittest.TestCase):
    def test_task_progress_counters_are_not_response_usage(self):
        stream = lines(
            assistant(),
            *[
                {"type": "system", "subtype": subtype, "usage": usage(1000, 200)}
                for subtype in ("task_progress", "task_notification")
            ],
            assistant("msg-child", agent="child"),
        )
        events, total = ClaudeAdapter().normalize_stream(stream)
        self.assertEqual(len([event for event in events if event.kind == "usage_delta"]), 2)
        self.assertEqual(total.total_tokens, 38)
        self.assertEqual(total.measurement, "partial")
        self.assertEqual(terminal_usage_from_stream(stream).total_tokens, 38)
        complete = stream + lines({"type": "result", "usage": usage(20, 4, 6, 8)})
        self.assertEqual(ClaudeAdapter().normalize_stream(complete)[1].measurement, "exact")
        self.assertEqual(terminal_usage_from_stream(complete).total_tokens, 38)
        report = summarize_usage(
            "claude",
            complete,
            {
                "main": lines(assistant()),
                "subagents/child": lines(assistant("msg-child", agent="child")),
            },
            finished=True,
        )
        self.assertEqual(report["response_count"], 2)
        self.assertEqual(report["total"]["total_tokens"], 38)
        self.assertEqual(report["total"]["measurement"], "exact")

    def test_progress_without_responses_does_not_fabricate_usage(self):
        stream = lines({"type": "system", "subtype": "task_progress", "usage": usage(1000)})
        events, total = ClaudeAdapter().normalize_stream(stream)
        self.assertEqual(events, ())
        self.assertIsNone(total.total_tokens)
        self.assertIsNone(terminal_usage_from_stream(stream).total_tokens)

    def test_ignoring_non_response_usage_keeps_tool_phase_receipts(self):
        receipt = "ATREX_TRACE_EVENT=" + json.dumps(
            {
                "schema": "atrex.iteration_trace.v1",
                "kind": "phase_marker",
                "action": "start",
                "phase": "benchmark",
                "marker_id": "marker-1",
            }
        )
        events, total = ClaudeAdapter().normalize_stream(
            lines(
                {
                    "type": "user",
                    "usage": usage(1000),
                    "message": {"content": [{"type": "tool_result", "content": receipt}]},
                }
            )
        )
        self.assertEqual(
            [(event.kind, event.marker_id) for event in events], [("phase_marker", "marker-1")]
        )
        self.assertIsNone(total.total_tokens)

    def test_native_last_response_usage_and_children_not_double_charged(self):
        for terminal in (usage(), usage(20, 4, 6, 8)):
            with self.subTest(terminal=terminal):
                stdout = lines(
                    assistant(counters=usage(output=0)), {"type": "result", "usage": terminal}
                )
                native = {
                    "provider/native/.claude/projects/p/main.jsonl": lines(
                        assistant(counters=usage(output=0)), assistant()
                    ),
                    "provider/native/.claude/projects/p/main/subagents/a.jsonl": lines(
                        assistant(agent="child"), assistant("msg-child", agent="child")
                    ),
                }
                report = summarize_usage("claude", stdout, native, finished=True)
                self.assertEqual(report["response_count"], 2)
                self.assertEqual(report["total"]["total_tokens"], 38)
                self.assertEqual(report["total"]["measurement"], "exact")
                self.assertEqual(report["by_agent"]["child"]["output_tokens"], 2)

    def test_resume_does_not_count_historical_context(self):
        path = "provider/native/.claude/projects/p/main.jsonl"
        prior = {path: lines(assistant())}
        native = {path: lines(assistant(), assistant("msg-new"))}
        report = summarize_usage(
            "claude",
            lines({"type": "result", "usage": usage()}),
            native,
            previous=prior,
            finished=True,
        )
        self.assertEqual(report["response_count"], 1)
        self.assertEqual(report["total"]["total_tokens"], 19)

    def test_unreconciled_missing_and_zero_are_distinct(self):
        missing = summarize_usage("claude", "", {}, finished=True)
        self.assertIsNone(missing["total"]["total_tokens"])
        report = summarize_usage(
            "claude",
            lines({"type": "result", "usage": usage(100)}),
            {"native": lines(assistant())},
            finished=True,
        )
        self.assertEqual(report["total"]["measurement"], "partial")
        self.assertEqual(report["total"]["total_tokens"], 109)  # not 109 + 19

    def test_codex_cumulative_resume_children_and_cache_buckets(self):
        def count(total, cached=2):
            return {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": total,
                            "output_tokens": 1,
                            "cached_input_tokens": cached,
                            "total_tokens": total + 1,
                        }
                    },
                },
            }

        root = "provider/native/.codex/sessions/root.jsonl"
        child = "provider/native/.codex/sessions/child.jsonl"
        report = summarize_usage(
            "codex",
            lines(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 20, "output_tokens": 1, "cached_input_tokens": 4},
                }
            ),
            {root: lines(count(20, 4), count(20, 4)), child: lines(count(5))},
            previous={root: lines(count(10))},
            finished=True,
        )
        self.assertEqual(report["total"]["total_tokens"], 16)
        self.assertEqual(report["total"]["input_tokens"], 11)
        self.assertEqual(report["total"]["cache_read_tokens"], 4)
        self.assertEqual(report["response_count"], 2)

    def test_qoder_credits_are_not_token_estimates(self):
        report = summarize_usage(
            "qodercli", lines({"type": "result", "credits": 1.25}), {}, finished=True
        )
        self.assertIsNone(report["total"]["total_tokens"])
        self.assertEqual(report["provider_credits"]["credits"], 1.25)


class CaptureTests(unittest.TestCase):
    def capture(self, root, **kwargs):
        return SessionCapture(
            root / "captures",
            backend="claude",
            command=["claude", "--session-id", "main", "the full initial prompt"],
            provider_home=root / "home",
            context={"attempt": "e0001"},
            **kwargs,
        )

    def write(self, root, name, text):
        path = root / "home/.claude/projects/project" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_live_full_copy_many_native_children_without_session_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.capture(root)
            self.write(
                root,
                "main.jsonl",
                lines(
                    assistant(),
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-1",
                                    "content": "kernel source\nfull result",
                                }
                            ]
                        },
                    },
                ),
            )
            for index in range(12):
                self.write(
                    root,
                    f"main/subagents/agent-{index}.jsonl",
                    lines(
                        assistant(
                            f"msg-child-{index}",
                            agent=f"child-{index}",
                            text=f"child reasoning {index}",
                        )
                    ),
                )
            capture.sync_native()
            live = (capture.root / "conversation.jsonl").read_text()
            self.assertIn("child reasoning 11", live)
            self.assertIn("kernel source", live)
            self.assertIn("the full initial prompt", live)
            capture._chunks["stdout"].append(lines({"type": "result", "usage": usage()}))
            capture.finish(exit_status=0)
            rows = list(
                map(json.loads, (capture.root / "conversation.jsonl").read_text().splitlines())
            )
            self.assertEqual(rows[-1]["state"], "completed")
            self.assertEqual([r["sequence"] for r in rows], list(range(len(rows))))
            report = json.loads((capture.root / "token-usage.json").read_text())
            self.assertEqual(report["response_count"], 13)
            self.assertEqual(report["total"]["total_tokens"], 247)
            self.assertEqual(len(list((capture.root / "provider/native").rglob("*.jsonl"))), 13)

    def test_resume_raw_delta_partial_line_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write(root, "main.jsonl", lines(assistant()))
            capture = self.capture(root)
            new = lines(assistant("new"))
            with path.open("a") as out:
                out.write(new[:-2])
            capture.sync_native()
            self.assertFalse(capture.native)
            with path.open("a") as out:
                out.write(new[-2:])
            secret = root / "private.jsonl"
            secret.write_text("do not capture credentials")
            path.with_name("unsafe.jsonl").symlink_to(secret)
            capture.sync_native()
            capture.sync_native()
            self.assertEqual(len(capture.native), 1)
            self.assertEqual(next(iter(capture.native.values())).decode(), new)
            capture.finish(exit_status=1)
            self.assertNotIn(
                "do not capture credentials", (capture.root / "conversation.jsonl").read_text()
            )
            report = json.loads((capture.root / "token-usage.json").read_text())
            self.assertEqual(report["total"]["measurement"], "partial")
            self.assertTrue(report["capture_errors"])

    def test_pipe_tee_and_timeout_keep_full_text_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.capture(root)
            code = (
                "import sys,time; print('full output',flush=True); "
                "print('failure details',file=sys.stderr,flush=True); time.sleep(30)"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    capture.communicate(process, timeout=0.2)
            finally:
                process.kill()
                stdout, stderr = capture.communicate(process)
                capture.finish(exit_status=process.returncode, timed_out=True)
            self.assertEqual(stdout, "full output\n")
            self.assertEqual(stderr, "failure details\n")
            rows = list(
                map(json.loads, (capture.root / "conversation.jsonl").read_text().splitlines())
            )
            self.assertEqual(rows[-1]["state"], "timed_out")
            self.assertIn("failure details", (capture.root / "conversation.jsonl").read_text())

    def test_provider_thinking_estimates_filtered_but_reasoning_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.capture(root)
            stream = lines(
                {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 1}, assistant()
            )
            process = subprocess.Popen(
                [sys.executable, "-c", "print(" + repr(stream) + ",end='')"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, _ = capture.communicate(process)
            capture.finish(exit_status=0)
            self.assertEqual(stdout, stream)
            for path in (
                capture.root / "conversation.jsonl",
                capture.root / "provider/stdout.stream-json",
            ):
                self.assertNotIn("estimated_tokens", path.read_text())
                self.assertIn("full thinking and content", path.read_text())


if __name__ == "__main__":
    unittest.main()
