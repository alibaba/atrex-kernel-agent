"""PR1 contract: observe existing invocations; do not require a new workflow."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from long_horizon.session import LongSessionRunner
from orchestrator.agent_runtime.adapter import ClaudeAdapter, CodexAdapter, QoderAdapter
from orchestrator.agent_runtime.codex_ledger import CodexSessionLedgerObserver, observe_codex_usage
from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)
from orchestrator.agent_runtime.process import run_bounded
from orchestrator.session_capture import SessionCapture, captured_observation, clear_capture
from orchestrator.test_session_capture import assistant, lines, usage
from orchestrator.test_unsandboxed_usage import ROOT_ID, count, metadata


class SessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(clear_capture)
        clear_capture()
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.environment = {
            **os.environ,
            "HOME": str(self.home),
            "CLAUDE_CONFIG_DIR": str(self.home / ".claude"),
            "ATREX_AGENT_CLI": "claude",
        }
        self.environment.pop("ATREX_SESSION_CAPTURE_DIR", None)

    def run_cli(self, code, *, timeout=5, environment=None):
        command = [sys.executable, "-c", code, "--session-id", "main", "exact initial prompt"]
        with patch("orchestrator.agent_runtime.process.dependency_guard"):
            return run_bounded(command, self.workspace, timeout, environment or self.environment)

    def reports(self, root=None):
        return sorted(
            (root or self.root / ".atrex-session-traces/workspace").glob("run-*/token-usage.json")
        )

    def test_default_capture_is_outside_workspace_and_preserves_result(self):
        stream = lines(assistant(), {"type": "result", "usage": usage()})
        result = self.run_cli(
            f"import sys; print({stream!r}, end=''); print('stderr', file=sys.stderr); sys.exit(7)"
        )
        self.assertEqual(result, (stream, "stderr\n", 7, False))
        self.assertEqual(list(self.workspace.iterdir()), [])
        (report_path,) = self.reports()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["exit_status"], 7)
        self.assertEqual(report["total"]["total_tokens"], 19)
        self.assertEqual(report["total"]["measurement"], "partial")
        conversation = (report_path.parent / "conversation.jsonl").read_text()
        self.assertIn("exact initial prompt", conversation)
        self.assertIn("stderr", conversation)
        self.assertEqual(report_path.parent.stat().st_mode & 0o777, 0o700)

    def test_capture_setup_failure_does_not_change_cli_result(self):
        with (
            patch("orchestrator.session_capture.SessionCapture", side_effect=OSError("disk full")),
            redirect_stdout(io.StringIO()) as warnings,
        ):
            self.assertEqual(self.run_cli("print('ok')"), ("ok\n", "", 0, False))
        self.assertIn("session capture setup failed", warnings.getvalue())

    def test_timeout_still_terminates_and_records_invocation(self):
        result = self.run_cli(
            "import time; print('started', flush=True); time.sleep(30)", timeout=0.1
        )
        self.assertTrue(result[3])
        self.assertNotEqual(result[2], 0)
        (report_path,) = self.reports()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["state"], "timed_out")
        self.assertEqual(report["total"]["measurement"], "unavailable")

    def test_non_agent_commands_do_not_create_capture(self):
        with patch("orchestrator.agent_runtime.process.dependency_guard"):
            self.assertEqual(
                run_bounded([sys.executable, "-c", "print('ok')"], self.workspace, 5, {}),
                ("ok\n", "", 0, False),
            )
        self.assertFalse(self.reports())

    def test_long_horizon_resume_has_separate_prompts_deltas_and_invocation_ids(self):
        destination = self.root / "episode/sessions"
        native = self.home / ".claude/projects/p/main.jsonl"
        native.parent.mkdir(parents=True)
        invocation = 0

        def command(prompt, *args):
            nonlocal invocation
            invocation += 1
            payload = lines(assistant(f"response-{invocation}"))
            stream = payload + lines({"type": "result", "usage": usage()})
            code = (
                "from pathlib import Path\n"
                f"with Path({str(native)!r}).open('a') as sink: sink.write({payload!r})\n"
                f"print({stream!r}, end='')\n"
            )
            return [sys.executable, "-c", code, "--session-id", "main", prompt]

        with (
            patch("long_horizon.main_adapter.session_environment", return_value=self.environment),
            patch("long_horizon.main_adapter.fresh_session_command", side_effect=command),
            patch("long_horizon.main_adapter.resume_session_command", side_effect=command),
            patch("orchestrator.agent_runtime.process.dependency_guard"),
        ):
            result = LongSessionRunner().run(
                self.workspace,
                "initial prompt",
                handoff_path=self.workspace / "handoff.json",
                handoff_resumes=1,
                session_id="main",
                completion_check=lambda _: "",
                telemetry_environment={
                    "ATREX_SESSION_CAPTURE_DIR": str(destination),
                    "ATREX_TELEMETRY_ITERATION_ID": "episode-0001",
                    "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
                },
            )
        self.assertEqual(result.resume_count, 1)
        self.assertEqual(result.tokens, 38)
        reports = [json.loads(path.read_text()) for path in self.reports(destination)]
        self.assertEqual(len(reports), 2)
        self.assertEqual(
            {row["context"]["attempt_id"] for row in reports}, {"invocation-1", "invocation-2"}
        )
        self.assertTrue(all(row["total"]["total_tokens"] == 19 for row in reports))
        self.assertTrue(all(row["total"]["measurement"] == "exact" for row in reports))
        self.assertEqual({row["response_count"] for row in reports}, {1})
        prompts = [
            (path.parent / "conversation.jsonl").read_text() for path in self.reports(destination)
        ]
        self.assertTrue(any("initial prompt" in value for value in prompts))
        self.assertTrue(any("handoff" in value for value in prompts))

    def test_captured_totals_do_not_reorder_existing_phase_events(self):
        stream = lines(assistant(), {"type": "result", "usage": usage()})
        capture = SessionCapture(
            self.root / "captures",
            backend="claude",
            command=["claude", "prompt"],
            provider_home=None,
            context={},
        )
        capture._chunks["stdout"].append(stream)
        capture.finish(exit_status=0)
        delta = TokenUsage(10, 2, 3, 4, 19, "exact")
        ordered = (
            NormalizedAgentEvent(
                0, "phase_marker", phase="research", action="start", marker_id="start"
            ),
            NormalizedAgentEvent(1, "usage_delta", usage=delta),
            NormalizedAgentEvent(
                2, "phase_marker", phase="research", action="end", marker_id="end"
            ),
        )
        observed = captured_observation(stream, ordered, AgentRuntimeCapabilities(True, True, True))
        self.assertEqual(observed.events[:-1], ordered)

    def test_latest_response_usage_keeps_original_stream_position(self):
        start = {
            "schema": "atrex.iteration_trace.v1",
            "kind": "phase_marker",
            "phase": "research",
            "action": "start",
            "marker_id": "start",
        }
        stream = lines(
            assistant(counters=usage(output=0)),
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "ATREX_TRACE_EVENT=" + json.dumps(start)}
                    ]
                },
            },
            assistant(counters=usage(output=2)),
        )
        events, total = ClaudeAdapter().normalize_stream(stream)
        self.assertEqual([event.kind for event in events], ["usage_delta", "phase_marker"])
        self.assertEqual(events[0].usage.output_tokens, 2)
        self.assertEqual(total.total_tokens, 19)

    def test_codex_ledger_phase_order_survives_native_child_accounting(self):
        from orchestrator.session_capture import CapturedObservation

        home = self.root / "codex"
        rollout = home / f"sessions/rollout-{ROOT_ID}.jsonl"
        rollout.parent.mkdir(parents=True)
        marker = {
            "schema": "atrex.iteration_trace.v1",
            "kind": "phase_marker",
            "phase": "research",
            "action": "start",
            "marker_id": "start",
        }
        rollout.write_text(
            lines(
                metadata(ROOT_ID),
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "output": "ATREX_TRACE_EVENT=" + json.dumps(marker),
                    },
                },
                count(),
            )
        )
        root_usage = TokenUsage(100, 10, 20, 0, 110, "exact")
        combined = TokenUsage(96, 12, 24, 0, 132, "exact")
        captured = CapturedObservation(
            (), combined, AgentRuntimeCapabilities(True, True, True, True), (), True
        )
        events, total, _, _ = observe_codex_usage(
            CodexSessionLedgerObserver(home), ROOT_ID, root_usage, captured=captured
        )
        self.assertEqual(
            [event.kind for event in events], ["usage_delta", "phase_marker", "terminal_usage"]
        )
        self.assertEqual(events[0].usage.total_tokens, 110)
        self.assertEqual(total.total_tokens, 132)

    def test_adapter_options_keep_old_workflow_and_allow_qoder_native_capture(self):
        self.assertNotIn(
            "--skip-git-repo-check", CodexAdapter().build_command("prompt", "id", "max", "")
        )
        self.assertNotIn(
            "--no-session-persistence", QoderAdapter().build_command("prompt", "id", "max", "")
        )


if __name__ == "__main__":
    unittest.main()
