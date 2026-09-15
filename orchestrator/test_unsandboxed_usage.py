from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from long_horizon.main_adapter import normalize_stream
from orchestrator.agent_runtime.codex_ledger import CodexSessionLedgerObserver
from orchestrator.agent_runtime.model import AgentRunRequest, AgentRuntimeCapabilities
from orchestrator.agent_runtime.runtime import CodexRuntime
from orchestrator.session_capture import SessionCapture, captured_observation, clear_capture
from orchestrator.supervisor_runtime import SupervisorRuntime, SupervisorRuntimeConfig, activate_supervisor_runtime
from orchestrator.test_session_capture import assistant, lines, usage

ROOT_ID = "11111111-1111-1111-1111-111111111111"
CHILD_ID = "22222222-2222-2222-2222-222222222222"


def count(inputs=100, outputs=10, cached=20, *, last=None):
    counters = {"input_tokens": inputs, "output_tokens": outputs,
                "cached_input_tokens": cached, "total_tokens": inputs + outputs}
    return {"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": counters, "last_token_usage": last or counters,
    }}}


def metadata(identity, parent=None):
    body = {"id": identity, "cwd": "/test"}
    if parent:
        body["source"] = {"subagent": {"thread_spawn": {"parent_thread_id": parent}}}
    return {"type": "session_meta", "payload": body}


def codex_stream(inputs=100, outputs=10, cached=20):
    return lines({"type": "thread.started", "thread_id": ROOT_ID}, {
        "type": "turn.completed", "usage": {
            "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
        },
    })


class UnsandboxedUsageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(clear_capture)
        clear_capture()
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.environment = {"HOME": str(self.home)}

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def capture(self, backend="claude", command=None, **kwargs):
        return SessionCapture(
            self.root / "captures", backend=backend,
            command=command or [backend, "--session-id", "main", "prompt"],
            provider_home=None, native_environment=self.environment, context={}, **kwargs,
        )

    def finish(self, capture, stream):
        capture._chunks["stdout"].append(stream)
        capture.finish(exit_status=0)
        return json.loads((capture.root / "token-usage.json").read_text())

    def test_none_supervisor_captures_configured_claude_home_and_children(self):
        configured = self.root / "custom-claude"
        self.environment.update(CLAUDE_CONFIG_DIR=str(configured), ATREX_AGENT_CLI="claude")
        workspace = self.root / "workspace"
        workspace.mkdir()
        runtime = SupervisorRuntime(SupervisorRuntimeConfig(
            repository_root=Path(__file__).resolve().parents[1], hardware="test", agent_sandbox="none", sandbox_timeout=5,
        ))
        runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))
        try:
            lease = runtime.prepare_session(["claude", "--session-id", "main", "prompt"], workspace, self.environment)
            self.assertIsNone(lease.capture.home)
            self.assertEqual(lease.environment["CLAUDE_CONFIG_DIR"], str(configured))
            self.write(configured / "projects/p/main.jsonl", lines(assistant()))
            self.write(configured / "projects/p/main/subagents/child.jsonl", lines(assistant("child-msg", agent="child")))
            self.write(configured / "projects/p/other.jsonl", lines(assistant("unrelated")))
            self.write(configured / "auth.json", '{"secret":"not a transcript"}')
            report = self.finish(lease.capture, lines(assistant(), {"type": "result", "usage": usage()}))
            self.assertEqual(report["total"]["measurement"], "exact")
            self.assertEqual(report["total"]["total_tokens"], 38)
            self.assertEqual(report["response_count"], 2)
            self.assertEqual(len(lease.capture.native), 2)
            self.assertNotIn("unrelated", (lease.capture.root / "conversation.jsonl").read_text())
            lease.close()
        finally:
            runtime._server = None
            runtime.close()

    def test_shared_claude_resume_excludes_old_history_and_unrelated_sessions(self):
        native = self.home / ".claude/projects/p/main.jsonl"
        self.write(native, lines(assistant("old")))
        capture = self.capture()
        self.write(native, lines(assistant("old"), assistant("new")))
        self.write(native.parent / "other.jsonl", lines(assistant("other")))
        report = self.finish(capture, lines(assistant("new"), {"type": "result", "usage": usage()}))
        self.assertEqual(report["total"]["measurement"], "exact")
        self.assertEqual(report["response_count"], 1)
        self.assertEqual(report["total"]["total_tokens"], 19)

    def test_shared_home_session_directory_symlink_is_not_captured(self):
        capture = self.capture()
        secret = self.root / "private"
        self.write(secret / "child.jsonl", lines(assistant("secret")))
        self.write(self.home / ".claude/projects/p/main.jsonl", lines(assistant()))
        (self.home / ".claude/projects/p/main").symlink_to(secret, target_is_directory=True)
        report = self.finish(capture, lines(assistant(), {"type": "result", "usage": usage()}))
        self.assertEqual(report["response_count"], 1)
        self.assertNotIn("secret", (capture.root / "conversation.jsonl").read_text())

    def test_none_missing_native_usage_remains_partial(self):
        report = self.finish(self.capture(), lines(assistant(), {"type": "result", "usage": usage()}))
        self.assertEqual(report["total"]["measurement"], "partial")

    def test_shared_claude_unreconciled_native_usage_remains_partial(self):
        capture = self.capture()
        self.write(self.home / ".claude/projects/p/main.jsonl", lines(assistant()))
        report = self.finish(capture, lines({"type": "result", "usage": usage(1000)}))
        self.assertEqual(report["total"]["measurement"], "partial")
        self.assertTrue(report["warnings"])

    def test_qoder_and_pi_capture_only_the_requested_session(self):
        for backend, relative in (("qodercli", ".qoder/tasks/main.jsonl"), ("pi", ".pi/agent/sessions/p/date_main.jsonl")):
            with self.subTest(backend=backend):
                capture = self.capture(backend)
                path = self.write(self.home / relative, lines(assistant()))
                self.write(path.parent / "unrelated.jsonl", lines(assistant("other")))
                report = self.finish(capture, lines({"type": "result", "usage": usage(), "credits": 1.25}))
                self.assertEqual(report["response_count"], 1)
                if backend == "qodercli":
                    self.assertEqual(report["provider_credits"]["measurement"], "exact")

    def test_codex_captures_current_thread_family_and_resume_deltas(self):
        self.environment["CODEX_HOME"] = str(self.root / "custom-codex")
        sessions = Path(self.environment["CODEX_HOME"]) / "sessions"
        root = sessions / f"rollout-{ROOT_ID}.jsonl"
        child = sessions / f"rollout-{CHILD_ID}.jsonl"
        self.write(sessions / "unrelated.jsonl", lines(metadata("other"), count(900)))
        capture = self.capture("codex", ["codex", "exec", "prompt"])
        self.write(root, lines(metadata(ROOT_ID), count()))
        self.write(child, lines(metadata(CHILD_ID, ROOT_ID), count(20, 2, 4)))
        report = self.finish(capture, codex_stream())
        self.assertEqual(report["total"]["measurement"], "exact")
        self.assertEqual(report["total"]["total_tokens"], 132)
        self.assertEqual(report["total"]["input_tokens"], 96)
        self.assertEqual(report["total"]["cache_read_tokens"], 24)
        self.assertEqual(len(capture.native), 2)
        # Thread is learned from stdout after the resume process starts; even
        # then the initial byte offsets must exclude its pre-invocation counters.
        resumed = self.capture("codex", ["codex", "exec", "prompt"])
        delta = count(10, 1, 2)["payload"]["info"]["total_token_usage"]
        self.write(root, root.read_text() + lines(count(110, 11, 22, last=delta)))
        report = self.finish(resumed, codex_stream(110, 11, 22))
        self.assertEqual(report["total"]["measurement"], "exact")
        self.assertEqual(report["total"]["total_tokens"], 11)
        self.assertEqual(report["response_count"], 1)

    def test_complete_codex_family_capture_wins_over_root_only_ledger(self):
        capture = self.capture("codex", ["codex", "exec", "prompt"])
        sessions = self.home / ".codex/sessions"
        self.write(sessions / f"rollout-{ROOT_ID}.jsonl", lines(metadata(ROOT_ID), count()))
        self.write(sessions / "child.jsonl", lines(metadata(CHILD_ID, ROOT_ID), count(20, 2, 4)))
        self.finish(capture, codex_stream())
        observer = CodexSessionLedgerObserver(self.home / ".codex")
        _, total, _, _ = normalize_stream("codex", codex_stream(), session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "exact")
        self.assertEqual(total.total_tokens, 132)
        self.assertEqual(observer._session_usage.total_tokens, 110)

        # A later stream-only capture must use only the new ledger delta, not
        # replay the first invocation that was accounted from the native family.
        delta = count(10, 1, 2)["payload"]["info"]["total_token_usage"]
        root = sessions / f"rollout-{ROOT_ID}.jsonl"
        self.write(root, root.read_text() + lines(count(110, 11, 22, last=delta)))
        capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
        self.finish(capture, codex_stream(110, 11, 22))
        _, total, _, _ = normalize_stream("codex", codex_stream(110, 11, 22), session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "exact")
        self.assertEqual(total.total_tokens, 11)

    def test_partial_codex_family_is_not_replaced_by_root_only_ledger(self):
        capture = self.capture("codex", ["codex", "exec", "prompt"])
        sessions = self.home / ".codex/sessions"
        self.write(sessions / f"rollout-{ROOT_ID}.jsonl", lines(metadata(ROOT_ID), count()))
        self.write(sessions / "child.jsonl", lines(metadata(CHILD_ID, ROOT_ID), count(20, 2, 4)) + '{"incomplete":')
        self.finish(capture, codex_stream())
        observer = CodexSessionLedgerObserver(self.home / ".codex")
        _, total, _, errors = normalize_stream("codex", codex_stream(), session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "partial")
        self.assertEqual(total.total_tokens, 132)
        self.assertIn("codex_native_usage_incomplete_or_unreconciled", errors)

    def test_failed_cursor_advance_cannot_replay_an_accounted_invocation(self):
        capture = self.capture("codex", ["codex", "exec", "prompt"])
        root = self.home / f".codex/sessions/rollout-{ROOT_ID}.jsonl"
        self.write(root, lines(metadata(ROOT_ID), count()))
        self.finish(capture, codex_stream())
        observer = CodexSessionLedgerObserver(self.home / ".codex")
        with patch.object(observer, "observe_reconciled", side_effect=ValueError("temporary ledger failure")):
            _, total, _, _ = normalize_stream("codex", codex_stream(), session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "exact")
        self.assertEqual(total.total_tokens, 110)
        delta = count(10, 1, 2)["payload"]["info"]["total_token_usage"]
        self.write(root, root.read_text() + lines(count(110, 11, 22, last=delta)))
        capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
        self.finish(capture, codex_stream(110, 11, 22))
        _, total, _, errors = normalize_stream("codex", codex_stream(110, 11, 22), session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "partial")
        self.assertTrue(any("codex_ledger_unavailable" in error for error in errors))

    def test_codex_partial_capture_does_not_block_ledger_in_either_entrypoint(self):
        codex_home = self.root / "codex"
        self.write(codex_home / f"sessions/rollout-{ROOT_ID}.jsonl", lines(metadata(ROOT_ID), count()))
        stream = codex_stream()
        # Simulate a stream-only capture despite an independently usable ledger.
        capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
        self.finish(capture, stream)
        observer = CodexSessionLedgerObserver(codex_home)
        events, total, capabilities, errors = normalize_stream("codex", stream, session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "exact")
        self.assertEqual((total.input_tokens, total.cache_read_tokens, total.total_tokens), (80, 20, 110))
        self.assertTrue(capabilities.usage_delta_observed)
        self.assertEqual(errors, ())
        self.assertEqual(len([event for event in events if event.kind == "usage_delta"]), 1)

        runtime = CodexRuntime(process_runner=lambda *a, **k: (stream, "", 0, False))
        with patch("orchestrator.agent_runtime.runtime.CodexTemporaryHome") as temporary:
            temporary.return_value.open.return_value = codex_home
            temporary.return_value.close.return_value = None
            result = runtime.run(AgentRunRequest(self.root, "prompt", 5))
        self.assertEqual(result.terminal_usage.measurement, "exact")
        self.assertEqual(result.terminal_usage.total_tokens, 110)
        self.assertEqual(result.session_id, ROOT_ID)

    def codex_entrypoint_observations(self, stream, home):
        observer = CodexSessionLedgerObserver(home)
        normalized = normalize_stream("codex", stream, session_id=ROOT_ID, codex_observer=observer)
        # Ledger cursors must advance even when capture is selected.
        self.assertEqual(observer._session_usage.total_tokens, 110)
        runtime = CodexRuntime(process_runner=lambda *a, **k: (stream, "", 0, False))
        with patch("orchestrator.agent_runtime.runtime.CodexTemporaryHome") as temporary:
            temporary.return_value.open.return_value = home
            temporary.return_value.close.return_value = None
            result = runtime.run(AgentRunRequest(self.root, "prompt", 5))
        return {
            "long_horizon": normalized,
            "runtime": (result.events, result.terminal_usage, result.capabilities, result.observation_errors),
        }

    def test_incomplete_capture_is_preserved_independently_of_error_wording(self):
        home = self.root / "ledger"
        self.write(home / f"sessions/rollout-{ROOT_ID}.jsonl", lines(metadata(ROOT_ID), count()))
        stream = codex_stream()
        for label in ("stdout_capture_write", "codex_session_identity", "storage_write_failure"):
            with self.subTest(label=label):
                capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
                capture._capture_error(label, OSError("disk full"))
                report = self.finish(capture, stream)
                self.assertFalse(report["capture_complete"])
                self.assertEqual(report["response_count"], 0)
                observed = captured_observation(stream, (), AgentRuntimeCapabilities(True, True, False))
                self.assertIsNotNone(observed)
                self.assertFalse(observed.capture_complete)
                self.assertFalse(observed.capabilities.usage_delta_observed)
                for entrypoint, (_, total, capabilities, errors) in self.codex_entrypoint_observations(stream, home).items():
                    with self.subTest(entrypoint=entrypoint):
                        self.assertEqual(total.measurement, "partial")
                        self.assertEqual(total.total_tokens, 110)
                        self.assertFalse(capabilities.usage_delta_observed)
                        self.assertIn(f"{label}:OSError:disk full", errors)

    def test_warning_text_cannot_block_healthy_stream_only_ledger_fallback(self):
        home = self.root / "ledger"
        self.write(home / f"sessions/rollout-{ROOT_ID}.jsonl", lines(metadata(ROOT_ID), count()))
        stream = codex_stream()
        capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
        original_report = capture._usage.report

        def report_with_note(*args, **kwargs):
            report = original_report(*args, **kwargs)
            report["warnings"].append("provider note: capture pipe delivered terminal counters only")
            return report

        with patch.object(capture._usage, "report", side_effect=report_with_note):
            report = self.finish(capture, stream)
        self.assertTrue(report["capture_complete"])
        self.assertEqual(report["total"]["measurement"], "partial")
        observed = captured_observation(stream, (), AgentRuntimeCapabilities(True, True, False))
        self.assertIsNotNone(observed)
        self.assertTrue(observed.capture_complete)
        for entrypoint, (_, total, capabilities, errors) in self.codex_entrypoint_observations(stream, home).items():
            with self.subTest(entrypoint=entrypoint):
                self.assertEqual(total.measurement, "exact")
                self.assertEqual(total.total_tokens, 110)
                self.assertTrue(capabilities.usage_delta_observed)
                self.assertEqual(errors, ())

    def test_codex_failed_ledger_retains_partial_capture(self):
        stream = codex_stream()
        capture = SessionCapture(self.root / "captures", backend="codex", command=["codex", "prompt"], provider_home=None, context={})
        self.finish(capture, stream)
        observer = CodexSessionLedgerObserver(self.root / "absent")
        with patch.object(observer, "observe_reconciled", side_effect=ValueError("bad ledger")):
            _, total, _, errors = normalize_stream("codex", stream, session_id=ROOT_ID, codex_observer=observer)
        self.assertEqual(total.measurement, "partial")
        self.assertEqual(total.total_tokens, 110)
        self.assertTrue(any("codex_ledger_unavailable" in error for error in errors))

    def test_real_codex_child_capture_in_none_and_auto_without_bwrap(self):
        for mode in ("none", "auto"):
            with self.subTest(mode=mode):
                workspace = self.root / mode
                workspace.mkdir()
                runtime = SupervisorRuntime(SupervisorRuntimeConfig(
                    repository_root=Path(__file__).resolve().parents[1], hardware="test",
                    agent_sandbox=mode, sandbox_timeout=5,
                ))
                runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))
                code = (
                    "import os\nfrom pathlib import Path\n"
                    "root = Path(os.environ['CODEX_HOME']) / 'sessions'\n"
                    "root.mkdir(parents=True, exist_ok=True)\n"
                    f"(root / 'rollout-{ROOT_ID}.jsonl').write_text({lines(metadata(ROOT_ID), count())!r})\n"
                    f"(root / 'rollout-{CHILD_ID}.jsonl').write_text({lines(metadata(CHILD_ID, ROOT_ID), count(20, 2, 4))!r})\n"
                    f"print({codex_stream()!r}, end='', flush=True)\n"
                )
                try:
                    with (
                        patch.dict(os.environ, {"HOME": str(self.home), "CODEX_HOME": str(self.home / ".codex")}),
                        patch("orchestrator.agent_sandbox._bwrap_path", return_value=None),
                        patch.object(CodexRuntime, "build_command", return_value=[sys.executable, "-c", code]),
                        activate_supervisor_runtime(runtime),
                    ):
                        result = CodexRuntime().run(AgentRunRequest(workspace, "prompt", 5))
                    self.assertEqual(result.exit_status, 0, result.stderr_tail)
                    self.assertEqual(result.terminal_usage.measurement, "exact")
                    self.assertEqual(result.terminal_usage.total_tokens, 132)
                    self.assertEqual(result.session_id, ROOT_ID)
                    _, scope = runtime._private_scope_for(workspace)
                    reports = list((scope / "sessions").glob("*/token-usage.json"))
                    self.assertEqual(len(reports), 1)
                    report = json.loads(reports[0].read_text())
                    self.assertEqual(report["total"]["total_tokens"], 132)
                    self.assertEqual(report["total"]["measurement"], "exact")
                    self.assertEqual(report["response_count"], 2)
                    self.assertEqual(len(list(reports[0].parent.glob("provider/native/.codex/sessions/*.jsonl"))), 2)
                finally:
                    runtime._server = None
                    runtime.close()


if __name__ == "__main__":
    unittest.main()
