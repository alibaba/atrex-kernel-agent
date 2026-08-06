from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from long_horizon import main_adapter


class SessionAdapterTests(unittest.TestCase):
    def test_prompt_transport_is_long_horizon_local(self) -> None:
        claude, claude_input = main_adapter.session_prompt_transport(
            "claude", ["claude", "--print", "large prompt"], "large prompt"
        )
        codex, codex_input = main_adapter.session_prompt_transport(
            "codex", ["codex", "exec", "large prompt"], "large prompt"
        )
        pi, pi_input = main_adapter.session_prompt_transport(
            "pi", ["pi", "large prompt"], "large prompt"
        )

        self.assertEqual(claude, ["claude", "--print"])
        self.assertEqual(claude_input, "large prompt")
        self.assertEqual(codex, ["codex", "exec", "-"])
        self.assertEqual(codex_input, "large prompt")
        self.assertEqual(pi, ["pi", "large prompt"])
        self.assertIsNone(pi_input)

    def test_bounded_prompt_uses_private_tempfile_without_changing_main(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(command, workspace, timeout, environment):
            captured["command"] = command
            captured["prompt_path"] = command[4]
            captured["prompt"] = Path(command[4]).read_text(encoding="utf-8")
            return "out", "", 0, False

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            main_adapter.base, "_run_bounded", side_effect=fake_run
        ):
            result = main_adapter.run_bounded(
                ["claude", "--print"],
                Path(temp),
                60,
                {},
                "x" * 200_000,
            )

        command = captured["command"]
        self.assertEqual(result, ("out", "", 0, False))
        self.assertEqual(command[5:], ["claude", "--print"])
        self.assertEqual(captured["prompt"], "x" * 200_000)
        self.assertFalse(Path(captured["prompt_path"]).exists())

    def test_fresh_codex_session_is_persistent(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            command = main_adapter.fresh_session_command(
                "work", "unused-supervisor-id", "high", "codex"
            )
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertNotIn("--ephemeral", command)

    def test_codex_resume_command_uses_exec_resume(self) -> None:
        thread_id = "019c1234-5678-7abc-8def-0123456789ab"
        with mock.patch.dict("os.environ", {}, clear=True):
            command = main_adapter.resume_session_command(
                "continue", thread_id, "high", "codex"
            )
        self.assertEqual(command[:3], ["codex", "exec", "resume"])
        self.assertEqual(command[-2:], [thread_id, "continue"])
        self.assertNotIn("--ephemeral", command)
        self.assertNotIn("--color", command)

    def test_pi_uses_one_persistent_json_session_without_resume_support(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            command = main_adapter.fresh_session_command(
                "work", "pi-session-id", "high", "pi"
            )

        self.assertEqual(command[:3], ["pi", "--mode", "json"])
        self.assertEqual(
            command[command.index("--session-id") + 1], "pi-session-id"
        )
        self.assertFalse(main_adapter.supports_same_session_resume("pi"))

    def test_codex_thread_id_is_read_from_jsonl(self) -> None:
        thread_id = "019c1234-5678-7abc-8def-0123456789ab"
        stdout = "\n".join(
            [
                "not json",
                json.dumps({"type": "thread.started", "thread_id": thread_id}),
            ]
        )
        self.assertEqual(
            main_adapter.session_id_from_stream("codex", stdout, "unused"), thread_id
        )


if __name__ == "__main__":
    unittest.main()
