from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from long_horizon.protocol import atomic_write_json
from long_horizon.session import LongSessionRunner


class SessionRecoveryTests(unittest.TestCase):
    def test_missing_handoff_resumes_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            handoff = workspace / "handoff.json"
            commands: list[list[str]] = []

            def execute(command, cwd, timeout, environment):
                commands.append(command)
                if len(commands) == 2:
                    atomic_write_json(handoff, {"status": "pivot"})
                return "", "", 0, False

            with mock.patch("long_horizon.main_adapter.session_environment", return_value={}), mock.patch(
                "long_horizon.main_adapter.fresh_session_command",
                side_effect=lambda prompt, sid, effort: ["claude", "--session-id", sid, prompt],
            ), mock.patch(
                "long_horizon.main_adapter.resume_session_command",
                side_effect=lambda prompt, sid, effort: ["claude", "--resume", sid, prompt],
            ):
                result = LongSessionRunner(executor=execute).run(
                    workspace,
                    "work",
                    timeout=60,
                    handoff_path=handoff,
                    handoff_resumes=2,
                    completion_check=lambda value: "",
                )
            self.assertEqual(result.resume_count, 1)
            self.assertEqual(result.handoff.status, "pivot")
            self.assertEqual(commands[0][2], commands[1][2])
            self.assertEqual(commands[1][1], "--resume")

    def test_nonzero_exit_does_not_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            with mock.patch("long_horizon.main_adapter.session_environment", return_value={}), mock.patch(
                "long_horizon.main_adapter.fresh_session_command", return_value=["claude"]
            ):
                result = LongSessionRunner(
                    executor=lambda command, cwd, timeout, environment: ("", "boom", 2, False)
                ).run(
                    workspace,
                    "work",
                    timeout=60,
                    handoff_path=workspace / "handoff.json",
                    handoff_resumes=2,
                    completion_check=lambda value: "",
                )
            self.assertEqual(result.resume_count, 0)
            self.assertEqual(result.exit_status, 2)


if __name__ == "__main__":
    unittest.main()
