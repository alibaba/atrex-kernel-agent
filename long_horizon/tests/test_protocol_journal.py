from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_horizon.journal import append_experiment, finalize, initialize, validate_terminal
from long_horizon.protocol import normalize_handoff, read_handoff


class HandoffTests(unittest.TestCase):
    def test_candidate_requires_commit(self) -> None:
        self.assertIsNone(normalize_handoff({"status": "candidate_ready"}))

    def test_pivot_is_terminal_without_commit(self) -> None:
        handoff = normalize_handoff({"status": "pivot"})
        self.assertIsNotNone(handoff)
        self.assertEqual(handoff.status, "pivot")

    def test_invalid_json_is_not_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "handoff.json"
            path.write_text("{", encoding="utf-8")
            self.assertIsNone(read_handoff(path))


class JournalTests(unittest.TestCase):
    def test_complete_candidate_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "journal.json"
            initialize(path, episode=1, base_commit="base", branch="episode")
            append_experiment(path, {"name": "tile", "result": "faster"})
            finalize(
                path,
                state="candidate_ready",
                candidate_commit="candidate",
                outcome={"summary": "won", "next_directions": []},
            )
            self.assertEqual(
                validate_terminal(
                    path,
                    expected_episode=1,
                    base_commit="base",
                    branch="episode",
                    state="candidate_ready",
                    candidate_commit="candidate",
                ),
                "",
            )

    def test_finalize_requires_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "journal.json"
            initialize(path, episode=1, base_commit="base", branch="episode")
            with self.assertRaisesRegex(ValueError, "at least one experiment"):
                finalize(
                    path,
                    state="pivot",
                    outcome={"summary": "no path", "next_directions": []},
                )

    def test_terminal_state_must_match_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "journal.json"
            initialize(path, episode=2, base_commit="base", branch="episode")
            append_experiment(path, {"name": "probe"})
            finalize(
                path,
                state="pivot",
                outcome={"summary": "pivot", "next_directions": ["rewrite"]},
            )
            diagnosis = validate_terminal(
                path,
                expected_episode=2,
                base_commit="base",
                branch="episode",
                state="blocked",
            )
            self.assertIn("state does not match", diagnosis)


if __name__ == "__main__":
    unittest.main()
