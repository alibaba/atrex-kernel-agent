"""Execute the Skill's documented requests against private temporary stores, without GPU jobs."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from orchestrator.constants import REPO_ROOT
from supervisor import gateway
from supervisor.errors import AgentRequestError
from supervisor.journal import SupervisorJournalService, initialize_journal
from tools import sandbox

REFERENCE = REPO_ROOT / "orchestrator/agent_skills/runtime-records/references"


def example(file: str, heading: str, index: int = 0) -> dict:
    section = (REFERENCE / file).read_text().split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return json.loads(re.findall(r"```json\n(.*?)```", section, re.DOTALL)[index])


class RuntimeRecordsSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.evidence = self.workspace / "private"
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.evidence),
                },
                clear=True,
            )
        )
        initialize_journal(
            self.evidence / "journal.json",
            episode=1,
            base_commit="a" * 40,
            branch="episode-1",
        )
        self.service = SupervisorJournalService(
            workspace=self.workspace,
            campaign_root=self.workspace,
            evidence_root=self.evidence,
        )
        self.ids: dict[str, str] = {}
        self.source = b"def run(x):\n    return x\n"
        (self.workspace / "kernel.py").write_bytes(self.source)

    def replace_ids(self, value: dict) -> dict:
        text = json.dumps(value)
        for before, after in self.ids.items():
            text = text.replace(before, after)
        return json.loads(text)

    def journal_cli(self, kind: str, body: dict | None = None, *options: str) -> dict:
        arguments = ["--kind", kind]
        if body is not None:
            request = self.workspace / "scratch/request.json"
            request.parent.mkdir(exist_ok=True)
            request.write_text(json.dumps(self.replace_ids(body)))
            arguments += ["--request-file", str(request)]
        arguments += list(options)

        def proxy(operation: str, **payload: object) -> int:
            print(json.dumps(self.service.execute({"operation": operation, **payload})))
            return 0

        output = io.StringIO()
        with patch.object(sandbox, "proxy_journal", side_effect=proxy), redirect_stdout(output):
            self.assertEqual(sandbox.main(arguments), 0)
        return json.loads(output.getvalue())

    def prepare_experiment(self) -> str:
        proposed = self.journal_cli(
            "update-direction", example("journal.md", "Propose a Direction")
        )
        self.ids["direction_11111111111111111111111111111111"] = proposed["direction_id"]
        self.assertEqual(
            proposed, self.replace_ids(example("journal.md", "Propose a Direction", 1))
        )
        self.journal_cli("update-direction", example("journal.md", "Start or resume exploration"))
        measured = example("records.md", "Read a Gateway result")["result"]
        record = gateway._record_episode_evaluation(self.workspace, measured, gateway_kind="run")
        self.assertIsNotNone(record)
        self.ids["gateway-100-111111111111"] = record["record_id"]
        self.ids["kernel-100-aaaaaaaaaaaa"] = record["kernel_id"]
        recorded = self.journal_cli(
            "record-experiment",
            example("journal.md", "Measure and record an Experiment"),
        )
        self.ids["experiment_22222222222222222222222222222222"] = recorded["experiment_id"]
        self.assertEqual(
            recorded,
            self.replace_ids(
                example("journal.md", "Measure and record an Experiment", 1),
            ),
        )
        return recorded["experiment_id"]

    def test_documented_workflow_persists_and_loads_then_accepts_report(self) -> None:
        experiment_id = self.prepare_experiment()
        loaded = self.journal_cli("load-experiment", None, "--record-id", experiment_id)
        for key, value in self.replace_ids(
            example("journal.md", "Measure and record an Experiment"),
        ).items():
            self.assertEqual(loaded[key], value)
        self.assertNotIn("sequence", loaded)
        direction = self.journal_cli("load-direction", None, "--record-id", loaded["direction_id"])
        self.assertEqual(direction["supporting_experiment_ids"], [experiment_id])
        self.assertEqual(direction["status"], "in_progress")
        for kind, key in (("list-directions", "directions"), ("list-experiments", "experiments")):
            path = f"scratch/{key}.json"
            response = self.journal_cli(kind, None, "--output-path", path)
            self.assertEqual(response, {"status": "written", "file": path, "count": 1})
            index = json.loads((self.workspace / path).read_text())
            self.assertEqual(len(index[key]), 1)
        self.journal_cli("update-direction", example("journal.md", "Close the Direction"))
        with patch(
            "long_horizon.git_episode.EpisodeWorktree.commit_candidate", return_value="b" * 40
        ) as commit:
            response = self.journal_cli("episode-report", example("journal.md", "Candidate report"))
        commit.assert_called_once_with(self.source)
        self.assertEqual(response, example("journal.md", "Candidate report", 1))

    def test_historical_result_and_source_examples_match_real_projections(self) -> None:
        self.prepare_experiment()
        record_id = self.ids["gateway-100-111111111111"]
        kernel_id = self.ids["kernel-100-aaaaaaaaaaaa"]
        cases = (
            (
                lambda: gateway._read_gateway_record(self.workspace, record_id),
                "Read a Gateway result",
            ),
            (
                lambda: gateway._read_kernel_gateway_records(self.workspace, kernel_id),
                "Discover a Kernel's Gateway records",
            ),
            (
                lambda: gateway._read_kernel_source(
                    self.workspace, kernel_id, "scratch/previous.py"
                ),
                "Recover exact Kernel source",
            ),
        )
        for invoke, heading in cases:
            with self.subTest(heading=heading), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(invoke(), 0)
            text = output.getvalue().strip()
            self.assertTrue(text.startswith(gateway.RECORD_RESULT_PREFIX))
            actual = json.loads(text.removeprefix(gateway.RECORD_RESULT_PREFIX))
            expected = self.replace_ids(example("records.md", heading))
            if heading == "Recover exact Kernel source":
                expected["size_bytes"] = len(self.source)
            self.assertEqual(actual, expected)
        self.assertEqual((self.workspace / "scratch/previous.py").read_bytes(), self.source)

    def test_documented_pivot_requires_experiment(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.journal_cli("episode-report", example("journal.md", "Pivot report"))
        self.prepare_experiment()
        self.journal_cli("update-direction", example("journal.md", "Close the Direction"))
        self.assertEqual(
            self.journal_cli(
                "episode-report",
                example("journal.md", "Pivot report"),
            )["status"],
            "accepted",
        )

    def test_documented_blocked_report_needs_no_experiment(self) -> None:
        self.assertEqual(
            self.journal_cli(
                "episode-report",
                example("journal.md", "Blocked report"),
            ),
            {"status": "accepted", "message": "Report accepted and recorded"},
        )

    def test_documented_error_fields_match_actual_validation(self) -> None:
        with self.assertRaises(AgentRequestError) as rejected:
            self.service.execute(
                {
                    "operation": "episode_report",
                    "request": {"status": "pivot", "decision": "stop"},
                }
            )
        self.assertEqual(rejected.exception.response, example("records.md", "Error recovery"))


if __name__ == "__main__":
    unittest.main()
