"""Repair guidance tests without a live model or GPU service."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supervisor import gateway
from supervisor.errors import AgentRequestError, RuntimeStateError
from supervisor.journal import SupervisorJournalService, initialize_journal, load_journal


class AgentErrorsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.evidence = self.workspace / "evidence"
        self.path = self.evidence / "journal.json"
        initialize_journal(self.path, episode=1, base_commit="a" * 40, branch="episode-1")
        self.service = SupervisorJournalService(
            workspace=self.workspace, campaign_root=self.workspace, evidence_root=self.evidence
        )

    def propose(self) -> str:
        return self.service.execute({
            "operation": "direction_update",
            "request": {
                "action": "propose", "name": "fusion", "hypothesis": "less traffic",
                "rationale": "extra writes", "plan": ["fuse"],
                "success_criteria": ["faster"], "stop_conditions": ["no improvement"],
            },
        })["direction_id"]

    def update(self, direction_id: str, action: str) -> dict:
        closure = {}
        if action != "start":
            (self.workspace / "kernel.py").write_text("def run(x): return x\n")
            with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.evidence)}):
                record = gateway._record_episode_evaluation(
                    self.workspace, {"passed": True}, gateway_kind="check",
                )
            receipt = self.service.execute({
                "operation": "experiment_record",
                "request": {
                    "direction_id": direction_id, "name": "check", "hypothesis": "compiles",
                    "change": "check the candidate", "gateway_record_ids": [record["record_id"]],
                    "evidence": "compiled", "analysis": "performance untested",
                    "action": "abandon_direction",
                },
            })
            closure = {"hypothesis_status": "unresolved", "supporting_experiment_ids": [receipt["experiment_id"]]}
        return self.service.execute({
            "operation": "direction_update",
            "request": {"action": action, "direction_id": direction_id, "analysis": "test", **closure},
        })

    def test_missing_and_extra_fields_are_separate_and_rejection_is_nonmutating(self) -> None:
        before = self.path.read_bytes()
        with self.assertRaises(AgentRequestError) as rejected:
            self.service.execute({
                "operation": "direction_update",
                "request": {"action": "propose", "name": "fusion", "typo": "value"},
            })
        response = rejected.exception.response
        self.assertTrue(response["repairable"])
        self.assertEqual(response["error"]["code"], "invalid_fields")
        self.assertIn("hypothesis", response["error"]["missing_fields"])
        self.assertEqual(response["error"]["unexpected_fields"], ["typo"])
        self.assertNotIn("schema", response["error"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.propose().startswith("direction_"))

    def test_direction_conflict_and_limit_explain_available_actions(self) -> None:
        started = [self.propose() for _ in range(3)]
        self.update(started[0], "start")
        with self.assertRaises(AgentRequestError) as conflict:
            self.update(started[1], "start")
        self.assertEqual(conflict.exception.response["error"]["direction_id"], started[0])
        self.update(started[0], "defer")
        for direction_id in started[1:]:
            self.update(direction_id, "start")
            self.update(direction_id, "defer")
        fourth = self.propose()
        with self.assertRaises(AgentRequestError) as limit:
            self.update(fourth, "start")
        error = limit.exception.response["error"]
        self.assertEqual(error["code"], "direction_limit_exceeded")
        self.assertEqual(set(error["started_direction_ids"]), set(started))
        self.assertIn("episode-report", error["next_action"])
        self.assertIn("propose", error["next_action"])
        self.update(started[0], "start")  # The recommended resume path really works.

    def test_corrupt_private_journal_is_not_an_agent_repair(self) -> None:
        for content in ("not json", "{}", '{"schema_version":2,"runtime_managed":true}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(RuntimeStateError) as failure:
                    load_journal(self.path)
                self.assertFalse(failure.exception.response["repairable"])
                self.assertIn("operator", failure.exception.response["error"]["next_action"])
                self.assertNotIn(str(self.path), json.dumps(failure.exception.response))

    def test_bad_gateway_argument_is_compact_and_does_not_dump_supervisor_flags(self) -> None:
        output = io.StringIO()
        with patch("sys.stderr", output), self.assertRaises(SystemExit) as failure:
            gateway.build_parser().parse_args(["--kind", "profile", "--launch-count", "oops"])
        self.assertEqual(failure.exception.code, 2)
        response = json.loads(output.getvalue())
        self.assertEqual(response["error"]["code"], "invalid_arguments")
        self.assertIn("--launch-count", response["error"]["message"])
        self.assertNotIn("--ssh", output.getvalue())
        self.assertLess(len(output.getvalue()), 700)

    def test_dev_infrastructure_error_survives_immediate_response_and_record_read(self) -> None:
        (self.workspace / "kernel.py").write_text("def run(x): return x\n")
        error = {
            "error_class": "infra", "reason": "exec_failed", "message": "runtime_env setup failed"
        }
        for remote in (None, {"stdout": "partial probe output", "stderr": "partial diagnostic"}):
            job = {
                "job_id": "dv_mock", "status": "failed", "result": remote,
                "error": {**error, "details": {"private": "do-not-project"}},
            }
            stdout = io.StringIO()
            with (
                self.subTest(remote=remote),
                patch.dict(os.environ, {}, clear=True),
                patch.object(gateway, "_find_agate", return_value="/unused/agate"),
                patch.object(gateway, "_uses_standard_oss_gateway", return_value=False),
                patch.object(gateway, "_run_agate_with_cancel_retry", return_value=
                    subprocess.CompletedProcess([], 1, stdout=json.dumps(job), stderr="")),
                patch.object(
                    gateway.subprocess, "run", side_effect=AssertionError("external call")
                ),
                patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()),
            ):
                code = gateway._main([
                    "--kind", "dev", "--workspace", str(self.workspace), "--hardware", "L20N",
                    "--url", "https://unused.invalid", "--sync", "scratch/out",
                    "--", "python3", "-c", "print(1)",
                ])
                response = json.loads(stdout.getvalue())
                record = gateway._load_gateway_record(self.workspace, response["gateway_record_id"])
                historical = gateway._gateway_record_public_result(
                    response["gateway_record_id"], record
                )
            self.assertEqual(code, 1)
            self.assertFalse(response["repairable"])
            for key, value in error.items():
                self.assertEqual(response["error"][key], value)
                self.assertEqual(historical["result"]["error"][key], value)
            self.assertEqual(historical["result"]["error"], response["error"])
            self.assertNotIn("do-not-project", json.dumps(response))
            if remote:
                self.assertEqual(historical["result"]["stdout"], remote["stdout"])

    def test_all_typed_failed_records_keep_same_safe_error_fields(self) -> None:
        (self.workspace / "kernel.py").write_text("def run(x): return x\n")
        error = {"error_class": "infra", "reason": "logs_unavailable", "message": "logs missing"}
        for kind in ("run", "profile", "check", "disassemble"):
            with self.subTest(kind=kind), patch.dict(os.environ, {}, clear=True):
                record = gateway._record_episode_evaluation(
                    self.workspace, {"status": "failed", "error": error}, gateway_kind=kind,
                )
                stored = gateway._load_gateway_record(self.workspace, record["record_id"])
                response = gateway._gateway_record_public_result(record["record_id"], stored)
            self.assertEqual(response["status"], "failed")
            self.assertEqual(response["result"]["error"]["reason"], "logs_unavailable")
            self.assertFalse(response["result"]["repairable"])

    def test_main_classifies_missing_supervisor_dependency(self) -> None:
        with patch.object(gateway, "_main", side_effect=RuntimeStateError(
            "Supervisor Agate client unavailable", code="gateway_dependency_unavailable"
        )), self.assertRaises(SystemExit) as rejected:
            gateway.main(["--kind", "dev"])
        response = json.loads(rejected.exception.code)
        self.assertFalse(response["repairable"])
        self.assertIn("Do not install packages", response["error"]["next_action"])

    def test_duplicate_measurement_points_to_a_read_not_another_measurement(self) -> None:
        record_id = "gateway-100-0123456789ab"
        for previous in (record_id, None):
            with self.subTest(previous=previous), self.assertRaises(SystemExit) as rejected:
                gateway._reject_duplicate_task(previous)
            response = json.loads(rejected.exception.code)
            self.assertEqual(response["error"]["code"], "duplicate_gateway_task")
            if previous:
                self.assertEqual(response["error"]["gateway_record_id"], record_id)
                self.assertIn("--kind record-read", response["error"]["next_action"])
            else:
                self.assertNotIn("gateway_record_id", response["error"])
                self.assertIn("Wait for the original request", response["error"]["next_action"])
