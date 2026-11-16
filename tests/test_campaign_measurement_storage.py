"""Exercise shared Experiment execution against one Campaign store, without a GPU."""
import json
import tempfile
import unittest
from pathlib import Path

from aka.contracts.evaluation import CommandEvaluation, PreparedEvaluation, StepEvaluation
from aka.contracts.execution import BackendCapabilities, CommandSpec, ExecutionFacts, JobResult
from aka.experiment.factory import ExperimentFactory
from aka.experiment.measurement_store import MeasurementStore


class TestEvaluator:
    def prepare(self, command):
        if command.command == ("invalid-input",):
            raise ValueError("request_validation_failed: invalid benchmark input")
        return PreparedEvaluation((command,), 1, None)

    def observe_step(self, prepared, index, result):
        return StepEvaluation()

    def fallback(self, prepared, results):
        return False

    def interpret(self, prepared, results):
        return CommandEvaluation(results[0])


class TestExecutor:
    capabilities = BackendCapabilities(operations=frozenset({
        "run", "same_allocation_abba", "profile", "dev", "command", "check", "disassemble"}))
    resource_identity = None

    def run(self, command):
        if command.command == ("timeout",):
            raise TimeoutError("timed out while collecting measurement")
        failed = command.command == ("compile-error",)
        return JobResult(0, stdout="raw-" + command.operation + "\n" + "x" * 25000,
            stderr="compiler: candidate compilation failed" if failed else "", exit_code=1 if failed else 0,
            transport_status="succeeded", facts=ExecutionFacts(disposition="terminal"))


class CampaignMeasurementStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = MeasurementStore(self.root / "measurements")
        self.factory = ExperimentFactory(command_evaluator=TestEvaluator()).for_campaign(self.store.root)
        self.executor = TestExecutor()

    def run_operation(self, trajectory, request_id, operation="run", command=("measure",)):
        workspace = self.root / "lineages" / trajectory / "agent-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        with self.factory.open(workspace=workspace, purpose="development", executor=self.executor) as experiment:
            return experiment.run_recorded(CommandSpec(workspace, command, operation=operation),
                                           request_id=request_id)

    def test_all_operations_and_both_lineages_share_one_store(self):
        for i, operation in enumerate(("run", "same_allocation_abba", "profile", "dev", "command", "check", "disassemble")):
            self.run_operation("t0001" if i % 2 else "t0002", f"request-{i}", operation)
        records = list(self.store.execution_records().iter_records())
        self.assertEqual(len(records), 7)
        self.assertEqual({record.ref.run_id for record in records}, {"campaign"})
        self.assertTrue(all(record.lifecycle == "terminal" for record in records))
        self.assertEqual(len(list((self.root / "measurements" / "executions").glob("*/record.json"))), 7)
        self.assertEqual(list((self.root / "lineages").rglob("*.json")), [])
        self.assertFalse(list(self.root.rglob(".aka-experiments")))
        self.assertFalse(list(self.root.rglob(".atrex_long_horizon")))

    def test_failed_timeout_and_preparation_errors_are_retained(self):
        self.run_operation("t0001", "compile", "run", ("compile-error",))
        with self.assertRaises(TimeoutError):
            self.run_operation("t0002", "timeout", "dev", ("timeout",))
        with self.assertRaises(ValueError):
            self.run_operation("t0002", "invalid", "same_allocation_abba", ("invalid-input",))
        records = {record.request_id: record for record in self.store.execution_records().iter_records()}
        self.assertEqual(set(records), {"compile", "timeout", "invalid"})
        self.assertEqual(records["compile"].attempts[-1].result.stderr, "compiler: candidate compilation failed")
        self.assertEqual(records["timeout"].attempts[-1].failure.exception_type, "TimeoutError")
        self.assertEqual(records["timeout"].attempts[-1].failure.message, "timed out while collecting measurement")
        self.assertEqual(records["invalid"].failure.exception_type, "ValueError")
        self.assertEqual(records["invalid"].failure.message, "request_validation_failed: invalid benchmark input")
        self.assertEqual(records["invalid"].attempts, ())

    def test_agent_receipt_does_not_replace_raw_execution_result(self):
        record = self.run_operation("t0001", "long-output", "command")
        returned = self.store.execution_records().command_result(record.ref)
        self.assertEqual(returned.stdout, "raw-command\n" + "x" * 25000)
        self.assertEqual(record.attempts[-1].result.stdout, returned.stdout)

    def test_gateway_record_archival_does_not_require_success_or_submission(self):
        for i, operation in enumerate(("evaluate", "same_allocation_abba", "profile", "dev", "command")):
            request = {"operation": operation, "options": {"evaluation_mode": "full"}, "test_id": i}
            with self.store.reserve(request, {}, request_id=f"runtime-{i}") as task:
                archived = task.finish({"exit_code": 1, "stdout": "raw diagnostics", "stderr": "timeout"}, cacheable=False)
            saved = self.store.read(archived["gateway_record_id"])
            self.assertEqual(saved["response"]["stdout"], "raw diagnostics")
            self.assertEqual(saved["runtime_request_id"], f"runtime-{i}")
        self.assertEqual(len(self.store.record_ids()), 5)


if __name__ == "__main__":
    unittest.main()
