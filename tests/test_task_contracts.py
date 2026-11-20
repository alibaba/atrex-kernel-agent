"""Contract workflow tests using real processes, Git snapshots and durable task state."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from aka.contracts.task import Assessment, EvaluationPlan, ExecutionReceipt, Metric
from aka.light.tasks.content import git
from aka.light.tasks.objective import MetricObjective
from aka.light.tasks.service import TaskService
from aka.light.tasks.worker import run


HARNESS = '''import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1])/"compute.py"
s=importlib.util.spec_from_file_location("candidate",p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
state=0
passed=True
for x in (1,2,3):
    state=m.update(state,x)
    passed=passed and state==sum(range(1,x+1))
print("AKA_TASK_RESULT="+json.dumps({"verdict":"pass" if passed else "fail", "complete":True,
 "environment":"cpu-test-v1","metrics":[{"name":"cost","value":m.COST,"unit":"units"}],
 "details":{"final_state":state}}))
'''


class TaskContractsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = self.root / "contract"
        self.source = self.root / "source"
        self.contract.mkdir()
        self.source.mkdir()
        (self.contract / "sequence.py").write_text(HARNESS)
        (self.source / "compute.py").write_text("COST=10\ndef update(state,x): return state+x\n")
        self.spec = {
            "schema_version": 1, "adapter": "command", "contract": "contract", "seed": "source",
            "target": {"backend": "local", "timeout_s": 10},
            "config": {"command": [sys.executable, "{contract}/sequence.py", "{candidate}"],
                       "instructions": "Implement update(state,x) as state+x for repeated calls.",
                       "delivery": {"entry": "compute.py:update"}},
            "objective": {"metrics": [{"name": "cost", "unit": "units", "direction": "minimize"}]},
            "budget": {"evaluations": 10}}
        self.spec_path = self.root / "task.json"
        self.write_spec()
        self.service = TaskService(self.root / "store")
        self.task = self.service.submit(self.spec_path, idempotency_key="test", start=False)["id"]
        self.workspace = self.service.directory(self.task) / "workspace"

    def write_spec(self):
        self.spec_path.write_text(json.dumps(self.spec))

    def edit(self, text):
        (self.workspace / "compute.py").write_text(text)
        git(self.workspace, "add", "--all")
        git(self.workspace, "commit", "-qm", "Candidate change")

    def candidate(self):
        evaluation = self.service.measure(self.task)
        return self.service.submit_candidate(self.task, evaluation["id"])

    def test_direct_contract_package_uses_same_python_and_json_path(self):
        import aka
        (self.contract / "contract.json").write_text(json.dumps({"adapter": "command", "config": self.spec["config"]}))
        job = aka.submit(contract=self.contract, baseline=self.source, workload={"cases": [1, 2]},
            validation={"exact": True}, execution=self.spec["target"], objective=self.spec["objective"],
            agent={}, root=self.service.root, key="direct", start=False)
        request = {"contract": str(self.contract), "baseline": str(self.source), "workload": {"cases": [1, 2]},
            "validation": {"exact": True}, "execution": self.spec["target"], "objective": self.spec["objective"],
            "agent": {}, "budget": {}}
        self.spec_path.write_text(json.dumps(request))
        cli = self.service.submit(self.spec_path, idempotency_key="direct", start=False)
        self.assertEqual(cli["id"], job.id)
        run(self.service.root, job.id)
        self.assertEqual(len(job.results()["candidates"]), 1)
        request.pop("agent")
        self.spec_path.write_text(json.dumps(request))
        default_cli = self.service.submit(self.spec_path, idempotency_key="default-agent", start=False)
        default_python = aka.submit(contract=self.contract, baseline=self.source,
            workload={"cases": [1, 2]}, validation={"exact": True}, execution=self.spec["target"],
            objective=self.spec["objective"], root=self.service.root, key="default-agent", start=False)
        self.assertEqual(default_cli["id"], default_python.id)

    def test_contract_identity_is_independent_of_starting_implementation(self):
        before = self.service.status(self.task)
        (self.source / "compute.py").write_text("COST=2\ndef update(state,x): return state+x\n")
        after = self.service.submit(self.spec_path, idempotency_key="other-seed", start=False)
        self.assertEqual(before["contract_id"], after["contract_id"])
        self.assertNotEqual(self.service.store.task(self.task)["manifest"]["content"]["seed"],
                            self.service.store.task(after["id"])["manifest"]["content"]["seed"])

    def test_caller_notes_share_api_cli_prompt_export_and_preserve_contract_identity(self):
        import aka
        from aka.light.tasks.worker import prompt
        (self.contract / "contract.json").write_text(json.dumps({"config": self.spec["config"]}))
        notes = "Run NCU through aka execution; retain the report and explain the measured bottleneck."
        job = aka.submit(contract=self.contract, baseline=self.source,
            execution=self.spec["target"], objective=self.spec["objective"],
            notes=notes, agent={}, root=self.service.root, key="notes", start=False)
        request = dict(contract=str(self.contract), baseline=str(self.source),
            execution=self.spec["target"], objective=self.spec["objective"],
            notes=notes, agent={}, budget={})
        self.spec_path.write_text(json.dumps(request))
        self.assertEqual(self.service.submit(self.spec_path, idempotency_key="notes", start=False)["id"], job.id)
        task, _, goal = self.service.bind(job.id)
        self.assertIn(notes, prompt(self.service, job.id, task["manifest"], goal))
        request["notes"] = "Try another research strategy."
        self.spec_path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "different task content"):
            self.service.submit(self.spec_path, idempotency_key="notes", start=False)
        other = self.service.submit(self.spec_path, idempotency_key="other-notes", start=False)
        self.assertEqual(other["contract_id"], job.status()["contract_id"])
        run(self.service.root, job.id)
        candidate = job.results()["candidates"][0]
        job.export(candidate["id"], self.root / "notes-export")
        exported = json.loads((self.root / "notes-export/task.json").read_text())
        self.assertEqual(exported["notes"], notes)

    def test_caller_notes_require_text(self):
        from aka.light.tasks.binding import validate_spec
        for notes in ({}, [], 1, None):
            with self.subTest(notes=notes), self.assertRaisesRegex(ValueError, "notes must be text"):
                validate_spec({**self.spec, "notes": notes})

    def test_binary_resources_and_current_candidate_imports(self):
        (self.source / "resource.bin").write_bytes(b"\x00\xff\x80")
        (self.source / "compute.py").write_text("from pathlib import Path\nCOST=2\ndef update(state,x):\n assert Path(__file__).with_name('resource.bin').read_bytes()==b'\\x00\\xff\\x80'\n return state+x\n")
        (self.contract / "sequence.py").write_text(HARNESS.replace('m=importlib.util.module_from_spec(s); s.loader.exec_module(m)', 'import compute as m\nassert Path(m.__file__).resolve()==p.resolve()'))
        task = self.service.submit(self.spec_path, idempotency_key="resources", start=False)["id"]
        result = self.service.measure(task)
        self.assertEqual(result["assessment"]["verdict"], "pass")
        self.assertTrue((self.service.directory(task) / "evaluations" / result["id"] / "candidate/resource.bin").exists())

    def test_git_ignored_build_outputs_are_not_source_or_research_payload(self):
        from aka.light.tasks.content import tree
        (self.workspace / ".gitignore").write_text("build/\n")
        (self.workspace / "build").mkdir()
        (self.workspace / "build/cache.so").write_bytes(b"\xff")
        (self.workspace / "scratch").mkdir()
        (self.workspace / "scratch/probe.py").write_text("print(1)")
        self.assertNotIn("build/cache.so", tree(self.workspace, include_scratch=True))
        self.assertIn("scratch/probe.py", tree(self.workspace, include_scratch=True))
        self.assertNotIn("scratch/probe.py", tree(self.workspace))

    def test_numerically_correct_candidate_fails_required_implementation_check(self):
        self.spec["config"]["implementation"] = {"forbidden_patterns": ["state\\+x"]}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="policy", start=False)["id"]
        record = self.service.measure(task)
        self.assertEqual(record["assessment"]["failure_kind"], "implementation")
        with self.assertRaisesRegex(ValueError, "does not satisfy"):
            self.service.submit_candidate(task, record["id"])

    def test_review_recovery_reuses_gpu_receipt_and_accounts_each_invocation(self):
        self.spec["config"]["implementation"] = {"self_authored": True}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="review-recovery", start=False)["id"]
        def review(source, policy, **context):
            return {"review_id": "first", "contract_id": context["contract_id"], "source_id": context["source_id"],
                    "complete": False, "passed": False, "errors": ["provider unavailable"],
                    "usage": {"total_tokens": 7, "measurement": "exact"}}
        with patch("aka.light.tasks.implementation.check", side_effect=review):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                self.service.measure(task)
        row = self.service.results(task)["evaluations"][0]
        self.assertEqual(row["state"], "unknown")
        def recovered(source, policy, **context):
            result = review(source, policy, **context)
            result.update(review_id="second", complete=True, passed=True, errors=[])
            return result
        with patch("aka.light.tasks.implementation.check", side_effect=recovered), patch("aka.light.tasks.execution.execute", side_effect=AssertionError("GPU rerun")):
            record = self.service.measure(task, evaluation_id=row["id"])
        self.assertEqual(self.service.status(task)["tokens"], 14)
        self.service.submit_candidate(task, record["id"])
        # A passing numerical envelope without source-bound review cannot publish.
        record["assessment"]["details"].pop("implementation")
        with self.service.store.transaction() as db:
            db.execute("UPDATE evaluations SET assessment=? WHERE id=?", (json.dumps(record["assessment"]), row["id"]))
        with self.assertRaisesRegex(ValueError, "implementation acceptance"):
            self.service.submit_candidate(task, row["id"])

    def test_review_budget_blocks_invocation_and_retains_completed_execution(self):
        self.spec["config"]["implementation"] = {"self_authored": True}
        self.spec["budget"]["tokens"] = 5
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="review-budget", start=False)["id"]
        with self.service.store.transaction() as db:
            db.execute("UPDATE tasks SET tokens=5 WHERE id=?", (task,))
        with patch("aka.light.tasks.implementation.check", side_effect=AssertionError("Review admitted")):
            with self.assertRaisesRegex(ValueError, "admission budget"):
                self.service.measure(task)
        record = self.service.store.evaluation(self.service.results(task)["evaluations"][0]["id"])
        self.assertEqual(record["receipt"]["status"], "completed")

    def test_failed_baseline_can_be_repaired_unless_required(self):
        (self.source / "compute.py").write_text("COST=10\ndef update(state,x): return x\n")
        task = self.service.submit(self.spec_path, idempotency_key="repair", start=False)["id"]
        run(self.service.root, task)
        self.assertEqual(self.service.status(task)["state"], "awaiting_candidate")
        workspace = self.service.directory(task) / "workspace"
        (workspace / "compute.py").write_text("COST=2\ndef update(state,x): return state+x\n")
        git(workspace, "add", "."); git(workspace, "commit", "-qm", "Repair")
        self.service.submit_candidate(task, self.service.measure(task)["id"])
        self.spec["config"]["baseline_required"] = True
        self.write_spec()
        required = self.service.submit(self.spec_path, idempotency_key="required-baseline", start=False)["id"]
        run(self.service.root, required)
        self.assertEqual(self.service.status(required)["state"], "needs_input")

    def test_failure_categories_do_not_turn_infrastructure_into_candidate_error(self):
        cases = [(ExecutionReceipt("failed", {"error": {"code": "infra/worker_lost"}}), "unknown", "infrastructure"),
                 (ExecutionReceipt("failed", {"error": {"code": "code/native_build_failed"}}), "fail", "build"),
                 (ExecutionReceipt("failed", {"exit_code": 1}), "fail", "execution")]
        for receipt, verdict, kind in cases:
            with self.subTest(kind=kind), patch("aka.light.tasks.execution.execute", return_value=receipt):
                result = self.service.measure(self.task)["assessment"]
                self.assertEqual((result["verdict"], result["failure_kind"]), (verdict, kind))

    def test_unsupported_runtime_stops_before_gpu_admission(self):
        self.spec["agent"] = {"runtime": "qodercli"}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="unsupported", start=False)["id"]
        with self.assertRaisesRegex(ValueError, "persistent runtime"), patch("aka.light.tasks.execution.execute") as execute:
            run(self.service.root, task)
        execute.assert_not_called()

    def test_non_benchmark_sequence_evaluate_submit_and_export(self):
        published = self.candidate()
        record = self.service.store.evaluation(published["evaluation_id"])
        self.assertEqual(record["assessment"]["details"]["final_state"], 6)
        target = self.root / "export"
        self.service.export(self.task, published["candidate_id"], target)
        self.assertEqual((target / "source" / "compute.py").read_text(), (self.source / "compute.py").read_text())
        self.assertTrue((target / "contract" / "sequence.py").exists())
        self.assertFalse((target / "contract" / "shapes.json").exists())
        self.assertNotIn("latency_us_geomean", record["assessment"])

    def test_objective_ranks_exact_candidates_without_overwriting_head(self):
        old = self.candidate()
        self.edit("COST=2\ndef update(state,x): return state+x\n")
        new = self.candidate()
        head = git(self.workspace, "rev-parse", "HEAD")
        results = self.service.results(self.task)
        self.assertEqual(results["frontier"], [new["candidate_id"]])
        self.assertEqual(len(results["candidates"]), 2)
        self.assertEqual(git(self.workspace, "rev-parse", "HEAD"), head)
        self.service.export(self.task, old["candidate_id"], self.root / "old")
        self.assertIn("COST=10", (self.root / "old/source/compute.py").read_text())

    def test_bad_state_transition_is_not_publishable(self):
        self.edit("COST=1\ndef update(state,x): return x\n")
        result = self.service.measure(self.task)
        self.assertEqual(result["assessment"]["verdict"], "fail")
        with self.assertRaisesRegex(ValueError, "does not satisfy"):
            self.service.submit_candidate(self.task, result["id"])
        public = self.service.results(self.task)
        self.assertEqual(public["evaluations"][0]["assessment"]["verdict"], "fail")
        self.assertNotIn("receipt", public["evaluations"][0])
        self.assertEqual(public["candidates"], [])

    def test_invalid_plan_is_rejected_before_evaluation_admission(self):
        with patch("aka.light.tasks.adapters.command.Adapter.prepare", return_value=EvaluationPlan("compile", {})):
            with self.assertRaisesRegex(ValueError, "eval or dev"):
                self.service.measure(self.task)
        self.assertEqual(self.service.status(self.task)["evaluations"], 0)

    def test_diagnostic_success_cannot_satisfy_acceptance(self):
        result = self.service.measure(self.task, scope="diagnostic")
        self.assertEqual(result["assessment"]["verdict"], "pass")
        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.service.submit_candidate(self.task, result["id"])

    def test_idempotency_and_task_mutation_are_separate(self):
        same = self.service.submit(self.spec_path, idempotency_key="test", start=False)
        self.assertEqual(same["id"], self.task)
        (self.contract / "sequence.py").write_text(HARNESS + "\n# caller revision\n")
        with self.assertRaisesRegex(ValueError, "different task content"):
            self.service.submit(self.spec_path, idempotency_key="test", start=False)
        # Editing the original caller directory does not mutate the frozen task.
        self.candidate()
        new = self.service.submit(self.spec_path, idempotency_key="revision-2", start=False)
        self.assertNotEqual(new["contract_id"], self.service.status(self.task)["contract_id"])

    def test_frozen_validation_cannot_be_changed(self):
        frozen = self.service.directory(self.task) / "contract/sequence.py"
        frozen.write_text("print('always passes')")
        with self.assertRaisesRegex(ValueError, "Frozen contract content changed"):
            self.service.measure(self.task)

    def test_dirty_source_and_symlinks_are_rejected(self):
        (self.workspace / "compute.py").write_text("not committed")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.service.measure(self.task)
        self.edit("COST=1\ndef update(state,x): return state+x\n")
        (self.workspace / "escape").symlink_to(self.spec_path)
        git(self.workspace, "add", "--all")
        git(self.workspace, "commit", "-qm", "Invalid link")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.service.measure(self.task)

    def test_failed_process_cannot_be_declared_pass_by_adapter(self):
        with patch("aka.light.tasks.execution.execute", return_value=ExecutionReceipt("failed", {"result": {}})):
            result = self.service.measure(self.task)
        self.assertEqual(result["assessment"]["verdict"], "unknown")
        with self.assertRaises(ValueError):
            self.service.submit_candidate(self.task, result["id"])

    def test_cancel_retains_accepted_artifacts_and_stops_new_evaluations(self):
        candidate = self.candidate()
        self.service.control(self.task, "cancel")
        self.service.export(self.task, candidate["candidate_id"], self.root / "cancelled-export")
        with self.assertRaisesRegex(ValueError, "paused or cancelled"):
            self.service.measure(self.task)
        with self.assertRaisesRegex(ValueError, "new task"):
            self.service.control(self.task, "resume")

    def test_event_cursor_feedback_and_incremental_publication(self):
        first = self.service.events(self.task)
        candidate = self.candidate()
        self.service.feedback(self.task, candidate["candidate_id"], {"adopted": False, "reason": "Integration cost"})
        events = self.service.events(self.task, after=first[-1]["seq"])
        self.assertIn("candidate_available", [e["kind"] for e in events])
        self.assertEqual(events[-1]["kind"], "integration_feedback")
        self.assertTrue(all(e["seq"] > first[-1]["seq"] for e in events))

    def test_worker_prepares_without_agent_or_format_branching(self):
        run(self.service.root, self.task)
        self.assertEqual(self.service.status(self.task)["state"], "awaiting_candidate")
        self.assertEqual(self.service.status(self.task)["candidates"], 1)
        run(self.service.root, self.task)
        self.assertEqual(self.service.status(self.task)["evaluations"], 2)

    def test_detached_worker_publishes_events(self):
        self.service.start(self.task)
        import time
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.service.status(self.task)["state"] == "awaiting_candidate":
                break
            time.sleep(0.05)
        self.assertEqual(self.service.status(self.task)["candidates"], 1)

    def test_unknown_submission_never_automatically_replays(self):
        with patch("aka.light.tasks.execution.execute", side_effect=TimeoutError("lost acceptance")):
            with self.assertRaises(TimeoutError):
                self.service.measure(self.task)
        with self.service.store.connect() as db:
            evaluation_id = db.execute("SELECT id FROM evaluations").fetchone()[0]
        with self.assertRaisesRegex(ValueError, "outcome is unknown"):
            self.service.measure(self.task, evaluation_id=evaluation_id)

    def test_interrupted_snapshot_creation_never_exposes_partial_frozen_source(self):
        from aka.light.tasks.content import atomic_snapshot, tree
        target=self.root/'sources'/'frozen'
        def interrupted(root, files):
            Path(root).mkdir(parents=True)
            (Path(root)/'first.py').write_text('partial')
            raise OSError('disk full')
        with patch('aka.light.tasks.content.materialize',side_effect=interrupted):
            with self.assertRaises(OSError):atomic_snapshot(target,{'a.py':b'correct'})
        self.assertFalse(target.exists())
        self.assertFalse(list((self.root/'.source-staging').glob('*')))
        atomic_snapshot(target,{'a.py':b'correct'})
        self.assertEqual(tree(target),{'a.py':b'correct'})

    def test_snapshot_creation_never_overwrites_corrupt_existing_evidence(self):
        from aka.light.tasks.content import atomic_snapshot
        target=self.root/'sources'/'frozen';target.mkdir(parents=True)
        (target/'a.py').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'Frozen candidate source changed'):
            atomic_snapshot(target,{'a.py':b'correct'})
        self.assertEqual((target/'a.py').read_bytes(),b'wrong')

    def test_explicit_research_inputs_are_checked_even_under_ignored_cache_paths(self):
        from aka.light.tasks.content import atomic_snapshot,tree,identity
        target=self.root/'sources'/'research'
        files={'kernel.cu':b'source','.aka-output/profile.bin':b'profile',
               'scratch/__pycache__/explicit.pyc':b'explicit'}
        atomic_snapshot(target,files,include_ignored=True)
        self.assertEqual(identity(tree(target,include_ignored=True)),identity(files))
        self.assertNotEqual(identity(tree(target)),identity(files))
        (target/'.aka-output/profile.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen candidate source changed'):
            atomic_snapshot(target,files,include_ignored=True)

    def test_task_research_preserves_explicit_report_inputs_and_their_hash(self):
        from aka.light.execution.package import encode_files
        from aka.light.tasks.content import tree,identity
        receipt=ExecutionReceipt('completed',{'exit_code':0,'stdout':'OK','stderr':''},'test-job')
        with patch('aka.light.tasks.execution.execute',return_value=receipt):
            result=self.service.command(self.task,argv=['true'],files=encode_files({'.aka-output/profile.bin':b'profile'}))
        record=self.service.store.evaluation(result['execution_id'])
        source=self.service.directory(self.task)/'sources'/record['candidate_id']
        self.assertEqual(identity(tree(source,include_ignored=True)),record['candidate_id'])
        self.assertEqual((source/'.aka-output/profile.bin').read_bytes(),b'profile')

    def test_pinned_remote_unknown_submission_routes_to_explicit_inspection(self):
        self.spec['target']={'backend':'remote','config':{'pod':'pod-a','uid':'uid-a'},'timeout_s':10}
        self.write_spec()
        task=self.service.submit(self.spec_path,idempotency_key='remote-inspection',start=False)['id']
        with patch('aka.light.tasks.execution.execute',side_effect=TimeoutError('handle commit interrupted')):
            with self.assertRaises(TimeoutError):self.service.measure(task)
        with self.service.store.connect() as db:
            eid=db.execute('SELECT id FROM evaluations WHERE task_id=?',(task,)).fetchone()[0]
        receipt=ExecutionReceipt('failed',{'exit_code':None,'submission_not_admitted':True,
            'error':{'code':'infra/submission_interrupted','message':'directory absent'}},'remote-reserved')
        with patch('aka.light.tasks.execution.execute',return_value=receipt) as execute:
            result=self.service.measure(task,evaluation_id=eid)
        self.assertTrue(execute.call_args.kwargs['recover_unrecorded'])
        self.assertEqual(result['state'],'finished')
        self.assertFalse(result['assessment']['complete'])
        self.assertEqual(result['assessment']['verdict'],'unknown')
        with self.assertRaises(ValueError):self.service.submit_candidate(task,eid)

    def test_known_job_recovery_uses_same_identity(self):
        def interrupted(plan, target, directory, evaluation_id, previous_job, on_job, control):
            on_job("dv-existing")
            raise TimeoutError("lost polling")
        with patch("aka.light.tasks.execution.execute", side_effect=interrupted):
            with self.assertRaises(TimeoutError):
                self.service.measure(self.task)
        with self.service.store.connect() as db:
            evaluation_id = db.execute("SELECT id FROM evaluations").fetchone()[0]
        receipt = ExecutionReceipt("completed", {"stdout": 'AKA_TASK_RESULT={"verdict":"pass","complete":true,"environment":"cpu-test-v1","metrics":[{"name":"cost","unit":"units","value":1}]}\n'}, "dv-existing")
        with patch("aka.light.tasks.execution.execute", return_value=receipt) as execute:
            result = self.service.measure(self.task, evaluation_id=evaluation_id)
        self.assertEqual(execute.call_args.args[4], "dv-existing")
        self.assertEqual(result["assessment"]["verdict"], "pass")

    def test_custom_adapter_owns_its_raw_result_format(self):
        adapter = '''from aka.contracts.task import Assessment,EvaluationPlan,Metric
from .helpers import parse
class Module:
 protocol_version=1
 capabilities=frozenset({"acceptance"})
 def validate(self,contract,config): pass
 def instructions(self,contract,config): return "Preserve module behavior."
 def delivery(self,contract,config): return {"entry":"compute.py:update"}
 def prepare(self,c):
  return EvaluationPlan("local",{"argv":["python3","runner.py"],"files":{"runner.py":"print('validated:42')"}})
 def interpret(self,c,r):
  value=parse(r.payload["stdout"])
  return Assessment("pass",True,(Metric("benefit",value,"points"),),"fixture")
'''
        (self.contract / "adapter.py").write_text(adapter)
        (self.contract / "helpers.py").write_text('def parse(text): return int(text.strip().split(":")[1])\n')
        self.spec["adapter"] = "python:adapter.py:Module"
        self.spec["objective"] = {"metrics": [{"name": "benefit", "unit": "points", "direction": "maximize"}]}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="custom", start=False)["id"]
        record = self.service.measure(task)
        candidate = self.service.submit_candidate(task, record["id"])
        self.assertEqual(record["assessment"]["metrics"][0]["value"], 42)
        self.assertEqual(self.service.results(task)["frontier"], [candidate["candidate_id"]])
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.service.measure(task, scope="diagnostic")

    def test_cancel_recovers_known_job_after_worker_crash(self):
        def interrupted(plan, target, directory, evaluation_id, previous_job, on_job, control):
            on_job("dv-in-flight")
            raise TimeoutError("worker lost polling")
        with patch("aka.light.tasks.execution.execute", side_effect=interrupted):
            with self.assertRaises(TimeoutError):
                self.service.measure(self.task)
        with patch.object(self.service, "start") as start:
            self.assertEqual(self.service.control(self.task, "cancel")["state"], "cancelling")
            start.assert_called_once_with(self.task, recover=True)
        with patch("aka.light.tasks.execution.execute", return_value=ExecutionReceipt("cancelled", {}, "dv-in-flight")) as execute:
            run(self.service.root, self.task)
        self.assertEqual(execute.call_args.args[4], "dv-in-flight")
        self.assertEqual(execute.call_args.args[6](), "cancel")
        self.assertEqual(self.service.status(self.task)["state"], "cancelled")
        self.assertEqual(self.service.status(self.task)["evaluations"], 1)

    def test_cancel_does_not_claim_unknown_submission_stopped(self):
        with patch("aka.light.tasks.execution.execute", side_effect=TimeoutError("ambiguous submission")):
            with self.assertRaises(TimeoutError):
                self.service.measure(self.task)
        with patch.object(self.service, "start"):
            self.service.control(self.task, "cancel")
        with self.assertRaisesRegex(ValueError, "outcome is unknown"):
            run(self.service.root, self.task)
        self.assertEqual(self.service.status(self.task)["state"], "cancelling")

    def test_custom_objective_cannot_admit_invalid_evidence(self):
        self.edit("COST=1\ndef update(state,x): return x\n")
        measured = self.service.measure(self.task)
        with patch.object(MetricObjective, "eligible", return_value=True):
            with self.assertRaisesRegex(ValueError, "does not satisfy"):
                self.service.submit_candidate(self.task, measured["id"])

    def test_evaluation_budget_applies_across_service_restarts(self):
        self.spec["budget"] = {"evaluations": 1}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="budget", start=False)["id"]
        self.service.measure(task)
        restored = TaskService(self.service.root)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            restored.measure(task)

    def test_record_from_other_task_cannot_be_submitted(self):
        record = self.service.measure(self.task)
        another = self.service.submit(self.spec_path, idempotency_key="another", start=False)["id"]
        with self.assertRaisesRegex(ValueError, "this task"):
            self.service.submit_candidate(another, record["id"])

    def test_corrupted_source_export_is_rejected(self):
        candidate = self.candidate()
        source = self.service.directory(self.task) / "sources" / candidate["source_id"] / "compute.py"
        source.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.service.export(self.task, candidate["candidate_id"], self.root / "invalid")

    def test_pause_blocks_new_measurement_without_erasing_candidates(self):
        candidate = self.candidate()
        self.service.control(self.task, "pause")
        self.assertEqual(self.service.status(self.task)["state"], "paused")
        with self.assertRaisesRegex(ValueError, "paused"):
            self.service.measure(self.task)
        self.assertEqual(self.service.results(self.task)["frontier"], [candidate["candidate_id"]])

    def test_adapter_interpretation_failure_keeps_receipt_without_reexecution(self):
        with patch("aka.light.tasks.adapters.command.Adapter.interpret", side_effect=RuntimeError("decoder failed")):
            with self.assertRaisesRegex(RuntimeError, "decoder"):
                self.service.measure(self.task)
        with self.service.store.connect() as db:
            evaluation_id = db.execute("SELECT id FROM evaluations").fetchone()[0]
        self.assertIsNotNone(self.service.store.evaluation(evaluation_id)["receipt"])
        with patch("aka.light.tasks.execution.execute", side_effect=AssertionError("must not run")):
            recovered = self.service.measure(self.task, evaluation_id=evaluation_id)
        self.assertEqual(recovered["assessment"]["verdict"], "pass")

    def test_persistent_agent_session_and_budget_admission(self):
        from aka.contracts.agent import TokenUsage
        self.spec["agent"] = {"runtime": "claude"}
        self.spec["budget"] = {"evaluations": 8, "episodes": 2, "tokens": 100}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="agent", start=False)["id"]
        calls = []
        class Session:
            supports_resume = True
            def invoke(inner, message, *, resume, reasoning_effort):
                calls.append(resume)
                self.assertIn('"name":"cost"', message)
                return SimpleNamespace(session_id="persistent-session", exit_status=0, timed_out=False,
                    terminal_usage=TokenUsage(2, 3, 0, 0, 5, "exact"))
            def close(inner): pass
        class Agent:
            def open_session(inner, *args, **kwargs): return Session()
            def close(inner): pass
        run(self.service.root, task, agent_factory=Agent)
        self.assertEqual(calls, [False, True])
        state = self.service.status(task)
        self.assertEqual((state["state"], state["episodes"], state["tokens"]), ("finished", 2, 10))
        candidate = self.service.results(task)["candidates"][0]["id"]
        self.service.feedback(task, candidate, {"note": "continue with integration feedback"})
        with patch.object(self.service, "start"):
            self.service.control(task, "resume", budget={"episodes": 3})
        run(self.service.root, task, agent_factory=Agent)
        continued = self.service.status(task)
        self.assertEqual(calls, [False, True, True])
        self.assertEqual((continued["episodes"], continued["tokens"]), (3, 15))
        self.assertEqual(continued["contract_id"], state["contract_id"])
        self.assertEqual(continued["evaluations"], state["evaluations"])
        self.assertEqual(self.service.store.task(task)["session"], "persistent-session")

    def test_native_codex_reopen_keeps_identity_and_primes_usage(self):
        from aka.agent.native import NativeSession
        from aka.agent.session_hooks import SessionHooks
        import unittest.mock as mock
        hooks = SessionHooks()
        with patch.object(hooks, "observer", return_value=mock.Mock()) as observer:
            session = NativeSession("codex", self.workspace, "saved-thread", lambda *a: None,
                                    session_hooks=hooks, reopen=True)
            self.assertEqual(session.session_id, "saved-thread")
            observer.return_value.observe.assert_called_once_with("saved-thread")
            session.close()

    def test_reopen_provider_rejected_thread_without_usage(self):
        from aka.agent.native import NativeSession
        from aka.agent.session_hooks import SessionHooks
        from aka.agent.codex_ledger import CodexLedgerError
        import unittest.mock as mock
        hooks = SessionHooks()
        with patch.object(hooks, "observer", return_value=mock.Mock()) as observer:
            observer.return_value.observe.side_effect = CodexLedgerError("Codex ledger exposed no new token_count delta")
            session = NativeSession("codex", self.workspace, "rejected-thread", lambda *a: None,
                                    session_hooks=hooks, reopen=True)
            self.assertEqual(session.session_id, "rejected-thread")
            self.assertFalse(session._ledger_usable)
            session.close()

    def test_local_command_escaping_is_argument_based(self):
        # No shell should interpret source paths or parent-supplied argv.
        self.spec["config"]["command"] = [sys.executable, "-c", "import sys;print(sys.argv[1])", "$(touch unwanted)"]
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="argv", start=False)["id"]
        result = self.service.measure(task)
        self.assertEqual(result["assessment"]["verdict"], "unknown")
        self.assertEqual(result["receipt"]["payload"]["stdout"].strip(), "$(touch unwanted)")

    def test_session_tool_reports_busy_without_changing_task_state(self):
        from aka.light.tasks.runtime import TaskRuntime, invoke
        from aka.light.tasks.service import TaskBusy
        before = self.service.status(self.task)
        with TaskRuntime(self.service, self.task) as runtime, \
             patch.dict("os.environ", runtime.environment), \
             patch.object(self.service, "measure", side_effect=TaskBusy("Already owned")):
            with self.assertRaisesRegex(TaskBusy, "Already owned"):
                invoke("measure", {})
        self.assertEqual(self.service.status(self.task), before)

    def test_worker_waits_for_existing_measurement_owner_without_reexecution(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from aka.light.tasks.service import TaskBusy, exclusive
        from aka.light.tasks.execution import execute
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()

        def watched_lock(path, **kwargs):
            if kwargs.get("wait"):
                waiting.set()
            return exclusive(path, **kwargs)

        def held_execution(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test owner was not released")
            return execute(*args, **kwargs)

        with patch("aka.light.tasks.execution.execute", side_effect=held_execution) as execution, \
             patch("aka.light.tasks.service.exclusive", side_effect=watched_lock):
            with ThreadPoolExecutor(max_workers=2) as pool:
                owner = pool.submit(self.service.measure, self.task, role="baseline")
                try:
                    self.assertTrue(entered.wait(5))
                    before = self.service.status(self.task)
                    with self.assertRaises(TaskBusy):
                        self.service.measure(self.task)
                    self.assertEqual(self.service.status(self.task)["state"], before["state"])
                    worker = pool.submit(run, self.service.root, self.task)
                    self.assertTrue(waiting.wait(5))
                    self.assertFalse(worker.done())
                finally:
                    release.set()
                self.assertEqual(owner.result(timeout=10)["assessment"]["verdict"], "pass")
                worker.result(timeout=10)
            self.assertEqual(execution.call_count, 2)  # One baseline, one seed evaluation.
        self.assertEqual(self.service.status(self.task)["state"], "awaiting_candidate")
        self.assertFalse(any(e["kind"] == "evaluation_interrupted" for e in self.service.events(self.task)))

    def test_interrupted_agent_does_not_reset_admission_or_usage(self):
        self.spec["agent"] = {"runtime": "claude"}
        self.spec["budget"] = {"evaluations": 8, "episodes": 2, "tokens": 100}
        self.write_spec()
        task = self.service.submit(self.spec_path, idempotency_key="crashed-agent", start=False)["id"]
        calls = []
        class Session:
            supports_resume = True
            session_id = "saved-session"
            def invoke(inner, *args, **kwargs):
                calls.append(True)
                raise RuntimeError("agent process lost")
            def close(inner): pass
        class Agent:
            def open_session(inner, *args, **kwargs): return Session()
            def close(inner): pass
        with self.assertRaisesRegex(RuntimeError, "process lost"):
            run(self.service.root, task, agent_factory=Agent)
        self.assertEqual(self.service.store.task(task)["session"], "saved-session")
        self.assertEqual(self.service.status(task)["episodes"], 1)
        run(self.service.root, task, agent_factory=Agent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.service.status(task)["state"], "needs_input")

    def test_agent_tools_use_task_scoped_host_and_reject_admin_operations(self):
        from aka.light.tasks.runtime import TaskRuntime, invoke
        import urllib.request
        import urllib.error
        with TaskRuntime(self.service, self.task) as runtime:
            with patch.dict("os.environ", runtime.environment):
                measured = invoke("measure", {})
                self.assertNotIn("plan", measured)
                self.assertNotIn("receipt", measured)
                candidate = invoke("submit-candidate", {"evaluation_id": measured["id"]})
                self.assertEqual(invoke("results", {})["frontier"], [candidate["candidate_id"]])
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    invoke("control", {"action": "cancel"})
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    invoke("measure", {"task_id": "different-task"})
            req = urllib.request.Request(runtime.environment["AKA_TASK_ENDPOINT"] + "/results", data=b"{}")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()

    def test_parent_api_binds_once_and_reconnects_to_same_workflow(self):
        import aka
        (self.root / "module.json").write_text(json.dumps(self.spec["config"]))
        request = dict(adapter="module.v1", source=self.root, baseline=self.source,
                       workload={"sequence": [1, 2, 3]}, validation={"exact": True},
                       execution=self.spec["target"], objective=self.spec["objective"],
                       root=self.service.root, key="short-api", agent={}, start=False)
        job = aka.submit(**request)
        self.assertEqual(aka.submit(**request).id, job.id)
        run(self.service.root, job.id)
        restored = aka.job(job.id, root=self.service.root)
        candidate = restored.results()["candidates"][0]
        target = self.root / "api-export"
        restored.export(candidate["id"], target)
        self.assertEqual(json.loads((target / "contract/workload.json").read_text()), {"sequence": [1, 2, 3]})
        self.assertEqual(restored.status()["state"], "awaiting_candidate")
        revised = aka.submit(**{**request, "key": "short-api-v2", "validation": {"exact": False}, "parent_task": job})
        self.assertNotEqual(revised.status()["contract_id"], job.status()["contract_id"])
        self.assertEqual(revised.results()["candidates"], [])

    def test_execution_cli_uses_bound_host_and_recovers_same_receipt(self):
        import os
        import subprocess
        from aka.light.tasks.runtime import TaskRuntime
        probe = self.root / "probe.py"
        probe.write_text("from pathlib import Path\nimport os\nPath('answer.txt').write_text(os.environ['PROBE'])\n" + "# padding\n" * 8000)
        with TaskRuntime(self.service, self.task) as runtime:
            env = {**os.environ, **runtime.environment,
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
            command = [sys.executable, "-m", "aka.light.execution.command_cli"]
            result = subprocess.run(command + ["--input", f"probe.py={probe}", "--output", "answer.txt",
                "--env", "PROBE=from-agent", "--", sys.executable, "probe.py"],
                env=env, cwd=self.workspace, capture_output=True, text=True, check=True)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["receipt"]["status"], "completed")
            import base64
            self.assertEqual(base64.b64decode(receipt["receipt"]["payload"]["artifacts"]["answer.txt"]["base64"]), b"from-agent")
            resumed = subprocess.run(command + ["--execution-id", receipt["execution_id"]],
                env=env, cwd=self.workspace, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(resumed.stdout), receipt)
        self.assertEqual(self.service.status(self.task)["executions"], 1)
        self.assertEqual(self.service.status(self.task)["evaluations"], 0)

    def test_local_execution_cancels_running_process(self):
        import threading
        from aka.light.execution.task_transport import execute
        cancel = threading.Event()
        timer = threading.Timer(0.3, cancel.set)
        timer.start()
        try:
            receipt = execute(EvaluationPlan("local", {
                "argv": [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"],
                "files": {}}), {"backend": "local", "timeout_s": 10}, self.root,
                "cancel-test", "", lambda _: None, lambda: "cancel" if cancel.is_set() else "run")
        finally:
            timer.join()
        self.assertEqual(receipt.status, "cancelled")
        self.assertIn("started", receipt.payload["stdout"])
        self.assertNotEqual(receipt.payload["exit_code"], 0)

    def test_research_execution_is_freeform_and_never_acceptance(self):
        from aka.light.tasks.runtime import TaskRuntime, invoke
        import base64
        with TaskRuntime(self.service, self.task) as runtime, patch.dict("os.environ", runtime.environment):
            with patch("aka.light.tasks.adapters.command.Adapter.prepare", side_effect=AssertionError("not an adapter command")):
                result = invoke("execute", {"argv": [sys.executable, "probe.py"],
                    "files": {"probe.py": "from pathlib import Path; print('research'); Path('output.txt').write_text('artifact')"},
                    "outputs": ["output.txt"]})
            self.assertEqual(result["receipt"]["status"], "completed")
            artifact = result["receipt"]["payload"]["artifacts"]["output.txt"]["base64"]
            self.assertEqual(base64.b64decode(artifact), b"artifact")
            with self.assertRaisesRegex(ValueError, "candidate acceptance"):
                self.service.submit_candidate(self.task, result["execution_id"])
            with patch("aka.light.tasks.execution.execute", side_effect=AssertionError("no replay")):
                self.assertEqual(invoke("execute", {"execution_id": result["execution_id"]}), result)

    def test_installed_binding_and_executor_extensions_share_common_loop(self):
        import aka
        from aka.light.tasks.api import bind_package
        from aka.light.execution.contract import BackendCapabilities, JobResult, ResourceIdentity
        (self.root / "module.json").write_text(json.dumps(self.spec["config"]))
        def bind(**values): return bind_package(adapter="module.v1", **values)
        entry = SimpleNamespace(load=lambda: bind)
        with patch("aka.light.tasks.api.entry_points", return_value=[entry]):
            job = aka.submit(adapter="team.module.v1", source=self.root, baseline=self.source,
                execution={"backend": "test-provider"}, objective=self.spec["objective"],
                root=self.service.root, key="plugin", agent={}, start=False)
        commands = []
        class Provider:
            capabilities = BackendCapabilities()
            resource_identity = ResourceIdentity("test-provider")
            def __init__(self, **context): pass
            def run(self, command):
                commands.append(command)
                return JobResult(0, 'AKA_TASK_RESULT={"verdict":"pass","complete":true,"environment":"test","metrics":[{"name":"cost","unit":"units","value":1}]}', exit_code=0)
        with patch("importlib.metadata.entry_points", return_value=[SimpleNamespace(load=lambda: Provider)]):
            run(self.service.root, job.id)
        self.assertEqual(len(commands), 2)
        self.assertEqual(len(job.results()["candidates"]), 1)


class ExecutionRouteTests(unittest.TestCase):
    def test_rejected_declaration_is_failed_but_ambiguous_submission_is_not_replayed(self):
        from aka.light.execution.task_transport import execute
        class GatewayError(Exception):
            def __init__(self, status):
                self.status, self.reason, self.details, self.payload = status, "invalid_source", {"field": "candidate"}, {"status": status}
        with patch.dict(sys.modules, {"atrex_gateway_client.errors": SimpleNamespace(GatewayError=GatewayError)}):
            for status in (400, 422, 503, None):
                with self.subTest(status=status), patch("aka.light.execution.task_transport.gateway_client") as factory:
                    client = factory.return_value
                    client.submit_job.side_effect = GatewayError(status)
                    args = (EvaluationPlan("dev", {"command": "true", "files": {}}),
                            {"backend": "agate", "gpu": "H100"}, Path("unused"), "rejection", "",
                            lambda value: self.fail("Rejected request produced a job"), lambda: "run")
                    if status in (400, 422):
                        receipt = execute(*args)
                        self.assertEqual(receipt.status, "failed")
                        self.assertEqual(receipt.payload["error"]["details"], {"field": "candidate"})
                    else:
                        with self.assertRaises(GatewayError):
                            execute(*args)
                    self.assertEqual(client.submit_job.call_count, 1)
                    client.get_job.assert_not_called()

    def test_binary_archive_transport_reuses_saved_upload(self):
        from aka.light.evaluation.source_transport import uploaded_request
        from aka.light.execution.package import encode_files
        import io,zipfile
        files = {"payload.bin": b"\x00\xff", "main.py": b"print('done')"}
        request = {"command": "python3 main.py", "files": encode_files(files), "spec": {"target_hardware": ["H100"]}}
        uploaded = []
        def reserve(method, path, body, timeout):
            return {"job_id": "reserved", "uploads": [{"path": row["path"], "put_url": "test-url", "upload_ref": "reference"} for row in body["files"]]}
        with tempfile.TemporaryDirectory() as temporary:
            wire = uploaded_request(request, "binary", kind="dev", state_directory=temporary,
                json_call=reserve, upload=lambda url,data: uploaded.append(data))
            again = uploaded_request(request, "binary", kind="dev", state_directory=temporary,
                json_call=lambda *a: self.fail("duplicate reservation"), upload=lambda *a: self.fail("duplicate upload"))
        self.assertEqual(wire, again)
        with zipfile.ZipFile(io.BytesIO(uploaded[0])) as archive:
            self.assertEqual({name:archive.read(name) for name in archive.namelist()}, files)

    def test_native_manifest_preserves_declared_files_build_and_dependencies(self):
        from aka.contracts.task import TaskContext
        from aka.light.tasks.adapters.atrex import Adapter
        from aka.light.evaluation.source_transport import ARCHIVES
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); candidate=root/'candidate';candidate.mkdir()
            definition={"schema_version":1,"name":"native","version":"1","source":{"type":"local","files":["kernel.cu","helper.cuh"]},
                "entry_point":{"type":"cuda","symbol":"run","build":{"toolchain":"torch_extension","sources":["kernel.cu"],"cuda_cflags":["-O3"]}},
                "dependencies":[{"id":"headers","kind":"git","repository":"https://example.org/headers.git","ref":"v1"}]}
            (candidate/'solution.json').write_text(json.dumps(definition));(candidate/'kernel.cu').write_text('#include "helper.cuh"\n');(candidate/'helper.cuh').write_text('// header\n')
            contract=Path(__file__).resolve().parents[1]/'examples/task-contracts/native-vector/contract'
            plan=Adapter().prepare(TaskContext(contract,candidate,{}, {"backend":"agate"},"acceptance"))
            self.assertEqual(plan.request['candidate']['entry_point'], definition['entry_point'])
            self.assertEqual(plan.request['candidate']['dependencies'], definition['dependencies'])
            self.assertEqual(set(plan.request[ARCHIVES]['archives/candidate.tar.gz']), {'kernel.cu','helper.cuh'})

    def test_pi_current_cli_resume_and_provider_failure(self):
        from aka.agent.runtime import build_agent_runtime
        from aka.agent.native import fresh_session_command,resume_session_command,session_id_from_stream
        runtime=build_agent_runtime('pi')
        env={'ATREX_PI_SESSION_SETTINGS':json.dumps({'provider':'example','model':'model-a'})}
        first=fresh_session_command('prompt','hint','max','pi',runtime=runtime,environment=env)
        again=resume_session_command('next','native-id','high','pi',runtime=runtime,environment=env)
        self.assertNotIn('--approve',first);self.assertNotIn('--session-id',first)
        self.assertIn('--print',first);self.assertIn('model-a',first)
        self.assertEqual(again[again.index('--session')+1],'native-id')
        stream=json.dumps({'type':'session','id':'native-id'})+'\n'+json.dumps({'type':'message_end','message':{'role':'assistant','stopReason':'error','errorMessage':'rate limit'}})
        self.assertEqual(session_id_from_stream('pi',stream,'hint'),'native-id')
        self.assertEqual(runtime._adapter.validated_exit_status(stream,0),1)

    def test_removed_gateway_operations_never_submit_or_poll(self):
        from aka.light.tasks.execution import execute
        for kind in ("profile", "compile", "disassemble"):
            for job in ("", "old-job"):
                with self.subTest(kind=kind, job=job), patch("aka.light.execution.task_transport.gateway_client") as client:
                    with self.assertRaisesRegex(ValueError, "eval or dev"):
                        execute(EvaluationPlan(kind, {}), {"backend": "agate"}, Path("unused"),
                                "eval-route", job, lambda value: None, lambda: "run")
                    client.assert_not_called()

    def test_dev_carries_all_diagnostic_commands_and_preserves_receipt(self):
        from aka.light.tasks.execution import execute
        for command in ("ncu --version", "nvcc --version", "cuobjdump --version"):
            with self.subTest(command=command), patch("aka.light.execution.task_transport.gateway_client") as factory:
                client = factory.return_value
                client.submit_job.return_value = {"job_id": "dv-diagnostic"}
                raw = {"status": "succeeded", "command_ok": True, "result": {"stdout": "tool output"}}
                client.get_job.return_value = raw
                jobs = []
                receipt = execute(EvaluationPlan("dev", {"command": command, "files": {}}),
                                  {"backend": "agate", "gpu": "H100", "timeout_s": 60},
                                  Path("unused"), "eval-diagnostic", "", jobs.append, lambda: "run")
                kind, request = client.submit_job.call_args.args
                self.assertEqual((kind, request["command"]), ("dev", command))
                self.assertEqual(request["idempotency_key"], "eval-diagnostic")
                self.assertEqual(jobs, ["dv-diagnostic"])
                self.assertEqual((receipt.status, receipt.payload), ("completed", raw))

    def test_dev_success_still_requires_command_success(self):
        from aka.light.tasks.execution import execute
        with patch("aka.light.execution.task_transport.gateway_client") as factory:
            factory.return_value.submit_job.return_value = {"job_id": "dv-failed-command"}
            factory.return_value.get_job.return_value = {"status": "succeeded", "command_ok": False}
            receipt = execute(EvaluationPlan("dev", {"command": "false"}),
                              {"backend": "agate", "gpu": "H100"}, Path("unused"),
                              "eval-failed-command", "", lambda value: None, lambda: "run")
            self.assertEqual(receipt.status, "failed")

    def test_recovered_job_uses_short_reads_and_observes_cancellation(self):
        from aka.light.tasks.execution import execute
        with patch("aka.light.execution.task_transport.gateway_client") as factory, \
             patch("aka.light.execution.task_transport.time.sleep") as sleep:
            client = factory.return_value
            client.get_job.side_effect = [{"status": "running"}, {"status": "cancelled"}]
            controls = iter(["run", "cancel"])
            receipt = execute(EvaluationPlan("dev", {"command": "unused"}),
                              {"backend": "agate", "gpu": "H100"}, Path("unused"),
                              "eval-existing", "dv-existing", lambda value: None,
                              lambda: next(controls))
            client.submit_job.assert_not_called()
            client.cancel_job.assert_called_once_with("dv-existing")
            self.assertEqual(client.get_job.call_count, 2)
            for args in client.get_job.call_args_list:
                self.assertEqual(args.args, ("dv-existing",))
                self.assertEqual(args.kwargs, {"wait": False})
            sleep.assert_called_once_with(3)
            self.assertEqual(receipt.status, "cancelled")


class ObjectiveTests(unittest.TestCase):
    def test_units_missing_metrics_and_environment(self):
        goal = MetricObjective({"metrics": [{"name": "throughput", "unit": "tokens/s", "direction": "maximize"}]})
        a = Assessment("pass", True, (Metric("throughput", 10, "tokens/s"),), "a")
        b = Assessment("pass", True, (Metric("throughput", 20, "tokens/s"),), "a")
        self.assertEqual(goal.compare(b, a), "better")
        self.assertEqual(goal.compare(b, Assessment("pass", True, a.metrics, "b")), "incomparable")
        self.assertFalse(goal.eligible(Assessment("pass", True, (Metric("throughput", 10, "ms"),))))
        self.assertFalse(goal.eligible(Assessment("pass", True)))

    def test_pareto_and_resource_constraints(self):
        goal = MetricObjective({"ordering": "pareto", "metrics": [
            {"name": "latency", "unit": "us", "direction": "minimize"},
            {"name": "memory", "unit": "bytes", "direction": "minimize"}],
            "constraints": [{"name": "memory", "unit": "bytes", "operator": "le", "limit": 50}]})
        a = Assessment("pass", True, (Metric("latency", 10, "us"), Metric("memory", 10, "bytes")), "gpu")
        b = Assessment("pass", True, (Metric("latency", 5, "us"), Metric("memory", 20, "bytes")), "gpu")
        self.assertEqual(goal.compare(a, b), "incomparable")
        self.assertFalse(goal.eligible(Assessment("pass", True, (Metric("latency", 1, "us"), Metric("memory", 100, "bytes")), "gpu")))


if __name__ == "__main__":
    unittest.main()
