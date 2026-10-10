"""CPU regressions for the public GPU integration runner; no GPU/model required."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location("local_sandbox_runner", Path(__file__).with_name("runner.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class LocalSandboxRunnerTests(unittest.TestCase):
    def test_disposable_fixture_has_native_evaluator_ancestor_without_modifying_bench(self):
        layout_spec = importlib.util.spec_from_file_location("operator_layout", worker.SOURCE_ROOT / "orchestrator/operator_layout.py")
        layout = importlib.util.module_from_spec(layout_spec)
        layout_spec.loader.exec_module(layout)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench, work = root / "source-bench", root / "work"
            (bench / "scripts").mkdir(parents=True)
            (bench / "src/atrex_bench").mkdir(parents=True)
            (bench / "scripts/run_eval.py").write_text("# canonical evaluator")
            (bench / "src/atrex_bench/__init__.py").write_text("# public package")
            before = {str(p.relative_to(bench)): p.read_bytes() for p in bench.rglob("*") if p.is_file()}
            fixture = worker.prepare_fixture(work, bench)
            self.assertEqual(layout.find_atrex_bench_root(fixture), work / "bench")
            self.assertEqual((work / "bench/scripts/run_eval.py").read_bytes(), (bench / "scripts/run_eval.py").read_bytes())
            (fixture / "metadata.json").write_text("measured reference data")
            self.assertEqual({str(p.relative_to(bench)): p.read_bytes() for p in bench.rglob("*") if p.is_file()}, before)

    @mock.patch.dict(os.environ, {"PYTHONPATH": "/private", "UNRELATED_SECRET": "not-for-test", "ANTHROPIC_AUTH_TOKEN": "model-secret"}, clear=False)
    def test_gateway_environment_excludes_unrelated_and_model_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = worker.clean_environment(Path(temporary), model=False)
        self.assertFalse(set(env) & {"PYTHONPATH", "UNRELATED_SECRET", "ANTHROPIC_AUTH_TOKEN"})
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], "/dev/null")


    def campaign(self, root):
        (root / "memory").mkdir()
        (root / ".atrex_long_horizon").mkdir()
        for number in range(3):
            (root / f"memory/v{number}.json").write_text(json.dumps({"version": f"v{number}", "correctness": {"status": "PASS"}, "performance": {"latency_us": 1.0}}))
        attempt = {"mode": "fast", "status": "candidate_ready", "accepted": False, "verification": {
            "gate": "FAIL", "error": "fast evaluator performance-score improvement did not exceed threshold",
            "incumbent_performance_score": 1.0, "runs": [{"exit_code": 0, "result": {
                "all_pass": True, "latency_us_by_shape": {"0": 1, "1": 1, "2": 1}, "latency_us_geomean": 1}}]}}
        (root / ".atrex_long_horizon/state.json").write_text(json.dumps({"attempts": [attempt]}))
        (root / "kernel.py").write_text("# fixture")
        (root / "solution.json").write_text('{"entry_point":"original"}')
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], cwd=root, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        attempt.update(base_commit=head, episode_head=head)
        episode = root / ".atrex_long_horizon/episodes/e0001"
        archived = episode / "archive/worktree_files"
        archived.mkdir(parents=True)
        (episode / "episode_runtime").mkdir()
        (archived / "kernel.py").write_text("# measured candidate")
        (archived / "solution.json").write_text('{"entry_point":"kernel.py"}')
        (episode / "archive/git.json").write_text(json.dumps({"head": head, "base_commit": head}))
        receipt = {"schema_version": 3,
                   "kernel_sha256": hashlib.sha256((archived / "kernel.py").read_bytes()).hexdigest(),
                   "solution_sha256": hashlib.sha256((archived / "solution.json").read_bytes()).hexdigest(),
                   "result": {"all_pass": True, "performance_objective": "shape_speedup_arithmetic_mean",
                              "latency_us_by_shape": {"0": 1, "1": 1, "2": 1},
                              "latency_us_geomean": 1, "performance_score": 1}}
        (episode / "episode_runtime/evaluations.jsonl").write_text(json.dumps(receipt) + "\n")
        (root / ".atrex_long_horizon/state.json").write_text(json.dumps({"attempts": [attempt]}))
        return attempt


    def test_valid_slower_candidate_passes_but_wrong_result_and_missing_commit_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = self.campaign(root)
            self.assertFalse(worker.validate_campaign(root)["candidate_promoted"])
            attempt["status"] = "pivot"
            attempt["verification"] = None
            (root / ".atrex_long_horizon/state.json").write_text(json.dumps({"attempts": [attempt]}))
            self.assertFalse(worker.validate_campaign(root)["candidate_promoted"])
            receipt_path = root / ".atrex_long_horizon/episodes/e0001/episode_runtime/evaluations.jsonl"
            receipt = json.loads(receipt_path.read_text())
            receipt["result"]["all_pass"] = False
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, "passing candidate"):
                worker.validate_campaign(root)
            (root / "memory/v1.json").write_text('{"version":"v1"}')
            with self.assertRaisesRegex(RuntimeError, "differs from Git"):
                worker.validate_campaign(root)


    def test_pivot_requires_final_candidate_fingerprints_and_complete_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = self.campaign(root)
            receipt_path = root / ".atrex_long_horizon/episodes/e0001/episode_runtime/evaluations.jsonl"
            receipt = json.loads(receipt_path.read_text())
            for field, bad_value in (("kernel_sha256", "stale"), ("solution_sha256", "stale"), ("schema_version", 2)):
                with self.subTest(field=field):
                    receipt_path.write_text(json.dumps(dict(receipt, **{field: bad_value})))
                    with self.assertRaisesRegex(RuntimeError, "passing candidate"):
                        worker.validate_campaign(root)
            incomplete = dict(receipt, result=dict(receipt["result"], latency_us_by_shape={"0": 1}))
            receipt_path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(RuntimeError, "passing candidate"):
                worker.validate_campaign(root)
            receipt_path.write_text(json.dumps(receipt))
            for status, violation in (("blocked", None), ("pivot", "invalid candidate")):
                with self.subTest(status=status):
                    attempt.update(status=status, violation=violation)
                    (root / ".atrex_long_horizon/state.json").write_text(json.dumps({"attempts": [attempt]}))
                    with self.assertRaisesRegex(RuntimeError, "passing candidate"):
                        worker.validate_campaign(root)


    def test_pivot_final_candidate_gets_fresh_evaluation_without_incumbent_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = self.campaign(root)
            attempt["status"] = "pivot"
            (root / ".atrex_long_horizon/state.json").write_text(json.dumps({"attempts": [attempt]}))
            path = root / ".atrex_long_horizon/episodes/e0001/episode_runtime/evaluations.jsonl"
            stale = json.loads(path.read_text())
            stale["kernel_sha256"] = "stale-trial"
            path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(RuntimeError, "passing candidate"):
                worker.validate_campaign(root)
            originals = {name: (root / name).read_bytes() for name in ("kernel.py", "solution.json")}
            selected = worker.archived_candidate(root, attempt)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            def evaluate(command, cwd, env, log, timeout):
                self.assertEqual({name: (cwd / name).read_bytes() for name in selected}, selected)
                self.assertIn("--no-memory", command)
                self.assertNotIn("--correctness-only", command)
                return "[test_kernel] RESULT_JSON=" + json.dumps(stale["result"])
            with mock.patch.object(worker, "run", side_effect=evaluate) as run:
                receipt = worker.verify_archived_candidate(root, artifacts, {}, root / "sandbox.py")
                run.assert_called_once()
            self.assertEqual({name: (root / name).read_bytes() for name in originals}, originals)
            self.assertEqual(json.loads(path.read_text()), stale)
            self.assertEqual(json.loads((artifacts / "v2-final-receipt.json").read_text()), receipt)
            self.assertFalse(worker.validate_campaign(root, receipt)["candidate_promoted"])


    def test_independent_candidate_failure_or_mutation_restores_canonical_workspace(self):
        for failure in ("wrong-result", "missing-case", "mutation", "executor-error"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                attempt = self.campaign(root)
                originals = {name: (root / name).read_bytes() for name in ("kernel.py", "solution.json")}
                selected = worker.archived_candidate(root, attempt)
                receipt = json.loads((root / ".atrex_long_horizon/episodes/e0001/episode_runtime/evaluations.jsonl").read_text())
                artifacts = root / "artifacts"
                artifacts.mkdir()
                def evaluate(command, cwd, env, log, timeout):
                    if failure == "executor-error":
                        raise RuntimeError("evaluation failed")
                    if failure == "wrong-result":
                        receipt["result"]["all_pass"] = False
                    if failure == "missing-case":
                        receipt["result"]["latency_us_by_shape"] = {"0": 1}
                    if failure == "mutation":
                        (cwd / "kernel.py").write_text("# unexpected modification")
                    return "[test_kernel] RESULT_JSON=" + json.dumps(receipt["result"])
                with mock.patch.object(worker, "run", side_effect=evaluate):
                    with self.assertRaises(RuntimeError):
                        worker.verify_archived_candidate(root, artifacts, {}, root / "sandbox.py")
                self.assertEqual({name: (root / name).read_bytes() for name in originals}, originals)
                self.assertEqual({name: (artifacts / "v2-final-candidate" / name).read_bytes() for name in selected}, selected)


    def test_amd_visibility_and_virtualenv_python_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "0"}), mock.patch.object(sys, "executable", "/opt/venv/bin/python"):
            environment = worker.clean_environment(Path(temporary), model=False)
            self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], "3")
            self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "0")
            self.assertTrue(environment["PATH"].startswith("/opt/venv/bin:"))

    def test_nonempty_work_directory_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "existing-result.json"
            evidence.write_text("keep this evidence")
            process = subprocess.run([sys.executable, str(Path(worker.__file__)),
                                      "--work-dir", str(root), "--bench-root", str(root),
                                      "--platform", "MI308X"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("--work-dir must be empty", process.stderr)
            self.assertEqual(evidence.read_text(), "keep this evidence")
            self.assertFalse((root / "artifacts").exists())

    def test_execution_failure_returns_nonzero_and_redacts_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            args = ["runner", "--work-dir", str(root), "--bench-root", temporary,
                    "--platform", "MI308X"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(worker.signal, "signal"), \
                    mock.patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "test-secret"}), \
                    mock.patch.object(worker, "execute", side_effect=RuntimeError("failure test-secret")):
                self.assertEqual(worker.main(), 1)
            result = json.loads((root / "artifacts/result.json").read_text())
            self.assertEqual(result["status"], "FAIL")
            self.assertNotIn("test-secret", result["error"])
            self.assertGreaterEqual(result["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
