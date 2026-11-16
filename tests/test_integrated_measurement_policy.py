"""Public launch-to-request checks; no GPU or network is required."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from aka.bootstrap.workflow_request import register_request_options, request_arguments
from aka.contracts.evaluation import EvaluationConfiguration
from aka.contracts.workflow import CampaignRuntimeSettings
from aka.contracts.workflow_launch import decode_launch
from aka.evaluation.abba_shape_compat import full_reference_abba_batch_request
from aka.evaluation.gateway_request import prepare_request
from aka.evaluation.native_abba import parse_native_abba
from aka.evaluation.objectives import metadata_performance_metrics
from aka.evaluation.shapes import shape_batch_request
from aka.evaluation.transport_results import _merge_optimizer_results, _with_workspace_performance_score


class MeasurementPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {"performance_objective": "shape_latency_geometric_mean"}
        for name, text in {
            "kernel.py": "class Model: pass\n", "reference.py": "class Model: pass\n",
            "input.py": "def get_inputs(): return []\n", "shapes.json": '{"0": {"N": 16}}',
            "metadata.json": json.dumps(self.metadata),
        }.items():
            (self.root / name).write_text(text)

    def request(self, runtime, args=()):
        defaults = EvaluationConfiguration(warmup_iters=runtime.warmup_iters,
            bench_iters=runtime.bench_iters, evaluation_step_timeout=runtime.evaluation_step_timeout)
        return prepare_request(self.root, "L20N", runtime.timeout, [],
            ["python3", "test_kernel.py", *args], "run", reference_dir=self.root,
            measurement_defaults=defaults.measurement_defaults)

    def test_public_cli_values_survive_frozen_launch_and_request(self):
        parser = argparse.ArgumentParser()
        register_request_options(parser)
        options = parser.parse_args(["--problem", str(self.root), "--platform", "L20N",
            "--optimization-mode", "fast", "--warmup-iters", "0", "--bench-iters", "87",
            "--evaluation-step-timeout", "173"])
        runtime = decode_launch(request_arguments(options, parser)).request.runtime
        payload = self.request(runtime)
        self.assertEqual(payload["options"]["bench_iters"], 87)
        self.assertEqual(payload["runner_overrides"]["warmup_iters"], 0)
        self.assertEqual(payload["runner_overrides"]["candidate_timeout_s"], 173)
        self.assertEqual(payload["runner_overrides"]["perf_timeout_s"], 173)

    def test_reference_directories_use_public_launch_pairs(self):
        parser=argparse.ArgumentParser();register_request_options(parser)
        args=parser.parse_args(['--problem',str(self.root),'--platform','L20N',
            '--agent-reference-paths','knowledge='+str(self.root)])
        launch=decode_launch(request_arguments(args,parser))
        self.assertEqual(launch.request.settings.agent_reference_paths,(('knowledge',str(self.root.resolve())),))

    def test_defaults_and_explicit_agent_options(self):
        runtime = CampaignRuntimeSettings()
        payload = self.request(runtime)
        self.assertEqual(payload["options"]["bench_iters"], 20)
        self.assertEqual(payload["runner_overrides"]["warmup_iters"], 10)
        self.assertEqual(payload["runner_overrides"]["candidate_timeout_s"], 60)
        self.assertEqual(payload["runner_overrides"]["perf_timeout_s"], 60)
        explicit = self.request(runtime, ("--warmup", "7", "--timed-runs", "91", "--perf-timeout-s", "222"))
        self.assertEqual(explicit["options"]["bench_iters"], 91)
        self.assertEqual(explicit["runner_overrides"]["warmup_iters"], 7)
        self.assertEqual(explicit["runner_overrides"]["perf_timeout_s"], 222)

    def test_reject_invalid_sampling_settings(self):
        for values in ({"bench_iters": 0}, {"bench_iters": True},
                       {"warmup_iters": -1}, {"warmup_iters": False}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CampaignRuntimeSettings(**values)

    def test_eval_selector_and_isolated_abba_reference(self):
        payload = self.request(CampaignRuntimeSettings())
        payload["reference"]["shapes"]["1"] = {"N": 32}
        original = copy.deepcopy(payload)
        ordinary = shape_batch_request(payload, ["1"])
        self.assertEqual(ordinary["options"]["shape_ids"], ["1"])
        self.assertEqual(set(ordinary["reference"]["shapes"]), {"0", "1"})
        abba = full_reference_abba_batch_request(payload, ["1"])
        self.assertEqual(set(abba["reference"]["shapes"]), {"1"})
        self.assertNotIn("shape_ids", abba["options"])
        self.assertEqual(abba["options"]["shape_shards"], 1)
        self.assertTrue(abba["lock_clocks"])
        self.assertNotIn("gpu_clock_mhz", abba["runner_overrides"])
        self.assertEqual(payload, original)

    def test_complete_shape_union_uses_task_objective(self):
        metadata = {"performance_objective": "shape_latency_geometric_mean"}
        rows = [{"all_pass": True, "latency_us_by_shape": {"0": 1, "1": 1}},
                {"all_pass": True, "latency_us_by_shape": {"2": 64}}]
        result = _merge_optimizer_results(rows, ["0", "1", "2"], metadata)
        self.assertTrue(result["all_pass"])
        self.assertAlmostEqual(result["performance_score"], 4)
        self.assertIs(_with_workspace_performance_score(self.root, result), result)
        metadata = {"performance_objective": "shape_speedup_geometric_mean", "shapes": {
            "0": {"production_performance": {"performance_us": 16}},
            "1": {"production_performance": {"performance_us": 1}},
        }}
        metrics, errors = metadata_performance_metrics(metadata, ["0", "1"], {"0": 1, "1": 1})
        self.assertEqual(errors, [])
        self.assertAlmostEqual(metrics["performance_score"], 4)
        self.assertAlmostEqual(metrics["speedup_vs_ref_mean"], 8.5)

    def test_sdk_verdict_cannot_be_overridden_by_successful_rows(self):
        result = {"passed": {"compile": {"status": "passed"},
                             "correctness": {"0": {"status": "passed"}}},
                  "correctness": {"shapes": {"0": {}}},
                  "performance": {"shapes": {"0": {"samples": [{"end_to_end_time_ms": 0.01}]}}}}
        rows = [{"index": index, "revision": side, "label": label, "repeat": repeat,
                 "result": copy.deepcopy(result)}
                for index, (side, label, repeat) in enumerate([
                    ("baseline", "A", 0), ("candidate", "B", 0),
                    ("candidate", "B", 1), ("baseline", "A", 1)])]
        block = {"passed": {"abba": {"status": "passed"}}, "abba": {"runs": rows}}
        job = {"status": "succeeded", "result": {"abba": {"sdk_results": [block]}}}
        self.assertEqual(len(parse_native_abba(job, 2, ["0"])["runs"]), 4)
        block["passed"]["abba"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "rejected a comparison"):
            parse_native_abba(job, 2, ["0"])
        block["passed"]["abba"]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "completed comparison verdict"):
            parse_native_abba(job, 2, ["0"])


if __name__ == "__main__":
    unittest.main()
