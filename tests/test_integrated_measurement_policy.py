"""Public launch-to-request checks; no GPU or network is required."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
        self.assertTrue(payload["lock_clocks"])
        self.assertNotIn("lock_clocks", payload["options"])
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

    def assert_managed_clocks(self, payload):
        self.assertIs(payload['lock_clocks'], True)
        for values in (payload, payload.get('options', {}), payload.get('runner_overrides', {})):
            self.assertNotIn('gpu_clock_mhz', values)
            self.assertNotIn('graphics_mhz', values)
        self.assertNotIn('lock_clocks', payload.get('options', {}))
        self.assertNotIn('lock_clocks', payload.get('runner_overrides', {}))

    def test_runtime_eval_and_abba_sdk_payloads_request_managed_clocks(self):
        from aka.contracts.execution import BackendCapabilities
        from aka.contracts.workspace import RevisionContent
        from aka.evaluation.gateway_command import GatewayCommandEvaluator
        from aka.evaluation.tool_requests import request_from_argv
        from aka.execution.gateway import sdk
        from aka.execution.gateway.requests import operation_request
        from aka.workflow.light.measurement_policy import measurement_defaults
        source=RevisionContent({'kernel.py':(self.root/'kernel.py').read_bytes()})
        workspace=Mock()
        workspace.read_source.return_value=source
        workspace.read_working.return_value=RevisionContent({'solution.json':None})
        capabilities=BackendCapabilities(operations=frozenset({'run','same_allocation_abba'}),
            prepared_requests=True,source_archives=True,output_options=True)
        config=EvaluationConfiguration(hardware='L20N',shape_batch_size=10,evaluation_step_timeout=60)
        defaults=measurement_defaults(gpu_clock_mhz=1350)
        evaluator=GatewayCommandEvaluator(execution_capabilities=capabilities,shape_batch_size=10)
        for shapes in (1,12):
            (self.root/'shapes.json').write_text(json.dumps({str(i):{'N':i+1} for i in range(shapes)}))
            for mode,extra in [('full',[]),('correctness_only',['--mode','correctness_only']),
                               ('abba',['--baseline-commit','a'*40])]:
                with self.subTest(mode=mode,shapes=shapes):
                    tool=request_from_argv(['--kind','evaluate',*extra],workspace=self.root,evaluation_input=None,
                        configuration=config,workspace_provider=workspace,defaults=defaults,baseline_source=source)
                    self.assert_managed_clocks(json.loads(tool.request.submission)['request'])
                    prepared=evaluator.prepare(tool.request)
                    self.assertEqual(len(prepared.steps),1 if shapes==1 else 2)
                    covered=[]
                    for step,identity in zip(prepared.steps,prepared.measurement_identities):
                        payload=json.loads(step.submission)['request']
                        api=Mock()
                        with patch.object(sdk,'client',return_value=api):
                            sdk.request('endpoint','POST','/v1/jobs/eval',operation_request(payload,'run'),30)
                        self.assert_managed_clocks(api.submit_job.call_args.args[1])
                        self.assertIs(identity.parameters['lock_clocks'],True)
                        self.assertEqual(api.submit_job.call_args.args[1]['options']['shape_shards'],1)
                        selected=list(dict.fromkeys(point.shape_id for point in identity.points))
                        covered.extend(selected)
                        if mode=='abba':
                            self.assertNotIn('shape_ids',payload['options'])
                            self.assertEqual(set(payload['reference']['shapes']),set(selected))
                        else:
                            self.assertEqual(payload['options']['shape_ids'],selected)
                            self.assertEqual(len(payload['reference']['shapes']),shapes)
                    self.assertEqual(covered,[str(i) for i in range(shapes)])

    def test_native_selector_keeps_outer_batches_and_exact_measurement_coverage(self):
        from aka.evaluation.gateway_measurement import GatewayMeasurement
        from aka.evaluation.measurement import gateway_identity
        payload=self.request(CampaignRuntimeSettings())
        payload['reference']['shapes']={str(i):{'N':i+1} for i in range(12)}
        payload['options'].update(shape_ids=['11','2'],shape_shards=8)
        original=copy.deepcopy(payload)
        measurement=GatewayMeasurement(payload,workspace=self.root,hardware='L20N',private=False,batch_size=1)
        self.assertEqual(measurement.shape_ids,['2','11'])
        self.assertEqual(measurement.batches,[['2'],['11']])
        for index,selected in enumerate(measurement.batches):
            request=measurement.request_for(index)
            self.assertEqual(request['options']['shape_shards'],1)
            self.assertEqual(request['options']['shape_ids'],selected)
            self.assertEqual(request['reference'],original['reference'])
            self.assertEqual([point.shape_id for point in gateway_identity(request,{}).points],selected)
        self.assertEqual(payload,original)

    def test_prepared_requests_remove_stale_clock_fields_without_mutation(self):
        from aka.evaluation.gateway_measurement import GatewayMeasurement
        from aka.evaluation.measurement import gateway_identity
        payload=self.request(CampaignRuntimeSettings())
        payload.update(lock_clocks=False,gpu_clock_mhz=1350,graphics_mhz=1350)
        payload['options'].update(lock_clocks=True,graphics_mhz=1350)
        payload['runner_overrides'].update(gpu_clock_mhz=1350,lock_clocks=True)
        original=copy.deepcopy(payload)
        measurement=GatewayMeasurement(payload,workspace=self.root,hardware='L20N',private=False,batch_size=10)
        for request in (measurement.request_for(0),shape_batch_request(payload,['0'],timeout_s=123),
                        full_reference_abba_batch_request(payload,['0'])):
            self.assert_managed_clocks(request)
        self.assertEqual(shape_batch_request(payload,['0'],timeout_s=123)['options']['timeout_s'],123)
        self.assertNotEqual(gateway_identity(payload,{}).digest,
                            gateway_identity(measurement.request_for(0),{}).digest)
        self.assertEqual(payload,original)
        # Profiling is a separate route, not an Eval/ABBA policy override.
        profile=GatewayMeasurement(payload,workspace=self.root,hardware='L20N',private=False,batch_size=1,profiling=True)
        self.assertEqual(profile.request_for(0),original)

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
