"""Native Shape-selected batches must reach the public Evaluate receipt."""
import json
import math
from dataclasses import replace

import pytest

from aka.contracts.execution import CommandSpec, ExecutionFacts, JobResult
from aka.evaluation.gateway_command import GatewayCommandEvaluator
from aka.evaluation.tool_results import canonical_result, with_execution_error


def prepare(tmp_path, *, shape_count=30, selected=None, mode="full", repetitions=1):
    request = {
        "candidate": "class Model: pass\n",
        "reference": {
            "reference_py": "class Model: pass\n",
            "input_py": "def get_inputs(): return []\n",
            "shapes": {str(i): {"N": i + 1} for i in range(shape_count)},
            "metadata": {"performance_objective": "shape_latency_geometric_mean"},
        },
        "mode": mode,
        "options": {"shape_ids": selected} if selected is not None else {},
    }
    command = CommandSpec(tmp_path, (), hardware="L20D", operation="run",
        submission=json.dumps({"schema_version": 1, "kind": "run", "request": request}).encode())
    evaluator = GatewayCommandEvaluator(shape_batch_size=10, measurement_repetitions=repetitions)
    return evaluator, evaluator.prepare(command)


def results_for(prepared):
    results = []
    for index, step in enumerate(prepared.steps):
        request = json.loads(step.submission)["request"]
        selected = request["options"]["shape_ids"]
        raw = {
            "error": None,
            "passed": {stage: {sid: {"status": "passed"} for sid in selected}
                       for stage in ("compile", "correctness", "performance")},
            "performance": {"shapes": {sid: {"samples": [
                {"end_to_end_time_ms": (int(sid) + 1) / 1000}]} for sid in selected}},
            "shape_selection": {"partial": len(selected) < len(request["reference"]["shapes"]),
                "selected_shape_ids": selected, "all_shape_ids": list(request["reference"]["shapes"])},
        }
        if request["mode"] == "correctness_only":
            raw.pop("performance")
            raw["passed"].pop("performance")
        results.append(JobResult(0, stdout=json.dumps({"job_id": f"ev_batch{index}",
            "status": "succeeded", "result": raw}), transport_status="succeeded",
            facts=ExecutionFacts(disposition="terminal")))
    return results


def receipt(interpreted):
    result = interpreted.result
    response = with_execution_error({"exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}, result, operation="evaluate")
    return canonical_result(response, {"operation": "evaluate"})


@pytest.mark.parametrize("shape_count,repetitions", [(30, 1), (90, 1), (30, 3)])
def test_full_evaluate_merges_successful_native_batches(tmp_path, shape_count, repetitions):
    evaluator, prepared = prepare(tmp_path, shape_count=shape_count, repetitions=repetitions)
    assert len(prepared.steps) == shape_count // 10 * repetitions
    for step in prepared.steps:
        request = json.loads(step.submission)["request"]
        assert len(request["reference"]["shapes"]) == shape_count
        assert len(request["options"]["shape_ids"]) == 10
    raw = results_for(prepared)
    original = [result.stdout for result in raw]
    interpreted = evaluator.interpret(prepared, raw)
    agent = receipt(interpreted)
    assert interpreted.verification.gate == "PASS"
    assert interpreted.result.returncode == 0
    assert agent["status"] == "completed"
    assert agent["result"]["correct"] is True
    assert agent["result"]["latency_us_by_shape"] == {str(i): i + 1 for i in range(shape_count)}
    assert agent["result"]["latency_us_geomean"] == pytest.approx(
        math.exp(sum(math.log(i + 1) for i in range(shape_count)) / shape_count))
    assert "error" not in agent["result"]
    assert [result.stdout for result in raw] == original


def test_noncontiguous_subset_is_not_required_to_cover_the_reference(tmp_path):
    selected = [str(i) for i in range(1, 30, 2)]
    evaluator, prepared = prepare(tmp_path, selected=selected)
    assert len(prepared.steps) == 2
    interpreted = evaluator.interpret(prepared, results_for(prepared))
    assert interpreted.verification.gate == "PASS"
    assert set(receipt(interpreted)["result"]["latency_us_by_shape"]) == set(selected)


def test_correctness_only_batches_do_not_require_performance(tmp_path):
    evaluator, prepared = prepare(tmp_path, mode="correctness_only")
    interpreted = evaluator.interpret(prepared, results_for(prepared))
    assert interpreted.verification.gate == "PASS"
    assert receipt(interpreted)["result"]["correct"] is True


@pytest.mark.parametrize("stage", ["compile", "correctness", "performance"])
def test_missing_selected_shape_is_an_explicit_incomplete_result(tmp_path, stage):
    evaluator, prepared = prepare(tmp_path)
    results = results_for(prepared)
    job = json.loads(results[1].stdout)
    if stage == "performance":
        job["result"]["performance"]["shapes"].pop("17")
    else:
        job["result"]["passed"][stage].pop("17")
    results[1] = replace(results[1], stdout=json.dumps(job))
    interpreted = evaluator.interpret(prepared, results)
    agent = receipt(interpreted)
    assert interpreted.verification.gate == "ERROR"
    assert interpreted.verification.failure.category == "incomplete"
    assert agent["result"]["correct"] is None
    assert agent["result"]["incomplete_shape_ids"] == ["17"]
    assert agent["result"]["evaluation_failure"]["code"] == "evaluation-evidence-incomplete"
    assert "17" in agent["result"]["evaluation_failure"]["message"]
    assert "execution_failed" not in json.dumps(agent)
    assert not agent["result"]["latency_us_by_shape"]


@pytest.mark.parametrize("stage", ["compile", "correctness", "performance"])
def test_actual_candidate_rejection_remains_a_completed_failure(tmp_path, stage):
    evaluator, prepared = prepare(tmp_path)
    results = results_for(prepared)
    job = json.loads(results[1].stdout)
    job["result"]["passed"][stage]["17"] = {"status": "failed", "reason": "candidate rejected"}
    results[1] = replace(results[1], stdout=json.dumps(job))
    interpreted = evaluator.interpret(prepared, results)
    agent = receipt(interpreted)
    assert interpreted.verification.gate == "FAIL"
    assert agent["status"] == "completed"
    assert agent["result"]["correct"] is False
    assert any("17" in value for value in agent["result"]["failures"])
    assert "evaluation_failure" not in agent["result"]


@pytest.mark.parametrize("shape_count", [10, 30])
@pytest.mark.parametrize("error_source", ["verdict", "performance", "both"])
def test_performance_timeout_preserves_upstream_error_and_other_shapes(tmp_path, shape_count, error_source):
    evaluator, prepared = prepare(tmp_path, shape_count=shape_count)
    results = results_for(prepared)
    job = json.loads(results[0].stdout)
    failed_shapes = {"1", "2", "7"}
    message = "candidate exceeded 300.0s wall-clock timeout"
    for sid in failed_shapes:
        job["result"]["passed"]["performance"][sid] = {"status": "failed"}
        timed = job["result"]["performance"]["shapes"][sid]
        timed["samples"] = []
        if error_source in {"verdict", "both"}:
            job["result"]["passed"]["performance"][sid]["reason"] = message
        if error_source in {"performance", "both"}:
            timed["error"] = message
    results[0] = replace(results[0], stdout=json.dumps(job))
    original = [result.stdout for result in results]
    interpreted = evaluator.interpret(prepared, results)
    agent = receipt(interpreted)
    assert interpreted.verification.gate == "FAIL"
    assert agent["status"] == "completed"
    assert "evaluation_failure" not in agent["result"]
    assert "incomplete_shape_ids" not in agent["result"]
    assert agent["result"]["latency_us_by_shape"] == {
        str(i): i + 1 for i in range(shape_count) if str(i) not in failed_shapes}
    assert agent["result"]["latency_us_geomean"] is None
    assert agent["result"]["latency_us_arith_mean"] is None
    diagnostics = agent["result"]["diagnostics"]
    assert {sid for row in diagnostics for sid in row.get("shape_ids", [row["shape_id"]])} == failed_shapes
    assert all(row["stage"] == "performance" and row["error"] == message for row in diagnostics)
    assert [result.stdout for result in results] == original


def test_failed_performance_reaches_workflow_receipt_without_unknown(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from aka.workflow.light.measurements import SharedMeasurements

    evaluator, prepared = prepare(tmp_path, shape_count=10)
    identity = replace(prepared.measurement_identities[0], dependency_policy="explicit-requirements")
    prepared = replace(prepared, measurement_identities=(identity,))
    job = json.loads(results_for(prepared)[0].stdout)
    message = "candidate exceeded 300.0s wall-clock timeout"
    job["result"]["passed"]["performance"]["7"] = {"status": "failed", "reason": message}
    job["result"]["performance"]["shapes"]["7"] = {"samples": [], "error": message}
    result = replace(results_for(prepared)[0], stdout=json.dumps(job), environment_identity="same-worker")
    interpreted = evaluator.interpret(prepared, [result])
    assessment = evaluator.diagnose_step(prepared, 0, result)
    record = SimpleNamespace(
        ref=SimpleNamespace(experiment_id="test", as_dict=lambda: {}),
        steps=[SimpleNamespace(task_id="step", operation="run", measurement_identity=identity)],
        attempts=[SimpleNamespace(task_id="step", disposition="terminal", result=result, assessment=assessment)],
        evaluation=interpreted.verification, lifecycle="terminal", phase="complete", kind="command", created_at=None)
    owner = SharedMeasurements.__new__(SharedMeasurements)
    # Source lookup is independent of result interpretation; retain a valid binding.
    owner._source = Mock(return_value=object())
    owner.kernel = Mock(return_value={})
    experiment = SimpleNamespace(command_result=lambda ref: interpreted.result)
    agent = owner._response(experiment, record)["agent_result"]
    assert agent["status"] == "completed"
    assert "qualification" not in agent["result"]
    assert agent["result"]["diagnostics"] == [{"stage": "performance", "shape_id": "7", "error": message}]
    assert len(agent["result"]["latency_us_by_shape"]) == 9


def test_native_abba_does_not_accept_failed_performance_with_samples(tmp_path):
    from aka.evaluation.native_abba import summarize

    _, prepared = prepare(tmp_path, shape_count=10)
    raw = json.loads(results_for(prepared)[0].stdout)["result"]
    raw["correctness"] = {"shapes": {str(i): {} for i in range(10)}}
    raw["passed"]["performance"]["7"] = {"status": "failed", "reason": "benchmark failed"}
    result = summarize(raw, [str(i) for i in range(10)])
    assert result["all_pass"] is False
    assert "7" not in result["latency_us_by_shape"]
    assert result["diagnostics"] == [{"stage": "performance", "shape_id": "7", "error": "benchmark failed"}]


def output_check(*, passed=True, name="output"):
    return {
        "allowed_mismatched_elements": 0,
        "error": None if passed else {"reason": "mismatch", "message": "output differs"},
        "max_elementwise_abs_diff": 0.000091552734375,
        "max_elementwise_rel_diff": 0.0040471553802490234,
        "mismatched_elements": 0 if passed else 7,
        "name": name,
        "passed": passed,
        "relative_l2": 0.0004981350234806307,
        "tolerance": {"atol": 0.01, "max_mismatch_fraction": 0, "max_rel_l2": None,
                      "mode": "elementwise", "relative_to": "candidate", "rtol": 0.05},
        "total_elements": 16777216,
    }


@pytest.mark.parametrize("mode", ["full", "correctness_only"])
@pytest.mark.parametrize("failed", [False, True])
def test_per_shape_checks_survive_batch_merge_and_agent_projection(tmp_path, mode, failed):
    from aka.workflow.light.host.projection import evaluation
    from aka.evaluation.tool_results import compact_receipt

    evaluator, prepared = prepare(tmp_path, mode=mode)
    results = results_for(prepared)
    expected = {}
    for i, response in enumerate(results):
        job = json.loads(response.stdout)
        shapes = {}
        for sid in json.loads(prepared.steps[i].submission)["request"]["options"]["shape_ids"]:
            passed = not (failed and sid == "17")
            checks = [output_check(passed=passed), output_check(name="final_state")]
            cases = [{"passed": passed, "error": None, "outputs": checks,
                      "mutated_inputs": [], "unexpected_mutations": [],
                      "seed": 98765, "inputs": ["PRIVATE_INPUT"], "distribution": "PRIVATE_GENERATOR"}
                     for _ in range(5)]
            shapes[sid] = {"cases": cases}
            job["result"]["passed"]["correctness"][sid] = {"status": "passed" if passed else "failed"}
            expected[sid] = checks
        job["result"]["correctness"] = {"shapes": shapes}
        results[i] = replace(response, stdout=json.dumps(job))
    original = [r.stdout for r in results]
    agent = compact_receipt(receipt(evaluator.interpret(prepared, results)))
    projected = evaluation(agent["result"])
    for value in (agent["result"], projected):
        details = value["correctness_by_shape"]
        assert set(details) == set(expected)
        for sid, checks in expected.items():
            assert details[sid]["status"] == ("failed" if failed and sid == "17" else "passed")
            assert len(details[sid]["cases"]) == 5
            assert all(case["outputs"] == checks for case in details[sid]["cases"])
        assert "PRIVATE_INPUT" not in json.dumps(value)
        assert "PRIVATE_GENERATOR" not in json.dumps(value)
        assert "98765" not in json.dumps(value)
    assert agent["result"]["correct"] is not failed
    assert [r.stdout for r in results] == original


def test_repeated_measurements_keep_earlier_failing_checks(tmp_path):
    from aka.evaluation.feedback import project_feedback

    evaluator, prepared = prepare(tmp_path, shape_count=10, repetitions=3)
    results = results_for(prepared)
    for i, response in enumerate(results):
        job = json.loads(response.stdout)
        job["result"]["correctness"] = {"shapes": {
            str(sid): {"cases": [{"outputs": [output_check(passed=not (i == 0 and sid == 0))]}]}
            for sid in range(10)
        }}
        if i == 0:
            job["result"]["passed"]["correctness"]["0"] = {"status": "failed"}
        results[i] = replace(response, stdout=json.dumps(job))
    agent = receipt(evaluator.interpret(prepared, results))
    assert agent["result"]["correct"] is False
    details = agent["result"]["correctness_by_shape"]
    assert len(details["0"]["runs"]) == 3
    assert [run["status"] for run in details["0"]["runs"]] == ["failed", "passed", "passed"]
    assert details["0"]["runs"][0]["cases"][0]["outputs"] == [output_check(passed=False)]
    assert project_feedback(agent["result"])["correctness_by_shape"] == details


def test_correctness_feedback_filters_shape_selection_and_retains_native_errors():
    from aka.evaluation.feedback import evaluation_feedback, project_feedback

    error = {"reason": "candidate_exception", "details": {"traceback": "original traceback"}}
    raw = {"passed": {"correctness": {"2": {"status": "failed", "reason": "candidate exception"}}},
           "correctness": {"shapes": {"2": {"error": error, "cases": []},
                                      "99": {"cases": [{"outputs": [output_check()]}]}}}}
    value = evaluation_feedback(raw, ["2"])
    assert value["correctness_by_shape"] == {
        "2": {"status": "failed", "reason": "candidate exception", "error": error, "cases": []}}
    assert project_feedback(value)["correctness_by_shape"] == value["correctness_by_shape"]
    value["correctness_by_shape"]["2"]["error"]["details"]["traceback"] = "changed"
    assert raw["correctness"]["shapes"]["2"]["error"]["details"]["traceback"] == "original traceback"
