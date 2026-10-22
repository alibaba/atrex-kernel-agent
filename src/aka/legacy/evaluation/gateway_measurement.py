"""Measurement policy for the retained typed/dev CLI adapter.

Execution supplies raw typed receipts in planned batch order. This adapter owns
shape slicing, scoring, privacy and measurement records, and never submits jobs.
The legacy CLI input inference remains a separate compatibility boundary.
"""
from pathlib import Path
from typing import Any

from .transport_results import (
    _request_shape_ids, _shape_batches, _shape_batch_request, _shape_batch_reference,
    _optimizer_result_from_eval, _merge_optimizer_results, _record_episode_evaluation,
    _record_result_lines, _with_workspace_performance_score, _hydrate_result_lines,
    _public_numerical_result_line, _hydrate_abba_result_lines,
    TEST_RESULT_PREFIX, ABBA_RESULT_PREFIX, NUMERICAL_RESULT_PREFIX,
)


class GatewayMeasurement:
    def __init__(self, request, *, workspace, hardware, private, batch_size, profiling=False):
        self.request = request
        self.workspace = workspace
        self.hardware = hardware
        self.private = private
        self.shape_ids = _request_shape_ids(request)
        self.batches = [self.shape_ids] if profiling else _shape_batches(self.shape_ids, batch_size)

    def request_for(self, index):
        return (_shape_batch_request(self.request, self.batches[index])
                if len(self.batches) > 1 else self.request)

    def write_reference(self, index, directory):
        _shape_batch_reference(self.request_for(index)["reference"], directory)

    def interpret(self, jobs):
        if len(jobs) != len(self.batches):
            raise ValueError("measurement requires one receipt per planned batch")
        metadata = self.request["reference"].get("metadata")
        performance = self.request.get("mode") != "correctness_only"
        results = [_optimizer_result_from_eval(job["result"], shapes, metadata,
            require_performance=performance, hardware=self.hardware)
            for job, shapes in zip(jobs, self.batches)]
        result = (_merge_optimizer_results(results, self.shape_ids, metadata,
            require_performance=performance, hardware=self.hardware)
            if len(results) > 1 else results[0])
        result = _mask_generalized_result(self.workspace, result, private=self.private)
        _record_episode_evaluation(self.workspace, result, gateway_kind="run",
            job_id=",".join(str(job.get("job_id")) for job in jobs))
        return result


def prepare_dev_output(workspace, stdout, *, evaluator, private):
    if evaluator:
        stdout = _hydrate_abba_result_lines(workspace, stdout)
        stdout = _hydrate_result_lines(workspace, stdout)
        _record_result_lines(workspace, stdout, gateway_kind="dev")
    if private:
        stdout = "\n".join(
            _public_numerical_result_line(line) if line.startswith(NUMERICAL_RESULT_PREFIX) else line
            for line in stdout.splitlines()
            if line.startswith((TEST_RESULT_PREFIX, ABBA_RESULT_PREFIX, NUMERICAL_RESULT_PREFIX)))
    return stdout


def _mask_generalized_result(
    workspace: Path, result: dict[str, Any], *, private: bool
) -> dict[str, Any]:
    """Hide exact inputs and failures but retain real latency keyed by opaque shape id."""
    result = _with_workspace_performance_score(workspace, result)
    if not private:
        return result
    masked = dict(result)
    if result.get("failures"):
        evaluation_failure = result.get("evaluation_failure")
        if isinstance(evaluation_failure, dict):
            masked["failures"] = [
                str(evaluation_failure.get("message") or "evaluation unavailable")
            ]
        elif result.get("actionable_diagnostics"):
            masked["failures"] = [
                "candidate compile/import/runtime failure; see actionable_diagnostics"
            ]
        else:
            masked["failures"] = [
                "one or more hidden evaluator cases failed; reproduce within the public shape_domain"
            ]
    masked["hidden_case_details"] = "shape inputs and failure details withheld"
    return masked



def with_measurement_conditions(stdout, *, gateway_warmup_iters):
    """Attach the selected warmup condition to each public measurement receipt."""
    import json
    lines = stdout.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(TEST_RESULT_PREFIX):
            payload = json.loads(line[len(TEST_RESULT_PREFIX):])
            if isinstance(payload, dict):
                payload.setdefault("measurement_conditions", {})["gateway_warmup_iters"] = gateway_warmup_iters
                lines[index] = TEST_RESULT_PREFIX + json.dumps(payload) + "\n"
    return "".join(lines)
