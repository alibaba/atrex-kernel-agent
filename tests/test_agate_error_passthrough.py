"""Agate owns its errors; Light only interprets them for internal retry policy."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aka.contracts.execution import ExecutionAttempt, ExecutionDiagnostic, ExecutionFacts, JobResult
from aka.contracts.execution_recovery import JobReference
from aka.evaluation.response_policy import execution_error, upstream_error
from aka.evaluation.tool_results import canonical_result, with_execution_error
from aka.execution.gateway.errors import GatewayHTTPError
from aka.execution.gateway.jobs import causes_from_error, result_from_job


OPERATIONS = ["evaluate", "same_allocation_abba", "command", "dev", "profile", "check", "disassemble", "env"]
INFRA_ERROR = {
    "reason": "dev_submission_ambiguous", "error_class": "infra",
    "message": "the dev command may have started remotely; it was not submitted again",
    "trace_id": "req-original", "details": {"execution_state": "unknown",
        "failure_origin": "infrastructure", "gateway_attempts": 1,
        "gateway_max_attempts": 2, "gateway_retry_stop_reason": "non_replayable_submission",
        "retry_advice": "never", "user_action": "Keep the original job_id and idempotency key."},
    "future_field": {"values": [None, False, 0, "unaltered"]},
}


def public(receipt, operation, stdout=None):
    response = with_execution_error({"exit_code": receipt.returncode,
        "stdout": receipt.stdout if stdout is None else stdout, "stderr": receipt.stderr},
        receipt, operation=operation)
    return canonical_result(response, {"operation": operation}, record_id="gateway-test")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("status,error", [
    (400, {"reason": "invalid_source", "error_class": "validation", "details": {
        "violations": [{"file": "kernel.py", "message": "Blocked import: external_kernel"}]}}),
    (422, {"reason": "request_validation_failed", "error_class": "validation", "details": {
        "validation_errors": [{"loc": ["options", "shape_ids"],
                               "msg": "Value error, abba uses every reference shape"}]}}),
    (503, INFRA_ERROR),
    (502, "upstream connection reset"),
    (422, [{"loc": ["spec"], "detail": "invalid spec"}]),
])
def test_http_errors_keep_exact_value_and_http_status(operation, status, error):
    original = GatewayHTTPError(status, json.dumps({"error": error}))
    receipt = JobResult(2, transport_status="failed", facts=ExecutionFacts("not_submitted"),
        causes=causes_from_error(original))
    result = public(receipt, operation)
    assert result["status"] == "failed"
    assert result["result"]["error"] == error
    assert result["result"]["http_status"] == status
    assert "next_action" not in result["result"]
    assert result["gateway_record_id"] == "gateway-test"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_terminal_infra_error_overrides_incomplete_evidence(operation):
    job = {"job_id": "dv_original", "status": "failed", "result": None, "error": INFRA_ERROR}
    receipt = result_from_job(job, JobReference("gateway", "dv_original"), operation=operation)
    # This prefix formerly suppressed errors for Check and ABBA.
    stdout = '[sandbox] ABBA_JSON={"correct":null,"error":"incomplete evidence"}\n'
    result = public(receipt, operation, stdout=stdout)
    assert result["status"] == "failed"
    assert result["result"]["error"] == INFRA_ERROR
    assert "next_action" not in result["result"]
    if operation not in {"command", "dev"}:
        assert result["result"] == {"error": INFRA_ERROR}


def test_cause_archive_roundtrip_preserves_error():
    from aka.execution.outcomes import _cause, _cause_record
    cause = causes_from_error(GatewayHTTPError(503, json.dumps({"error": INFRA_ERROR})))[0]
    restored = _cause(json.loads(json.dumps(_cause_record(cause))))
    result = public(JobResult(75, causes=(restored,)), "evaluate")
    assert result["result"] == {"error": INFRA_ERROR, "http_status": 503}


def test_upload_reservation_keeps_original_http_error():
    from aka.execution.gateway.source_archives import SourceArchiveError
    wrapped = SourceArchiveError("reservation-unavailable", status=503,
        upstream_error=GatewayHTTPError(503, json.dumps({"error": INFRA_ERROR})))
    receipt = JobResult(75, attempts=(ExecutionAttempt("source-upload",
        "not_submitted", causes=causes_from_error(wrapped)),))
    assert public(receipt, "evaluate")["result"] == {"error": INFRA_ERROR, "http_status": 503}


def test_non_json_http_body_is_not_replaced():
    receipt = JobResult(2, causes=causes_from_error(GatewayHTTPError(502, "<html>Bad gateway</html>")))
    assert public(receipt, "evaluate")["result"] == {"error": "<html>Bad gateway</html>", "http_status": 502}


def test_later_success_does_not_replay_previous_error():
    failed = result_from_job({"job_id": "ev_old", "status": "failed", "error": INFRA_ERROR},
        JobReference("gateway", "ev_old"))
    succeeded = result_from_job({"job_id": "ev_new", "status": "succeeded", "error": None,
        "result": {"exit_code": 0}}, JobReference("gateway", "ev_new"))
    receipt = replace(succeeded, attempts=failed.attempts + succeeded.attempts, causes=failed.causes)
    assert upstream_error(receipt) is None
    assert execution_error(receipt) is None
    agent = public(receipt, "evaluate", stdout='[test_kernel] RESULT_JSON={"all_pass":true}\n')
    assert agent["status"] == "completed"
    assert "error" not in agent["result"]


def test_dev_stdout_is_never_interpreted_as_agate_error():
    stdout = json.dumps({"http_status": 503, "error": "probe's own test data"})
    receipt = JobResult(0, stdout=stdout, output_scope="command", exit_code=0)
    result = public(receipt, "command")
    assert result["result"]["stdout"] == stdout
    assert "error" not in result["result"]


def test_light_timeout_still_has_a_light_error():
    receipt = JobResult(75, transport_status="timed_out",
        diagnostics=(ExecutionDiagnostic("infrastructure", "original-job-wait-incomplete"),))
    assert execution_error(receipt)["error"]["code"] == "gateway_wait_timeout"


@pytest.mark.parametrize("operation", ["run", "same_allocation_abba"])
def test_workflow_does_not_replace_upstream_error_with_unknown(operation):
    from aka.workflow.light.measurements import SharedMeasurements
    owner = SharedMeasurements.__new__(SharedMeasurements)
    owner._evidence = Mock(return_value={"result": {"correct": None, "qualification": {
        "status": "unknown", "reasons": ["measurement-incomplete"]}}, "gateway_kind": operation,
        "gateway_record_id": "gateway-test", "execution_status": "unknown", "exit_code": 75})
    receipt = result_from_job({"job_id": "ev_failed", "status": "failed", "error": INFRA_ERROR},
        JobReference("gateway", "ev_failed"))
    record = SimpleNamespace(kind="command", phase="complete", lifecycle="terminal", evaluation=None,
        ref=object(), attempts=[])
    agent = owner._response(SimpleNamespace(command_result=lambda ref: receipt), record)["agent_result"]
    assert agent["status"] == "failed"
    assert agent["result"] == {"error": INFRA_ERROR}


def test_failed_batch_error_is_not_replaced_by_successful_last_batch(tmp_path):
    from test_evaluate_batch_coverage import prepare, results_for
    evaluator, prepared = prepare(tmp_path)
    results = results_for(prepared)
    results[0] = result_from_job({"job_id": "ev_failed", "status": "failed", "error": INFRA_ERROR},
        JobReference("gateway", "ev_failed"))
    interpreted = evaluator.interpret(prepared, results)
    assert public(interpreted.result, "evaluate")["result"] == {"error": INFRA_ERROR}


def test_sdk_preserves_unfamiliar_error_payload():
    from aka.execution.gateway.sdk import _sdk_error_process
    error = SimpleNamespace(payload=[{"detail": "validation error", "new": True}], status=422)
    response = _sdk_error_process(error, stage="request")
    assert json.loads(response.stdout) == {"status": "failed", "error": error.payload, "http_status": 422}
    assert response.stderr == ""


@pytest.mark.parametrize("reason", ["invalid_source", "request_validation_failed", "logs_unavailable",
    "dev_submission_ambiguous", "timeout", "gateway_collection_timeout", "agate_clock_configuration_invalid"])
def test_host_projection_does_not_add_recovery_instructions(reason):
    from aka.workflow.light.host.projection import gateway_error_result, gateway_http_error_result
    error = {**INFRA_ERROR, "reason": reason}
    envelope = {"status": "failed", "error": error, "future_field": [1, None]}
    assert gateway_error_result(envelope) == envelope
    assert gateway_http_error_result(503, json.dumps(envelope), "evaluate") == {
        **envelope, "operation": "evaluate", "http_status": 503}


@pytest.mark.parametrize("status", [401, 503])
def test_job_query_http_error_survives_permanent_failure_or_budget_expiry(monkeypatch, status):
    from aka.execution.gateway import sandbox
    from aka.execution import outcomes
    clock = [0]
    facts = []
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sandbox.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(sandbox, "emit_fact", facts.append)
    monkeypatch.setattr(outcomes, "emit_fact", facts.append)
    error = {"reason": "authentication_failed" if status == 401 else "service_unavailable",
             "error_class": "authentication" if status == 401 else "infra", "details": {"upstream": "original"}}
    monkeypatch.setattr(sandbox, "_gateway_json", Mock(side_effect=GatewayHTTPError(status, json.dumps({"error": error}))))
    process = sandbox._wait_sdk_job(url="https://example.test", kind="eval", job_id="ev_pending", deadline=3, started=0)
    from aka.contracts.execution import ResourceIdentity
    receipt = outcomes.captured_result(process, resource=ResourceIdentity("gateway"), records=facts, decode_output=False)
    result = public(receipt, "evaluate")
    assert result["result"] == {"error": error, "http_status": status}
    assert result["status"] == "failed"
    assert json.loads(process.stdout)["status"] == "unknown"  # No claim about the remote job.


def test_env_retains_original_errors_in_discovery_snapshot(monkeypatch):
    from aka.contracts.execution_environment import EnvironmentQuery
    from aka.execution.environment_query import BackendEnvironment
    from aka.execution.gateway import sandbox, sdk
    monkeypatch.setattr(sdk, "route", lambda *args: "https://example.test")
    monkeypatch.setattr(sandbox, "_gateway_json", Mock(side_effect=GatewayHTTPError(503, json.dumps({"error": INFRA_ERROR}))))
    backend = BackendEnvironment(backend="gateway", operations=(), environment={}, url="https://example.test")
    snapshot = backend.query(EnvironmentQuery(hardware="L20D", fresh=True)).as_dict()
    assert snapshot["raw"] == {name: {"error": INFRA_ERROR, "http_status": 503}
                              for name in ("environment", "capabilities")}


def test_submission_error_without_job_id_is_not_rewritten(monkeypatch):
    from aka.execution.gateway import sandbox
    from aka.execution import outcomes
    from aka.contracts.execution import ResourceIdentity
    error = {"reason": "invalid_source", "error_class": "validation", "future_field": {"a": 1}}
    facts = []
    monkeypatch.setattr(outcomes, "emit_fact", facts.append)
    monkeypatch.setattr(sandbox, "emit_fact", facts.append)
    monkeypatch.setattr(sandbox, "_submission_route", lambda _: ("https://example.test", "prod"))
    request = Mock(return_value={"error": error})
    monkeypatch.setattr(sandbox, "_gateway_json", request)
    process = sandbox._run_sdk_job(url="https://example.test", kind="eval", payload={}, timeout=60, queue_wait_grace=0)
    assert request.call_count == 1
    receipt = outcomes.captured_result(process, resource=ResourceIdentity("gateway"), records=facts, decode_output=False)
    assert public(receipt, "evaluate")["result"] == {"error": error}
