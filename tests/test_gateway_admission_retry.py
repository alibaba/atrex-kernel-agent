"""Bounded SDK submission recovery and explicitly permitted ambiguous Dev replay."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from aka.contracts.execution import CommandSpec, ResourceIdentity
from aka.contracts.execution_recovery import InfrastructureRetryPolicy
from aka.execution import outcomes
from aka.execution.gateway import sandbox
from aka.execution.gateway.admission_retry import POLICY_ENV
from aka.execution.gateway.errors import GatewayHTTPError


class Clock:
    def __init__(self):
        self.now = 0
        self.delays = []

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay


@pytest.fixture
def transport(monkeypatch):
    clock, facts = Clock(), []
    monkeypatch.delenv(POLICY_ENV, raising=False)
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(sandbox.time, "sleep", clock.sleep)
    monkeypatch.setattr(sandbox, "_submission_route", lambda _: ("https://example.test", "prod"))
    monkeypatch.setattr(sandbox, "emit_fact", lambda fact: facts.append(deepcopy(fact)))
    monkeypatch.setattr(outcomes, "emit_fact", lambda fact: facts.append(deepcopy(fact)))
    return clock, facts


def invoke(kind="dev", *, timeout=200, payload=None, on_submitted=None):
    return sandbox._run_sdk_job(url="https://example.test", kind=kind,
        payload=payload if payload is not None else {"command": "python3 -m scratch.probe"},
        timeout=timeout, queue_wait_grace=0, on_submitted=on_submitted)


@pytest.mark.parametrize("kind", ["dev", "eval", "profile", "compile", "disassemble"])
@pytest.mark.parametrize("http_status", [429, 502, 503, 504])
def test_retry_preserves_wire_key_and_retains_failed_attempt(transport, monkeypatch, tmp_path, kind, http_status):
    clock, facts = transport
    html = "<html><h1>503 Service Temporarily Unavailable</h1></html>"
    error = GatewayHTTPError(http_status, json.dumps({"error": {"message": html}}))
    requests = []
    terminal = {"job_id": "dv_once", "kind": kind, "status": "succeeded",
                "result": {"stdout": "probe output\n", "stderr": "", "exit_code": 0}}

    def request(url, method, path, payload, timeout):
        requests.append((url, method, path, deepcopy(payload)))
        if len(requests) == 1:
            raise error
        return {"job_id": "dv_once", "status": "queued"} if method == "POST" else terminal

    monkeypatch.setattr(sandbox, "_gateway_json", request)
    payload = {"command": "python3 -m scratch.probe", "files": {"probe.py": "print(1)"}}
    process = invoke(kind, payload=payload)
    assert [r[1] for r in requests] == ["POST", "POST", "GET"]
    assert requests[0] == requests[1]  # Including the same endpoint, body and key.
    assert requests[0][3]["idempotency_key"]
    assert "idempotency_key" not in payload
    assert clock.delays == [5]
    assert json.loads(process.stdout) == terminal
    attempts = [f for f in facts if f["event"] == "attempt"]
    assert len(attempts) == 2
    assert attempts[0]["backend_response"] == {"http_status": http_status, "error": {"message": html}}
    result = outcomes.captured_result(process, resource=ResourceIdentity("remote"), records=facts)
    assert result.transport_status == "succeeded"
    assert result.facts.disposition == "terminal"
    assert result.facts.remote_terminated is True
    assert len(result.attempts) == 2  # Failed admission remains auditable.
    assert result.attempts[0].disposition in {"unknown", "not_submitted"}
    assert result.attempts[0].idempotency_key == result.attempts[1].idempotency_key
    assert [ref.identifier for ref in result.job_references] == ["dv_once"]
    from aka.experiment.file_records import FileExperimentRecords
    records = FileExperimentRecords(tmp_path / "records", "campaign")
    with records.operation("request") as ref:
        records.register(ref, "request", {"operation": kind}, kind="command", purpose="development")
        records.prepared(ref, (CommandSpec(workspace=tmp_path, command=("probe",),
                                          timeout=60, operation=kind),))
        attempt_id = records.intent(ref, 0)
        records.observed(ref, attempt_id, result=result)
        persisted = records.get(ref)
        assert persisted.attempts[0].disposition == "terminal"
        assert len(persisted.attempts[0].result.attempts) == 2

    # A different key, wire body, or operation cannot resolve the old uncertainty.
    if http_status != 429:
        for field in ("idempotency_key", "request_identity"):
            changed = deepcopy(facts)
            next(f for f in changed if f["event"] == "submission_intent")[field] = "different"
            assert outcomes.captured_result(process, resource=ResourceIdentity("remote"),
                records=changed).facts.disposition == "unknown"

        # Even a caller-provided terminal aggregate cannot hide an unrelated
        # unresolved submission in the durable Experiment record.
        from dataclasses import replace
        mismatched = replace(result, attempts=(replace(result.attempts[0], idempotency_key="different"),
                                               result.attempts[1]))
        records.observed(ref, attempt_id, result=mismatched)
        assert records.get(ref).attempts[0].disposition == "unknown"


def test_exhaustion_returns_last_original_http_error(transport, monkeypatch):
    clock, facts = transport
    errors = [GatewayHTTPError(503, json.dumps({"error": {"message": f"proxy failure {i}"}})) for i in range(3)]
    calls = []

    def request(*args):
        calls.append(deepcopy(args))
        raise errors[len(calls) - 1]

    monkeypatch.setattr(sandbox, "_gateway_json", request)
    with pytest.raises(GatewayHTTPError) as failed:
        invoke(payload={"idempotency_key": "original", "command": "probe"})
    assert failed.value is errors[-1]
    assert len(calls) == 3
    assert clock.delays == [5, 10]
    assert all(call[3]["idempotency_key"] == "original" for call in calls)
    assert len([f for f in facts if f["event"] == "attempt"]) == 3


@pytest.mark.parametrize("body", [
    {"job_id": "dv_already_accepted"},
    {"replay_policy": "never"},
    {"error": {"details": {"retry_advice": "never"}}},
    {"error": {"details": {"execution_state": "unknown"}}},
])
def test_native_prohibition_prevents_http_retry(transport, monkeypatch, body):
    clock, _ = transport
    submit = Mock(side_effect=GatewayHTTPError(503, json.dumps(body)))
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    with pytest.raises(GatewayHTTPError):
        invoke()
    assert submit.call_count == 1
    assert clock.delays == []


@pytest.mark.parametrize("status", [None, 400, 401, 403, 404, 409, 422, 500])
def test_other_http_errors_are_not_guessed_retryable(transport, monkeypatch, status):
    submit = Mock(side_effect=GatewayHTTPError(status, '{"error":{"message":"original"}}'))
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    with pytest.raises(GatewayHTTPError):
        invoke()
    assert submit.call_count == 1


def ambiguous_job(job_id):
    return {"job_id": job_id, "kind": "dev", "status": "failed", "execution_state": "unknown",
        "replay_policy": "never", "result": None, "error": {"reason": "dev_submission_ambiguous",
        "error_class": "infra", "details": {"retry_advice": "never"}}}


def successful_job(job_id):
    return {"job_id": job_id, "status": "succeeded",
            "result": {"exit_code": 0, "stdout": "output", "stderr": ""}}


def test_ambiguous_dev_reposts_with_new_key_and_preserves_both_jobs(transport, monkeypatch, tmp_path):
    clock, facts = transport
    original = ambiguous_job("dv_original")
    terminal = successful_job("dv_replacement")
    calls = []
    responses = iter([{"job_id": "dv_original"}, original, {"job_id": "dv_replacement"}, terminal])
    def request(*args):
        calls.append(deepcopy(args))
        return next(responses)
    monkeypatch.setattr(sandbox, "_gateway_json", request)
    payload = {"idempotency_key": "original", "command": "python3 -m scratch.probe",
               "files": {"scratch/probe.py": "print(1)"}}
    process = invoke(payload=payload)
    assert [c[1] for c in calls] == ["POST", "GET", "POST", "GET"]
    assert calls[0][3]["idempotency_key"] == "original"
    assert calls[2][3]["idempotency_key"] != "original"
    assert {k: v for k, v in calls[2][3].items() if k != "idempotency_key"} == {
        k: v for k, v in payload.items() if k != "idempotency_key"}
    assert payload["idempotency_key"] == "original"
    assert json.loads(process.stdout) == terminal
    assert clock.delays == [5]
    result = outcomes.captured_result(process, resource=ResourceIdentity("remote"), records=facts)
    assert result.transport_status == "succeeded"
    assert result.facts.disposition == "terminal"
    assert result.facts.remote_terminated is None  # Never claim the original command stopped.
    assert [ref.identifier for ref in result.job_references] == ["dv_original", "dv_replacement"]
    assert [attempt.backend_response for attempt in result.attempts] == [original, terminal]
    from aka.experiment.file_records import FileExperimentRecords
    records = FileExperimentRecords(tmp_path / "records", "campaign")
    with records.operation("request") as ref:
        records.register(ref, "request", {"operation": "dev"}, kind="command", purpose="development")
        records.prepared(ref, (CommandSpec(workspace=tmp_path, command=("probe",), timeout=60, operation="dev"),))
        attempt_id = records.intent(ref, 0)
        records.observed(ref, attempt_id, result=result)
        stored = records.get(ref).attempts[0]
        assert stored.disposition == "terminal"
        assert len(stored.result.attempts) == 2


def test_ambiguous_dev_retries_are_bounded_and_return_last_native_failure(transport, monkeypatch):
    clock, facts = transport
    posts = []
    def request(url, method, path, payload, timeout):
        if method == "POST":
            posts.append(deepcopy(payload))
            return {"job_id": f"dv_{len(posts)}"}
        return ambiguous_job(f"dv_{len(posts)}")
    monkeypatch.setattr(sandbox, "_gateway_json", request)
    process = invoke()
    assert process.returncode == 1
    assert json.loads(process.stdout) == ambiguous_job("dv_3")
    assert clock.delays == [5, 10]
    assert len({p["idempotency_key"] for p in posts}) == 3
    assert len([f for f in facts if f["event"] == "attempt"]) == 3


@pytest.mark.parametrize("case", ["disabled", "wait_budget", "retry_budget", "overslept", "cancelled"])
@pytest.mark.parametrize("reason", ["dev_submission_ambiguous", "logs_unavailable"])
def test_ambiguous_dev_respects_budget_disable_and_cancellation(transport, monkeypatch, case, reason):
    clock, _ = transport
    if case in {"disabled", "retry_budget"}:
        monkeypatch.setenv(POLICY_ENV, json.dumps({"delays": [] if case == "disabled" else [5, 10],
                                                "max_elapsed_s": 120 if case == "disabled" else 4}))
    terminal = ambiguous_job("dv_original")
    terminal['error']['reason'] = reason
    submit = Mock(side_effect=[{"job_id": "dv_original"}, terminal])
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    if case == "overslept":
        monkeypatch.setattr(sandbox.time, "sleep", lambda _: setattr(clock, "now", 300))
    if case == "cancelled":
        monkeypatch.setattr(sandbox.time, "sleep", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            invoke()
    else:
        assert json.loads(invoke(timeout=4 if case == "wait_budget" else 200).stdout) == terminal
    assert [c.args[1] for c in submit.call_args_list] == ["POST", "GET"]


@pytest.mark.parametrize("kind", ["dev", "eval", "profile", "compile", "disassemble"])
@pytest.mark.parametrize("reason", ["dev_submission_ambiguous", "logs_unavailable", "another_infrastructure_error"])
def test_every_explicit_infra_failure_retries_despite_backend_prohibition(transport, monkeypatch, kind, reason):
    clock, facts = transport
    terminal = ambiguous_job("job_original")
    terminal.update(kind=kind)
    terminal["error"]["reason"] = reason
    terminal["error"]["details"].update(execution_state="unknown", failure_origin="infrastructure",
        gateway_attempts=1, gateway_max_attempts=2, gateway_retry_stop_reason="non_replayable_submission")
    success = successful_job("job_replacement")
    calls = []
    responses = iter([{"job_id": "job_original"}, terminal, {"job_id": "job_replacement"}, success])
    def request(*args):
        calls.append(deepcopy(args))
        return next(responses)
    submit = Mock(side_effect=request)
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    process = invoke(kind)
    assert json.loads(process.stdout) == success
    assert clock.delays == [5]
    assert [call.args[1] for call in submit.call_args_list] == ["POST", "GET", "POST", "GET"]
    assert calls[0][3]["idempotency_key"] != calls[2][3]["idempotency_key"]
    assert [fact["backend_response"] for fact in facts if fact["event"] == "attempt"] == [terminal, success]
    result = outcomes.captured_result(process, resource=ResourceIdentity("remote"), records=facts)
    assert result.transport_status == "succeeded"
    from aka.evaluation.tool_results import normal_evaluation_execution
    assert normal_evaluation_execution(result)


@pytest.mark.parametrize("status,error_class", [("cancelled", "infra"), ("failed", "code"),
                                                    ("failed", "validation"), ("failed", "unknown")])
def test_retry_does_not_guess_candidate_errors_or_override_cancellation(transport, monkeypatch, status, error_class):
    clock, _ = transport
    terminal = ambiguous_job("dv_original")
    terminal["status"] = status
    terminal["error"].update(reason="command_failed", error_class=error_class)
    submit = Mock(side_effect=[{"job_id": "dv_original"}, terminal])
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    assert json.loads(invoke().stdout) == terminal
    assert submit.call_count == 2
    assert clock.delays == []


@pytest.mark.parametrize("status", [400, 401, 422, 500, 503])
def test_http_infra_class_overrides_status_and_never_advice(transport, monkeypatch, status):
    clock, facts = transport
    body = {"error": {"error_class": "infra", "reason": "transient_service_failure",
                      "details": {"retry_advice": "never", "execution_state": "unknown"}}}
    calls = []
    responses = iter([GatewayHTTPError(status, json.dumps(body)), {"job_id": "job_ok"}, successful_job("job_ok")])
    def request(*args):
        calls.append(deepcopy(args))
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(sandbox, "_gateway_json", request)
    process = invoke()
    assert process.returncode == 0
    assert [call[1] for call in calls] == ["POST", "POST", "GET"]
    assert clock.delays == [5]
    assert next(f for f in facts if f["event"] == "attempt")["backend_response"] == dict(body, http_status=status)
    assert calls[0][3]["idempotency_key"] == calls[1][3]["idempotency_key"]
    assert outcomes.captured_result(process, resource=ResourceIdentity("remote"), records=facts).facts.disposition == "terminal"


def test_infra_submission_response_without_job_identity_retries(transport, monkeypatch):
    clock, _ = transport
    failure = {"error": {"error_class": "infra", "reason": "temporarily_unavailable"}}
    submit = Mock(side_effect=[failure, {"job_id": "job_ok"}, successful_job("job_ok")])
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    assert invoke().returncode == 0
    assert [call.args[1] for call in submit.call_args_list] == ["POST", "POST", "GET"]
    assert clock.delays == [5]


@pytest.mark.parametrize('path,method', [('/v1/env/L20N', 'GET'), ('/v1/env/L20N/capabilities', 'GET'),
                                      ('/v1/uploads', 'POST')])
@pytest.mark.parametrize('as_exception', [False, True])
def test_env_and_upload_requests_retry_infra(transport, monkeypatch, path, method, as_exception):
    from aka.execution.gateway import sdk
    clock, _ = transport
    failure = {'error': {'error_class': 'infra', 'reason': 'service_failure',
                         'details': {'retry_advice': 'never'}}}
    response = GatewayHTTPError(500, json.dumps(failure)) if as_exception else failure
    once = Mock(side_effect=[response, {'normal': 'response'}])
    monkeypatch.setattr(sdk, '_request_once', once)
    assert sdk.request('https://example.test', method, path, {}, 30) == {'normal': 'response'}
    assert once.call_count == 2
    assert clock.delays == [5]


def test_prepared_recovery_path_also_retries_terminal_infra(tmp_path):
    from aka.execution.gateway.jobs import GatewayJobs
    clock = Clock()
    original = ambiguous_job('ev_original')
    original['kind'] = 'eval'
    original['error']['reason'] = 'logs_unavailable'
    success = successful_job('ev_replacement')
    replies = iter([{'job_id': 'ev_original'}, original, {'job_id': 'ev_replacement'}, success])
    calls = []
    def request(*args):
        calls.append(deepcopy(args))
        return next(replies)
    jobs = GatewayJobs(endpoint='https://example.test', http=request,
                       clock=lambda: clock.now, sleep=clock.sleep, idempotent_replay=True)
    command = CommandSpec(tmp_path, ('evaluate',), timeout=200, operation='run',
        submission=json.dumps({'schema_version': 1, 'kind': 'run', 'request': {'options': {}}}).encode())
    result = jobs.replay(command, idempotency_key='original', retry=InfrastructureRetryPolicy())
    assert result.transport_status == 'succeeded'
    assert clock.delays == [5]
    assert [call[1] for call in calls] == ['POST', 'GET', 'POST', 'GET']
    assert calls[0][3]['idempotency_key'] != calls[2][3]['idempotency_key']
    assert any(attempt.backend_response == original for attempt in result.attempts)
    assert result.attempts[-1].backend_response == success


def test_http_retry_and_ambiguous_replay_share_attempt_budget(transport, monkeypatch):
    clock, _ = transport
    calls = []
    responses = iter([GatewayHTTPError(503, "busy"), {"job_id": "dv_original"},
                      ambiguous_job("dv_original"), {"job_id": "dv_replacement"},
                      successful_job("dv_replacement")])
    def request(*args):
        calls.append(deepcopy(args))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(sandbox, "_gateway_json", request)
    assert invoke().returncode == 0
    assert [c[1] for c in calls] == ["POST", "POST", "GET", "POST", "GET"]
    assert calls[0][3] == calls[1][3]
    assert calls[1][3]["idempotency_key"] != calls[3][3]["idempotency_key"]
    assert clock.delays == [5, 10]


def test_query_503_keeps_original_job(transport, monkeypatch):
    submit = Mock(side_effect=[{"job_id": "dv_original"}, GatewayHTTPError(503, "busy"),
        {"job_id": "dv_original", "status": "succeeded", "result": {"exit_code": 0}}])
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    assert invoke().returncode == 0
    assert [c.args[1] for c in submit.call_args_list] == ["POST", "GET", "GET"]
    assert all("dv_original" in c.args[2] for c in submit.call_args_list[1:])


@pytest.mark.parametrize("case", ["disabled", "wait_budget", "retry_budget", "retry_after", "overslept", "cancelled"])
def test_no_retry_beyond_budget_or_cancellation(transport, monkeypatch, case):
    clock, _ = transport
    if case in {"disabled", "retry_budget"}:
        monkeypatch.setenv(POLICY_ENV, json.dumps({"delays": [] if case == "disabled" else [5, 10],
                                                "max_elapsed_s": 120 if case == "disabled" else 4}))
    error = GatewayHTTPError(503, "original", retry_after_s=130 if case == "retry_after" else None)
    submit = Mock(side_effect=error)
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    if case == "overslept":
        monkeypatch.setattr(sandbox.time, "sleep", lambda _: setattr(clock, "now", 300))
    if case == "cancelled":
        monkeypatch.setattr(sandbox.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt if case == "cancelled" else GatewayHTTPError):
        invoke(timeout=4 if case == "wait_budget" else 200)
    assert submit.call_count == 1


def test_callback_failure_after_acceptance_does_not_retry(transport, monkeypatch):
    submit = Mock(return_value={"job_id": "dv_original"})
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    with pytest.raises(OSError, match="archive failed"):
        invoke(on_submitted=Mock(side_effect=OSError("archive failed")))
    assert submit.call_count == 1


def test_sdk_retry_after_survives_http_adapter(monkeypatch):
    from aka.execution.gateway import sdk
    from atrex_gateway_client.errors import GatewayError
    api = Mock()
    api.submit_job.side_effect = GatewayError(429, {"reason": "gateway_overloaded"}, retry_after_s=17)
    monkeypatch.setattr(sdk, "client", lambda **_: api)
    with pytest.raises(GatewayHTTPError) as failure:
        sdk.request("https://example.test", "POST", "/v1/jobs/dev", {}, 30)
    assert failure.value.retry_after_s == 17


def test_retry_after_delays_same_key_request(transport, monkeypatch):
    clock, _ = transport
    submit = Mock(side_effect=[GatewayHTTPError(429, "overloaded", retry_after_s=17),
        {"job_id": "dv_original"}, {"job_id": "dv_original", "status": "succeeded", "result": {"exit_code": 0}}])
    monkeypatch.setattr(sandbox, "_gateway_json", submit)
    assert invoke().returncode == 0
    assert clock.delays == [17]


def test_experiment_policy_reaches_launcher(tmp_path):
    from aka.experiment.factory import ExperimentFactory
    from aka.contracts.execution import BackendCapabilities
    from types import SimpleNamespace
    executor = SimpleNamespace(capabilities=BackendCapabilities(operations=frozenset({"dev"})),
                               resource_identity=None, run=Mock())
    policy = InfrastructureRetryPolicy((2, 7), 90)
    factory = ExperimentFactory(command_evaluator=Mock(), retry_policy=policy)
    with factory.open(workspace=tmp_path, purpose="development", executor=executor) as experiment:
        assert experiment.submission_retry_policy == policy
        command = CommandSpec(workspace=tmp_path, command=("probe",), timeout=60, operation="dev")
        prepared = experiment._observed_command(command, None, None, None)
        assert json.loads(prepared.environment[POLICY_ENV]) == {"delays": [2, 7], "max_elapsed_s": 90}
        assert POLICY_ENV not in command.environment
