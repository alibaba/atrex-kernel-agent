"""Regress the Session/Review crash and the misleading Command failure receipt."""
from dataclasses import asdict
import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest

from aka.agent.adapter import ClaudeAdapter
from aka.agent.model import TokenUsage
from aka.agent.session_capture import _LAST_CAPTURE, captured_observation, SessionCapture
from aka.contracts.execution import ResourceIdentity
from aka.evaluation.tool_results import canonical_result, with_execution_error
from aka.execution.outcomes import captured_result, observe_attempt
from aka.experiment.file_records import _job


@pytest.mark.parametrize("purpose", ["main", "claim-review"])
def test_completed_captured_session_returns_usage_instead_of_crashing(tmp_path, purpose):
    stdout = json.dumps({"type": "result", "usage": {"input_tokens": 20, "output_tokens": 7}}) + "\n"
    capture = SessionCapture(tmp_path / purpose, backend="claude", command=["claude", "test"],
                             context={}, native_environment={"HOME": str(tmp_path)})
    capture._chunks["stdout"].append(stdout)
    capture._retained_chunks["stdout"].append(stdout)
    capture.finish(exit_status=0)
    adapter = ClaudeAdapter()
    events, _ = adapter.normalize_stream(stdout)
    try:
        result = captured_observation(stdout, events, adapter.capabilities)
        assert result.terminal_usage.total_tokens == 27
        assert result.capture_complete
        assert json.loads((capture.root / "token-usage.json").read_text())["state"] == "completed"
        assert json.loads((capture.root / "conversation.jsonl").read_text().splitlines()[-1])["exit_status"] == 0
    finally:
        _LAST_CAPTURE.set(None)


def test_settled_response_usage_replaces_provisional_once():
    first = {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 3, "output_tokens": 1}}}
    other = {"type": "assistant", "message": {"id": "m2", "usage": {"input_tokens": 2, "output_tokens": 2}}}
    stdout = "\n".join(json.dumps(x) for x in (first, first, other))
    settled = TokenUsage(input_tokens=3, output_tokens=11, cache_read_tokens=0, cache_write_tokens=0, total_tokens=14, measurement="exact")
    report = {"backend": "claude", "total": asdict(TokenUsage(input_tokens=5, output_tokens=13, cache_read_tokens=0, cache_write_tokens=0, total_tokens=18, measurement="exact")),
        "responses": [{"agent": "main", "message_id": "m1", "path": "provider/claude-session.raw-jsonl", "usage": asdict(settled)}],
        "warnings": [], "capture_complete": True}
    token = _LAST_CAPTURE.set((hashlib.sha256(stdout.encode()).hexdigest(), report))
    try:
        adapter = ClaudeAdapter()
        events, _ = adapter.normalize_stream(stdout)
        result = captured_observation(stdout, events, adapter.capabilities)
        assert [e.usage.total_tokens for e in result.events if e.kind == "usage_delta"] == [14, 4]
        assert result.terminal_usage.total_tokens == 18
    finally:
        _LAST_CAPTURE.reset(token)


def command_receipt(job, stdout, stderr):
    facts = []
    with patch("aka.execution.outcomes.emit_fact", side_effect=facts.append):
        observe_attempt(subprocess.CompletedProcess([], 0, json.dumps(job), ""), operation="dev")
    receipt = captured_result(subprocess.CompletedProcess([], job["result"]["exit_code"], stdout, stderr),
        resource=ResourceIdentity("gateway"), records=facts, decode_output=False)
    # Exercise the same serialization used by the central execution archive.
    return _job(json.loads(json.dumps(receipt.as_dict())))


@pytest.mark.parametrize("operation", ["dev", "command"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_command_keeps_complete_streams_without_synthetic_error(operation, exit_code):
    stdout = "x" * 50000 + '\n[sandbox] GATEWAY_ERROR_JSON={"error":"this is script output"}\n\n'
    stderr = "Traceback (most recent call last):\nAssertionError: cuda err 201\n" if exit_code else ""
    job = {"job_id": "dv_probe", "status": "succeeded", "result": {
        "exit_code": exit_code, "stdout": stdout, "stderr": stderr}}
    receipt = command_receipt(job, stdout, stderr)
    assert receipt.as_dict()["attempts"][0]["backend_response"] == job
    response = with_execution_error({"exit_code": exit_code, "stdout": stdout, "stderr": stderr}, receipt, operation=operation)
    public = canonical_result(response, {"operation": operation})
    assert public["status"] == ("completed" if exit_code == 0 else "failed")
    assert public["result"] == {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "outcome": "terminal"}


def test_command_passes_upstream_error_without_inventing_fields():
    error = {"reason": "command_failed", "message": "upstream detail", "details": {"a": [1, 2]}}
    job = {"job_id": "dv_probe", "status": "failed", "error": error, "result": {"exit_code": 1}}
    receipt = command_receipt(job, "stdout", "stderr")
    response = with_execution_error({"exit_code": 1, "stdout": "stdout", "stderr": "stderr"}, receipt, operation="command")
    assert response["error"] == error
    assert not {"candidate_repairable", "next_action", "execution_status"} & response.keys()


@pytest.mark.parametrize('failure', [None, 'candidate', 'infrastructure'])
def test_dev_launcher_does_not_inject_logs_into_stderr(tmp_path, monkeypatch, capsys, failure):
    from aka.execution.gateway import sandbox, sdk
    (tmp_path / 'kernel.py').write_text('def kernel(): pass\n')
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'scratch/probe.py').write_text('print("probe output")\n')
    remote_stderr = ('probe stderr without trailing newline' if failure == 'candidate'
                     else '__ATREX_INFRASTRUCTURE_UNAVAILABLE__\n\n' if failure is None else '')
    job = {'job_id': 'dv_probe', 'status': 'failed' if failure == 'infrastructure' else 'succeeded',
           'result': None if failure == 'infrastructure' else {
               'stdout': 'probe output\n', 'stderr': remote_stderr,
               'exit_code': 1 if failure == 'candidate' else 0}}
    if failure == 'infrastructure':
        job['error'] = {'error_class': 'infra', 'reason': 'logs_unavailable',
                        'message': 'backend terminal output was not available within the collection budget'}
    facts = []
    monkeypatch.setattr('aka.execution.outcomes.emit_fact', facts.append)
    monkeypatch.setattr(sdk, 'route', lambda *args: 'test-endpoint')
    def remote(**kwargs):
        process = subprocess.CompletedProcess([], 1 if failure == 'infrastructure' else 0,
                                              json.dumps(job), '')
        observe_attempt(process, operation='dev')
        return process
    monkeypatch.setattr(sandbox, '_run_sdk_dev', remote)
    code = sandbox._main(['--workspace', str(tmp_path), '--hardware', 'L20N',
                          '--url', 'test-endpoint', '--kind', 'dev', '--no-sync',
                          '--', 'python3', '-m', 'scratch.probe'])
    output = capsys.readouterr()
    # A real probe may print the former marker: retain it as ordinary output.
    assert output.err == remote_stderr
    assert output.out == ('' if failure == 'infrastructure' else 'probe output\n')
    receipt = captured_result(subprocess.CompletedProcess([], code, output.out, output.err),
                              resource=ResourceIdentity('gateway'), records=facts, decode_output=False)
    public = canonical_result(with_execution_error({'exit_code': code, 'stdout': output.out,
        'stderr': output.err}, receipt, operation='command'), {'operation': 'command'})
    assert public['result']['stderr'] == remote_stderr
    assert receipt.attempts[-1].backend_response == job
    if failure == 'infrastructure':
        assert code == 75
        assert receipt.transport_status == 'infrastructure'
        assert public['result']['error'] == job['error']
    else:
        assert code == (1 if failure == 'candidate' else 0)
        assert receipt.transport_status != 'infrastructure'


@pytest.mark.parametrize('failure', ['upload', 'http_503'])
def test_prepared_submission_reports_infrastructure_through_facts_not_stderr(tmp_path, monkeypatch, capsys, failure):
    from aka.execution.gateway import sandbox, sdk, submission, source_archives
    from aka.execution.gateway.errors import GatewayHTTPError
    monkeypatch.setattr(sdk, 'route', lambda *args: 'test-endpoint')
    monkeypatch.setattr(source_archives, 'read_archives', lambda _: ())
    monkeypatch.setattr(source_archives, 'validate_archives', lambda *args, **kwargs: None)
    from aka.execution.gateway.source_archives import SourceArchiveError
    error = (SourceArchiveError('oss_upload_failed') if failure == 'upload' else
             GatewayHTTPError(503, json.dumps({'error': {'error_class': 'infra', 'reason': 'unavailable'}})))
    def rejected(**kwargs):
        raise error
    monkeypatch.setattr(sandbox, '_run_sdk_job', rejected)
    facts = []
    monkeypatch.setattr('aka.execution.outcomes.emit_fact', facts.append)
    args = sandbox.build_parser().parse_args(['--workspace', str(tmp_path), '--hardware', 'L20N', '--kind', 'run'])
    code = submission.submit_prepared(args, tmp_path, 0, {'schema_version': 1, 'kind': 'run', 'request': {}})
    output = capsys.readouterr()
    assert output.err == ''
    assert code == 75
    receipt = captured_result(subprocess.CompletedProcess([], code, output.out, output.err),
                              resource=ResourceIdentity('gateway'), records=facts, decode_output=False)
    assert receipt.transport_status == 'infrastructure'
