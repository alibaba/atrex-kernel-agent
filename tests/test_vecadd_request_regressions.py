"""Failures observed at the deployed Runtime / Agate boundary."""
import json
import runpy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aka.execution.gateway import sandbox, sdk
from aka.execution.gateway.errors import GatewayHTTPError
from aka.execution.gateway.jobs import causes_from_error
from aka.contracts.execution import JobResult, ExecutionFacts, ExecutionDiagnostic
from aka.evaluation.tool_results import canonical_result, with_execution_error


def test_module_transport_and_sdk_share_exception_identity():
    # `python -m ...sandbox` loads a different module object. Exceptions still
    # need to reach its typed HTTP rejection branch rather than RuntimeError.
    with pytest.warns(RuntimeWarning):
        module = runpy.run_module('aka.execution.gateway.sandbox', run_name='gateway_entrypoint_test')
    assert module['GatewayHTTPError'] is GatewayHTTPError
    from atrex_gateway_client.errors import GatewayError
    api = Mock()
    payload = {'reason': 'request_validation_failed', 'details': {
        'validation_errors': [{'loc': ['intent'], 'msg': 'Extra inputs are not permitted'}]}}
    api.submit_job.side_effect = GatewayError(422, payload)
    with patch.object(sdk, 'client', return_value=api):
        with pytest.raises(module['GatewayHTTPError']) as rejected:
            sdk.request('endpoint', 'POST', '/v1/jobs/dev', {}, 30)
    assert json.loads(rejected.value.detail)['error'] == payload


def test_command_payload_does_not_invent_native_fields(tmp_path):
    script = tmp_path / 'probe.py'
    script.write_text('print(1)\n')
    with patch.object(sandbox, '_run_sdk_job') as submit:
        sandbox._run_sdk_dev(url='endpoint', hardware='L20N', timeout=60,
            queue_wait_grace=0, env_items=['TORCH_LOGS=recompiles'],
            files={'scratch/probe.py': script}, command='python3 -m scratch.probe')
    payload = submit.call_args.kwargs['payload']
    assert set(payload) == {'spec', 'command', 'timeout_s', 'env_vars', 'files'}
    assert payload['env_vars'] == {'TORCH_LOGS': 'recompiles'}
    assert payload['files']['scratch/probe.py'] == script.read_text()


@pytest.mark.parametrize('status,reason', [(400, 'invalid_source'), (422, 'request_validation_failed')])
def test_upstream_validation_details_reach_agent_receipt(status, reason):
    details = {'validation_errors': [{'loc': ['env_vars', 'HOME'], 'msg': 'not allowed'}]}
    error = GatewayHTTPError(status, json.dumps({'error': {
        'error_class': 'validation', 'reason': reason, 'details': details}}))
    result = JobResult(2, stderr=f'HTTP {status}', transport_status='failed',
        facts=ExecutionFacts(disposition='not_submitted'),
        diagnostics=(ExecutionDiagnostic('request', 'gateway-http-rejection', {'http_status': status}),),
        causes=causes_from_error(error))
    response = with_execution_error({'exit_code': 2, 'stdout': '', 'stderr': result.stderr}, result)
    receipt = canonical_result(response, {'operation': 'evaluate', 'options': {}})
    value = receipt['result']['error']
    assert value['http_status'] == status
    assert value['code'] == reason
    assert value['details'] == details
    assert value['job_submitted'] is False
    assert 'no GPU Job' in receipt['result']['next_action']


def test_wiki_exclusion_survives_child_registry_reconstruction(tmp_path, monkeypatch):
    from aka.resources.plugins import PluginRegistry
    from aka.contracts.agent_resources import AgentSkill
    plugin = tmp_path / 'plugin'
    plugin.mkdir()
    (plugin / 'SKILL.md').write_text('Wiki usage')
    (plugin / 'plugin.json').write_text(json.dumps({'id': 'gpu-wiki', 'version': '1.0.0', 'api_version': 1,
        'skills': {'wiki': {'path': '.'}}}))
    skill_root = tmp_path / 'skill'
    skill_root.mkdir()
    (skill_root / 'SKILL.md').write_text('GPU Wiki instructions')
    skills = (AgentSkill('kernel-wiki', skill_root, 'Query Wiki'),)
    enabled = PluginRegistry(plugins=[plugin], resources=(), agent_skills=skills,
        excluded_plugins=(), excluded_skills=())
    assert [item.id for item in enabled.plugins] == ['gpu-wiki']
    assert [item.name for item in enabled._agent_skills] == ['kernel-wiki']
    disabled = PluginRegistry(plugins=[plugin], resources=(), agent_skills=skills,
        excluded_plugins=('gpu-wiki',), excluded_skills=('kernel-wiki',))
    assert disabled.plugins == []
    assert disabled._agent_skills == ()
    disabled.check_resource_snapshot(disabled.resource_snapshot())
    for key, value in disabled.environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    child = PluginRegistry()
    assert child.plugins == []
    assert child._agent_skills == ()


def test_concurrent_lineages_share_one_measurement_reservation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from aka.experiment.measurement_store import MeasurementStore, TaskRunning
    stores = [MeasurementStore(tmp_path), MeasurementStore(tmp_path)]
    for index in range(20):
        barrier = threading.Barrier(2)
        loser = threading.Event()
        def reserve(store):
            barrier.wait(timeout=5)
            try:
                with store.reserve({'operation':'evaluate', 'test':index}, {}):
                    assert loser.wait(timeout=5)
                    return 'owner'
            except TaskRunning:
                loser.set()
                return 'waiting'
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(reserve, stores)) == ['owner', 'waiting']
