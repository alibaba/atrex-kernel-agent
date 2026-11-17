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


@pytest.mark.parametrize('shared_instance', [False, True])
@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('operation', ['evaluate', 'same_allocation_abba'])
def test_concurrent_lineages_receive_one_measurement_result(tmp_path, shared_instance, failed, operation):
    from concurrent.futures import ThreadPoolExecutor
    from copy import deepcopy
    import threading
    from aka.experiment.measurement_store import MeasurementStore
    from aka.workflow.light.measurements import SharedMeasurements

    store = MeasurementStore(tmp_path)
    entered, release = threading.Event(), threading.Event()
    waiting = [threading.Event(), threading.Event()]
    original = {'exit_code': 1 if failed else 0, 'stdout': 'original output',
                'stderr': 'candidate failed' if failed else '',
                'agent_result': {'operation': operation, 'status': 'failed' if failed else 'completed',
                                 'result': {'correct': None if failed else True, 'latency_us_geomean': 4.0}}}
    def execute(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return deepcopy(original)
    execution = Mock(side_effect=execute)
    def bridge():
        owner = SharedMeasurements.__new__(SharedMeasurements)
        owner.store = store if shared_instance else MeasurementStore(tmp_path)
        owner._execute = execution
        return owner
    owners = [bridge() for _ in range(3)]
    identity = {'operation': operation, 'options': {}, 'inputs': {}, 'command': ['measure']}
    def invoke(index):
        attempts = 0
        def check_wait():
            nonlocal attempts
            attempts += 1
            if index and attempts > 1:
                waiting[index - 1].set()  # A failed lock acquisition already occurred.
        return owners[index].execute(None, request_id=f'T{index + 1}', identity=identity,
                                     kernels={}, check_wait=check_wait)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(invoke, 0)
        try:
            assert entered.wait(5)
            followers = [pool.submit(invoke, i) for i in (1, 2)]
            assert all(event.wait(5) for event in waiting)
            assert all(not future.done() for future in followers)
            assert execution.call_count == 1
        finally:
            release.set()
        responses = [first.result(5), *(future.result(5) for future in followers)]
    assert execution.call_count == 1
    assert all({key: value for key, value in result['agent_result'].items()
                if key != 'reused_gateway_record_id'} == responses[0]['agent_result'] for result in responses)
    assert all(result['exit_code'] == original['exit_code'] for result in responses)
    assert all(result['stdout'] == original['stdout'] for result in responses)
    assert all('gateway_task_running' not in json.dumps(result) for result in responses)
    record_id = responses[0]['agent_result']['gateway_record_id']
    assert store.record_ids() == [record_id]
    assert store.read(record_id)['runtime_request_id'] == 'T1'
    assert responses[0]['agent_result']['record_path'] == f'records/{record_id}.json'
    if failed:
        # A later, independent retry may execute again; concurrent waiters may not.
        retried = invoke(0)
        assert execution.call_count == 2
        assert retried['agent_result']['gateway_record_id'] != record_id
    else:
        cached = invoke(0)
        assert execution.call_count == 1
        assert cached['agent_result'] == dict(responses[0]['agent_result'], reused_gateway_record_id=record_id)


@pytest.mark.parametrize('operation', ['evaluate', 'same_allocation_abba'])
@pytest.mark.parametrize('normal_result', [
    {'correct': False},
    {'error': {'error_class': 'compilation', 'message': 'candidate compile failed'}},
    {'latency_us_geomean': None},
    {},
])
def test_evaluate_retries_gateway_failures_then_reuses_normal_response(tmp_path, operation, normal_result):
    from copy import deepcopy
    from aka.experiment.measurement_store import MeasurementStore
    from aka.workflow.light.measurements import SharedMeasurements

    owner = SharedMeasurements.__new__(SharedMeasurements)
    owner.store = MeasurementStore(tmp_path)
    failures = [
        {'error': {'error_class': 'infra', 'reason': 'logs_unavailable'}},
        {'error': {'error_class': 'gateway', 'http_status': 503}},
        {'outcome': 'timeout', 'error': {'error_class': 'timeout'}},
    ]
    responses = [{'exit_code': 75, 'stdout': '', 'stderr': '',
                  'agent_result': {'operation': operation, 'status': 'failed', 'result': item}}
                 for item in failures]
    responses.append({'exit_code': 1, 'stdout': '', 'stderr': 'original candidate diagnostic',
                      'evaluation_response_received': True,
                      'agent_result': {'operation': operation, 'status': 'failed', 'result': normal_result}})
    owner._execute = Mock(side_effect=deepcopy(responses))
    identity = {'operation': operation, 'options': {}, 'command': ['same request']}
    returned = [owner.execute(None, identity=identity, kernels={}) for _ in responses]
    assert owner._execute.call_count == 4
    record_ids = [item['agent_result']['gateway_record_id'] for item in returned]
    assert len(set(record_ids)) == 4
    assert all('reused_gateway_record_id' not in item['agent_result'] for item in returned)
    assert [owner.store.read(record_id)['cacheable'] for record_id in record_ids] == [False, False, False, True]
    for _ in range(2):
        reused = owner.execute(None, identity=identity, kernels={})
        assert reused['agent_result'] == dict(returned[-1]['agent_result'], reused_gateway_record_id=record_ids[-1])
        assert reused['exit_code'] == returned[-1]['exit_code']
        assert reused['stderr'] == returned[-1]['stderr']
    assert owner._execute.call_count == 4
    assert len(owner.store.record_ids()) == 4  # Failed attempts remain auditable.


@pytest.mark.parametrize('status,disposition,cause,expected', [
    ('succeeded', 'terminal', None, True),
    ('failed', 'terminal', 'compilation', True),
    ('failed', 'terminal', 'correctness', True),
    ('failed', 'terminal', 'infra', False),
    ('infrastructure', 'terminal', None, False),
    ('unknown', 'unknown', None, False),
    ('failed', 'not_submitted', 'validation', False),
    ('timed_out', 'unknown', None, False),
    ('failed', 'terminal', 'gateway', False),
])
def test_evaluate_cache_uses_terminal_response_not_verdict(status, disposition, cause, expected):
    from aka.contracts.execution import ExecutionFacts, JobResult
    from aka.contracts.execution_recovery import ExecutionCause
    from aka.evaluation.tool_results import normal_evaluation_execution, cacheable_evaluation

    receipt = JobResult(0, exit_code=1, transport_status=status,
        facts=ExecutionFacts(disposition=disposition),
        causes=(ExecutionCause(cause, 'test'),) if cause else ())
    received = normal_evaluation_execution(receipt)
    assert received is expected
    response = {'evaluation_response_received': received,
                'agent_result': {'operation': 'evaluate', 'status': 'unknown',
                                 'result': {'correct': None, 'latency_us_geomean': None}}}
    assert cacheable_evaluation(response, {'operation': 'evaluate'}) is expected


def test_evaluate_cache_excludes_structured_and_embedded_infrastructure_failures():
    from aka.evaluation.tool_results import cacheable_evaluation
    response = {'evaluation_response_received': True,
                'agent_result': {'operation': 'evaluate', 'status': 'completed', 'result': {}}}
    request = {'operation': 'evaluate'}
    assert cacheable_evaluation(response, request)
    for failure in (
        {'error': {'error_class': 'infra'}},
        {'error': {'error_class': 'code', 'causes': [{'error_class': 'infra'}]}},
        {'error': 'CUDA error: all CUDA-capable devices are busy or unavailable'},
    ):
        response['agent_result']['result'] = failure
        assert not cacheable_evaluation(response, request)
    response['agent_result']['result'] = {'error': {'error_class': 'compilation',
        'message': 'undeclared variable infrastructure; no performance data'}}
    assert cacheable_evaluation(response, request)  # Error text cannot invent an infra failure.
    response['stdout'] = '[test_kernel] RESULT_JSON=' + json.dumps({
        'correct': False, 'error': 'CUDA error: all CUDA-capable devices are busy or unavailable'})
    assert not cacheable_evaluation(response, request)


@pytest.mark.parametrize('recovered', [False, True])
def test_evaluate_response_availability_uses_latest_attempt_of_every_shard(recovered):
    from types import SimpleNamespace
    from aka.contracts.execution import ExecutionFacts, JobResult
    from aka.evaluation.tool_results import cacheable_evaluation
    from aka.workflow.light.measurements import SharedMeasurements

    normal = JobResult(0, transport_status='succeeded', facts=ExecutionFacts(disposition='terminal'))
    infra = JobResult(75, transport_status='infrastructure', facts=ExecutionFacts(disposition='terminal'))
    attempts = [SimpleNamespace(task_id='first', result=infra),
                SimpleNamespace(task_id='second', result=normal)]
    if recovered:
        attempts.append(SimpleNamespace(task_id='first', result=normal))
    record = SimpleNamespace(lifecycle='terminal', phase='complete', kind='evaluation',
        evaluation=object(), attempts=attempts,
        steps=[SimpleNamespace(task_id=name) for name in ('first', 'second', 'unneeded-after-rejection')])
    owner = SharedMeasurements.__new__(SharedMeasurements)
    owner._evidence = Mock(return_value={'gateway_kind': 'run', 'gateway_record_id': 'gateway-test',
        'execution_status': 'unknown', 'exit_code': 0, 'result': {'correct': None}})
    response = owner._response(Mock(), record)
    assert response['evaluation_response_received'] is recovered
    assert cacheable_evaluation(response, {'operation': 'evaluate'}) is recovered


def _join_measurement_process(root, request, waiting, output):
    from aka.experiment.measurement_store import MeasurementStore, DuplicateTask
    attempts = 0
    def check_wait():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            waiting.set()
    try:
        store = MeasurementStore(root)
        with store.reserve(request, {}, check_wait=check_wait):
            output.put(('unexpected_owner', None))
    except DuplicateTask as result:
        output.put((result.record_id, store.read(result.record_id)['response']['stdout']))


def test_measurement_wait_crosses_process_boundary(tmp_path):
    import multiprocessing
    from aka.experiment.measurement_store import MeasurementStore
    context = multiprocessing.get_context('spawn')
    waiting, output = context.Event(), context.Queue()
    request = {'operation': 'command', 'inputs': {}, 'options': {}}
    store = MeasurementStore(tmp_path)
    process = context.Process(target=_join_measurement_process, args=(tmp_path, request, waiting, output))
    try:
        with store.reserve(request, {}) as owner:
            process.start()
            assert waiting.wait(10)
            owner.finish({'exit_code': 0, 'stdout': 'shared', 'stderr': ''}, cacheable=True)
        assert output.get(timeout=10) == (owner.record_id, 'shared')
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        output.close()
        output.join_thread()


@pytest.mark.parametrize('stop', ['timeout', 'cancel'])
def test_waiting_caller_can_exit_without_cancelling_owner(tmp_path, stop):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from aka.experiment.measurement_store import MeasurementStore, MeasurementWaitTimeout, DuplicateTask
    from aka.workflow.light.support.errors import SessionRevokedError
    owner_store, follower_store = MeasurementStore(tmp_path), MeasurementStore(tmp_path)
    request = {'operation': 'command'}
    cancel, waiting = threading.Event(), threading.Event()
    attempts = 0
    def check_wait():
        nonlocal attempts
        if cancel.is_set():
            raise SessionRevokedError('cancelled waiter')
        attempts += 1
        if attempts > 1:
            waiting.set()
    def follower():
        with follower_store.reserve(request, {}, check_wait=check_wait,
                deadline=time.monotonic() + .2 if stop == 'timeout' else None):
            pytest.fail('waiter must not execute')
    with owner_store.reserve(request, {}) as owner:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(follower)
            assert waiting.wait(5)
            if stop == 'cancel':
                cancel.set()
            with pytest.raises(MeasurementWaitTimeout if stop == 'timeout' else SessionRevokedError):
                future.result(timeout=5)
        owner.finish({'exit_code': 0, 'stdout': 'owner completed', 'stderr': ''}, cacheable=True)
    with pytest.raises(DuplicateTask) as hit:
        with follower_store.reserve(request, {}):
            pytest.fail('owner result should remain reusable')
    assert hit.value.record_id == owner.record_id
    assert len(owner_store.record_ids()) == 1


def test_waiter_recovers_interrupted_reservation_without_new_record(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from aka.experiment.measurement_store import MeasurementStore
    owner_store, follower_store = MeasurementStore(tmp_path), MeasurementStore(tmp_path)
    request = {'operation': 'command'}
    waiting = threading.Event()
    attempts = 0
    def check_wait():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            waiting.set()
    def follower():
        with follower_store.reserve(request, {}, check_wait=check_wait) as task:
            task.finish({'exit_code': 0, 'stdout': 'recovered', 'stderr': ''}, cacheable=True)
            return task.record_id
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='owner interrupted'):
            with owner_store.reserve(request, {}) as owner:
                future = pool.submit(follower)
                assert waiting.wait(5)
                raise RuntimeError('owner interrupted')
        assert future.result(timeout=5) == owner.record_id
    assert len(owner_store.record_ids()) == 1


def test_transport_exception_keeps_original_receipt_in_measurement_store(tmp_path):
    from aka.workflow.light.measurements import SharedMeasurements
    from aka.experiment.measurement_store import MeasurementStore
    from aka.contracts.execution import ExecutionUnavailable, ExecutionCause
    owner = SharedMeasurements.__new__(SharedMeasurements)
    owner.store = MeasurementStore(tmp_path)
    result = JobResult(75, stderr='service temporarily unavailable', transport_status='infrastructure',
        facts=ExecutionFacts(disposition='not_submitted'),
        diagnostics=(ExecutionDiagnostic('infrastructure', 'gateway-http-rejection', {'http_status':503}),),
        causes=(ExecutionCause('infrastructure','gateway_unavailable',details={'reason':'backend unavailable'}),))
    owner._execute = Mock(side_effect=ExecutionUnavailable('GPU validation transport unavailable',result))
    response = owner.execute(None,identity={'operation':'evaluate','options':{}},kernels={})
    agent = response['agent_result']
    assert agent['result']['error']['code'] == 'gateway_unavailable'
    assert agent['result']['error']['http_status'] == 503
    assert agent['result']['error']['details'] == {'reason':'backend unavailable'}
    assert owner.store.read(agent['gateway_record_id'])['response'] == response


# A functional-test reference of 100 us used to fabricate a 17x "speedup"
# for an unchanged V0. Normalized measurement evidence now contains latency only.
def _latency_job(shape_ids=('0', '1')):
    return {
        'passed': {'compile': {'status': 'passed'},
                   'correctness': {sid: {'status': 'passed'} for sid in shape_ids}},
        'correctness': {'shapes': {sid: {'cases': [{'outputs': [
            {'max_elementwise_abs_diff': 0., 'max_elementwise_rel_diff': 0.}]}]} for sid in shape_ids}},
        'performance': {'shapes': {sid: {'samples': [{'end_to_end_time_ms': value}]}
                                  for sid, value in zip(shape_ids, (.004, .016))}},
    }


@pytest.mark.parametrize('metadata', [None, {}, {
    'performance_objective': 'shape_speedup_geometric_mean',
    'shapes': {sid: {'production_performance': {'performance_us': 100.,
        'basis': 'fixed_scoring_normalization', 'measured': False}} for sid in ('0', '1')},
}])
def test_native_and_sdk_results_need_latency_not_reference_scores(metadata):
    from aka.evaluation.transport_results import _optimizer_result_from_eval
    from aka.evaluation.drivers.native_harness import result_from_eval
    from aka.evaluation.objectives import SCORE_FIELDS
    for convert in (_optimizer_result_from_eval, result_from_eval):
        result = convert(_latency_job(), ['0', '1'], metadata)
        assert result['all_pass'] is True
        assert result['latency_us_geomean'] == pytest.approx(8.)
        assert result['latency_us_arith_mean'] == pytest.approx(10.)
        assert result['latency_us_by_shape'] == {'0': 4., '1': 16.}
        assert not SCORE_FIELDS.intersection(result)
        raw = _latency_job()
        raw['passed']['correctness']['1']['status'] = 'failed'
        assert convert(raw, ['0', '1'], metadata)['all_pass'] is False
        raw = _latency_job()
        raw['performance']['shapes'].pop('1')
        assert convert(raw, ['0', '1'], metadata)['all_pass'] is False


def test_all_normalized_receipt_paths_remove_scores_without_touching_raw_output():
    import copy
    from aka.evaluation.objectives import SCORE_FIELDS
    from aka.evaluation.tool_results import compact_receipt
    side = {'all_pass': True, 'latency_us_geomean': 4., 'latency_us_by_shape': {'0': 4.},
            'speedup_vs_ref_geomean': 25., 'performance_score': 25.,
            'performance_objective': 'shape_speedup_geometric_mean'}
    for operation in ('evaluate', 'same_allocation_abba'):
        sample = side if operation == 'evaluate' else {
            'correct': True, 'baseline': side, 'candidate': side,
            'speedup': 1., 'improvement_pct': 0.}
        prefix = '[test_kernel] RESULT_JSON=' if operation == 'evaluate' else '[sandbox] ABBA_JSON='
        response = {'exit_code': 0, 'stdout': prefix + json.dumps(sample), 'stderr': ''}
        original = copy.deepcopy(response)
        receipt = canonical_result(response, {'operation': operation})
        assert response == original
        assert not SCORE_FIELDS.intersection(receipt['result'])
        for metrics in ([receipt['result']] if operation == 'evaluate' else
                        [receipt['result'][s] for s in ('baseline', 'candidate')]):
            assert metrics['latency_us_by_shape'] == {'0': 4.}
            assert not SCORE_FIELDS.intersection(metrics)
        cached = {'operation': operation, 'gateway_record_id': 'gateway-' + 'a'*32, 'result': sample}
        original = copy.deepcopy(cached)
        assert not SCORE_FIELDS.intersection(canonical_result({'agent_result': cached}, {'operation': operation})['result'])
        assert not SCORE_FIELDS.intersection(compact_receipt(cached)['result'])
        assert cached == original
    output = '{"performance_score": 17}'
    for operation in ('command', 'dev'):
        receipt = canonical_result({'exit_code': 0, 'stdout': output, 'stderr': 'raw error'}, {'operation': operation})
        assert receipt['result']['stdout'] == output
        assert receipt['result']['stderr'] == 'raw error'


def test_abba_and_repeated_measurements_do_not_require_scores():
    from aka.evaluation.abba import interpret_verification_payload, verification_schedule, _merge_batch_results
    from aka.evaluation.aggregation import aggregate
    from aka.evaluation.objectives import SCORE_FIELDS
    schedule = verification_schedule(2)
    sample = {'all_pass': True, 'latency_us_geomean': 4., 'latency_us_by_shape': {'0': 4.}}
    payload = {'schema_version': 1, 'runs': [dict(step, exit_code=0, result=dict(sample)) for step in schedule]}
    assert interpret_verification_payload(payload, schedule=schedule, repeats=2).gate == 'PASS'
    assert _merge_batch_results([sample], ['0'])['all_pass'] is True
    for count in (1, 2):
        result = aggregate([{**sample, 'performance_score': 25.} for _ in range(count)], 'evaluate')
        assert not SCORE_FIELDS.intersection(result)
        paired = aggregate([{'correct': True, 'baseline': sample, 'candidate': sample, 'speedup': 1.}
                            for _ in range(count)], 'same_allocation_abba')
        assert not SCORE_FIELDS.intersection(paired)
    payload['runs'][0]['result']['latency_us_geomean'] = None
    assert interpret_verification_payload(payload, schedule=schedule, repeats=2).gate == 'FAIL'


def test_sol_full_evidence_accepts_latency_without_speedup():
    from aka.task.problem.sol_harness import _summarize
    from aka.evaluation.sol_evidence import validate_result
    from aka.evaluation.objectives import SCORE_FIELDS
    traces = [{'workload': {'uuid': 'shape-a'}, 'evaluation': {'status': 'PASSED',
        'performance': {'latency_ms': .004}}}]
    summary = _summarize(traces)
    validate_result(summary, ('shape-a',))
    assert summary['all_pass'] is True
    assert not SCORE_FIELDS.intersection(summary)
    summary['latency_us_by_shape'] = {}
    with pytest.raises(ValueError, match='incomplete'):
        validate_result(summary, ('shape-a',))
