from copy import deepcopy
from types import SimpleNamespace
import pytest
from aka.workflow.light.predictions import freeze_predictions, validate_predictions
from aka.workflow.light.host import journal
from aka.workflow.light.host.errors import AgentRequestError
from aka.experiment.measurement_store import MeasurementStore, read_json
from aka.workflow.light.measurements import SharedMeasurements


def test_freeze_and_recovery_preserve_original(tmp_path):
    store = SimpleNamespace(root=tmp_path)
    rows = [{'scope': 'long', 'prediction': 'faster'}]
    original = freeze_predictions(store, 'gateway-test', rows)
    rows[0]['prediction'] = 'slower'
    assert freeze_predictions(store, 'gateway-test', rows) == original
    assert original['predictions'][0]['prediction'] == 'faster'
    assert freeze_predictions(store, 'gateway-empty', [])['predictions'] == []
    assert freeze_predictions(store, 'gateway-empty', rows)['predictions'] == []


def test_prediction_binding_and_revision(monkeypatch, tmp_path):
    frozen = {'recorded_at': 'before', 'predictions': [
        {'scope': 'long', 'prediction': 'faster'}, {'scope': 'short', 'prediction': 'no regression'}]}
    monkeypatch.setattr(journal, '_read_private_gateway_evidence', lambda *a, **k: {'frozen_predictions': frozen})
    current = {'gateway_record_ids': ['gateway-x']}
    journal._bind_experiment_predictions(current, None, tmp_path, tmp_path)
    assert len(current['expectations']) == 2
    assert all(r['prediction_origin'] == 'before_execution' for r in current['expectations'])
    assert all(r['assessment'] == 'unresolved' for r in current['expectations'])
    previous = deepcopy(current)
    current['expectations'] = [dict(scope='long', prediction='faster', observation='slower', assessment='missed')]
    journal._bind_experiment_predictions(current, previous, tmp_path, tmp_path)
    assert current['expectations'][0]['assessment'] == 'missed'
    assert len(current['expectations']) == 2
    previous = deepcopy(current)
    current = {'gateway_record_ids': [], 'expectations': []}
    journal._bind_experiment_predictions(current, previous, tmp_path, tmp_path)
    assert current['expectations'] == previous['expectations']
    current['expectations'] = [dict(scope='long', prediction='slower', observation='slower', assessment='met')]
    journal._bind_experiment_predictions(current, previous, tmp_path, tmp_path)
    assert current['expectations'][0]['prediction'] == 'faster'
    assert current['expectations'][-1]['prediction_origin'] == 'retrospective'


def test_ambiguous_forecasts_not_assessed_by_guess(monkeypatch, tmp_path):
    frozen = {'recorded_at': 'before', 'predictions': [{'scope': 'long', 'prediction': 'faster'}]}
    monkeypatch.setattr(journal, '_read_private_gateway_evidence', lambda *a, **k: {'frozen_predictions': frozen})
    current = {'gateway_record_ids': ['gateway-x', 'gateway-y'], 'expectations': [
        dict(scope='long', prediction='faster', observation='slower', assessment='missed')]}
    journal._bind_experiment_predictions(current, None, tmp_path, tmp_path)
    assert [r['assessment'] for r in current['expectations'][:2]] == ['unresolved', 'unresolved']


def test_execute_freezes_before_execution_and_cache_does_not_register(tmp_path):
    shared = object.__new__(SharedMeasurements)
    shared.store = MeasurementStore(tmp_path)
    calls = []
    def execute(request, request_id, **kwargs):
        frozen = read_json(tmp_path / 'records' / request_id / 'predictions.json')
        calls.append(frozen)
        return {'exit_code': 0, 'stdout': '', 'stderr': '', 'agent_result': {
            'operation': 'evaluate', 'status': 'completed', 'result': {'correct': False}}}
    shared._execute = execute
    identity = {'operation': 'evaluate', 'options': {}, 'inputs': {}}
    original = [{'scope': 'long', 'prediction': 'faster'}]
    first = shared.execute(None, identity=identity, kernels={}, predictions=original)
    assert first['agent_result']['frozen_predictions']['predictions'] == original
    second = shared.execute(None, identity=identity, kernels={}, predictions=[{'scope': 'long', 'prediction': 'slower'}])
    assert len(calls) == 1
    assert second['reused']
    assert second['agent_result']['frozen_predictions'] == first['agent_result']['frozen_predictions']
    assert 'not registered' in second['agent_result']['prediction_registration']


@pytest.mark.parametrize('value', [None, {}, [None], [{'scope':'x','prediction':''}], [{'scope':'x','prediction':'p','observation':'o'}]])
def test_bad_predictions(value):
    with pytest.raises(ValueError):
        validate_predictions(value)


def test_cli_removes_only_host_prediction_option(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).parents[1] / 'src/aka/resources/tools/sandbox.py'
    spec = importlib.util.spec_from_file_location('prediction_client', script)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    raw = '[{"scope":"long","prediction":"faster"}]'
    received = []
    monkeypatch.setattr(client, 'proxy_command', lambda route, payload: received.append(payload) or 0)
    client.main(['--kind', 'command', '--predictions', raw, '--', 'python', '--predictions', 'remote'])
    assert received[0]['argv'] == ['--kind', 'command', '--', 'python', '--predictions', 'remote']
    assert received[0]['predictions'] == [{'scope':'long','prediction':'faster'}]


@pytest.mark.parametrize("argv, expected", [
    (["--kind", "evaluate"], {"argv": ["--kind", "evaluate"]}),
    (["--kind", "evaluate", "--predictions=[]"], {"argv": ["--kind", "evaluate"], "predictions": []}),
    (["--kind", "evaluate", "--predictions=null"], {"argv": ["--kind", "evaluate"], "predictions": None}),
])
def test_cli_optional_prediction_field(monkeypatch, argv, expected):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("prediction_client", Path(__file__).parents[1] / "src/aka/resources/tools/sandbox.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    received = []
    monkeypatch.setattr(client, "proxy_command", lambda route, payload: received.append(payload) or 0)
    client.main(argv)
    assert received == [expected]
