from copy import deepcopy

import pytest

from aka.workflow.light.host.errors import AgentRequestError
from aka.workflow.light.host.journal import record_experiment, _public_experiment
from aka.workflow.light.host.request_feedback import journal_schema
from aka.workflow.light.research import ResearchStore


def state():
    return {"episode": 1, "state": "in_progress", "tool_modules": ["experiments"],
            "central_research": {"directions": {}, "experiments": {}, "queue": []},
            "experiments": []}


def request():
    return {"name": "Output layout probe", "intent": "Compare layouts", "interpretation": "Try a new layout",
            "implementation": "Microbenchmark contiguous and strided stores",
            "expectations": [{"scope": "long sequences", "prediction": "Lower latency",
                              "observation": "No valid measurement yet", "assessment": "unresolved"}]}


def test_expectations_reach_review_archive_projection_and_revisions(tmp_path):
    current = state()
    reviewed = []
    def review(kind, target):
        reviewed.append(deepcopy(target))
        return {"claim_review_quality": "review_not_requested", "review_skipped": True}
    first = record_experiment(current, tmp_path, request(), review_claim=review)
    original = deepcopy(current["experiments"][0])
    assert reviewed[0]["expectations"] == [dict(request()["expectations"][0], prediction_origin="retrospective")]
    store = ResearchStore(tmp_path / "research")
    store.observe_experiment(original)

    revision = {k: v for k, v in request().items() if k not in {"implementation", "expectations"}}
    revision["experiment_id"] = first["experiment_id"]
    record_experiment(current, tmp_path, revision, review_claim=review)
    assert reviewed[-1]["implementation"] == original["implementation"]
    assert reviewed[-1]["expectations"] == original["expectations"]

    revision["expectations"] = [dict(request()["expectations"][0], observation="Measured regression", assessment="missed")]
    record_experiment(current, tmp_path, revision, review_claim=review)
    latest = current["experiments"][0]
    store.observe_experiment(latest)
    assert store.read()["experiments"][first["experiment_id"]]["expectations"][0]["assessment"] == "missed"
    assert store.events()[0]["payload"]["experiment"]["expectations"][0]["assessment"] == "unresolved"
    assert _public_experiment(latest)["expectations"] == [dict(revision["expectations"][0], prediction_origin="retrospective")]
    assert _public_experiment(latest)["implementation"] == original["implementation"]
    assert 'Measured regression' in str(store.board())

    revision["expectations"] = []
    record_experiment(current, tmp_path, revision)
    assert current["experiments"][0]["expectations"] == []


@pytest.mark.parametrize("bad", [None, {}, [None], [{}], [{"scope": "x", "prediction": "p", "observation": "o", "assessment": "supported"}], [{"scope": "x", "prediction": "p", "observation": None, "assessment": "unresolved"}], [{"scope": "x", "prediction": "p", "observation": "o", "assessment": "met", "latency": 1}]])
def test_invalid_expectations_do_not_record_or_review(tmp_path, bad):
    current = state()
    reviewed = []
    value = request()
    value["expectations"] = bad
    with pytest.raises(AgentRequestError):
        record_experiment(current, tmp_path, value, review_claim=lambda *args: reviewed.append(args))
    assert not current["experiments"]
    assert not reviewed


def test_exposed_schema_matches_structured_request():
    import jsonschema
    schema = journal_schema("experiment_record", {}, experiments_enabled=True, directions_enabled=False)
    jsonschema.validate(request(), schema)
    assert "direction_id" not in schema["properties"]
    assert set(schema["properties"]["expectations"]["items"]["properties"]["assessment"]["enum"]) == {"met", "missed", "unresolved"}


def test_next_step_review_revision_and_projection(tmp_path):
    import jsonschema
    from aka.workflow.light.host.journal import _direction_attempt_views
    current, reviewed = state(), []
    value = dict(request(), next_step={'action': 'Preserve input reuse', 'reason': 'Repeated reads remain untested'})
    schema = journal_schema('experiment_record', {}, experiments_enabled=True, directions_enabled=False)
    jsonschema.validate(value, schema)
    result = record_experiment(current, tmp_path, value,
        review_claim=lambda kind, target: reviewed.append(deepcopy(target)) or {})
    assert reviewed[0]['next_step'] == value['next_step']
    eid = result['experiment_id']
    revision = dict(request(), experiment_id=eid)
    record_experiment(current, tmp_path, revision)
    experiment = current['experiments'][0]
    assert experiment['next_step'] == value['next_step']
    store = ResearchStore(tmp_path / 'research')
    store.observe_experiment(experiment)
    assert 'Preserve input reuse' in str(store.board())
    versions = {('directions','d',1): ('',{'name':'Parallelism'},{},set()),
                ('experiments',eid,1): ('',dict(experiment,direction_id='d'),{},set())}
    view = _direction_attempt_views(versions)['research-library/directions/d/attempts.md'].decode()
    assert '| Next action | Reason |' in view
    assert 'Preserve input reuse | Repeated reads remain untested' in view
    revision['next_step'] = {'action':'Test dispatch', 'reason':'Short shapes regressed'}
    record_experiment(current, tmp_path, revision)
    assert current['experiments'][0]['next_step']['action'] == 'Test dispatch'
    revision['next_step'] = None
    jsonschema.validate(revision, schema)
    record_experiment(current, tmp_path, revision)
    assert _public_experiment(current['experiments'][0])['next_step'] is None
    assert store.events()[0]['payload']['experiment']['next_step'] == value['next_step']


@pytest.mark.parametrize('bad', ['do it', {}, {'action':'x'}, {'action':'x','reason':''},
    {'action':[], 'reason':'why'}, {'action':'x','reason':'why','extra':True}])
def test_next_step_validation_before_review(tmp_path, bad):
    current, reviewed = state(), []
    with pytest.raises((AgentRequestError, ValueError)):
        record_experiment(current, tmp_path, dict(request(), next_step=bad),
            review_claim=lambda *args: reviewed.append(args))
    assert current['experiments'] == []
    assert reviewed == []
