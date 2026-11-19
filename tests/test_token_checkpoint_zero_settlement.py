import json
from datetime import datetime, timezone

from aka.agent.session_usage import UsageAccumulator
from aka.agent.usage_summary import session_usage
from aka.workflow.light.long_horizon.token_checkpoints import _main_rows


def accumulator(terminal):
    a = UsageAccumulator('claude')
    a.feed('provider/stdout.stream-json', json.dumps({'type':'assistant','message':{'id':'m1','role':'assistant','usage':{'input_tokens':10,'output_tokens':5}}})+'\n')
    a.feed('provider/stdout.stream-json', json.dumps({'type':'result','usage':terminal})+'\n')
    return a


def test_zero_provider_settlement_cannot_erase_observed_usage():
    r = accumulator({'input_tokens':0,'output_tokens':0}).report(finished=True)
    assert r['total']['total_tokens'] == 15
    assert r['total']['measurement'] == 'partial'
    assert r['terminal']['total_tokens'] == 0
    assert 'claude_zero_result_conflicts_with_responses' in r['warnings']


def test_positive_provider_total_keeps_its_scope():
    r = accumulator({'input_tokens':2,'output_tokens':1}).report(finished=True)
    assert r['total']['total_tokens'] == 3
    assert r['accounting_basis'] == 'provider_result_usage'


def test_true_zero_is_preserved():
    assert session_usage({'backend':'claude','state':'completed','total':{'total_tokens':0,'measurement':'exact'},'by_agent':{'main':{'total_tokens':0}}})['total_tokens'] == 0


def test_checkpoint_repairs_legacy_zero_without_adding_children(tmp_path):
    sessions = tmp_path/'e0001'/'sessions'
    run = sessions/'run-a'; run.mkdir(parents=True)
    (run/'token-usage.json').write_text(json.dumps({'backend':'claude','state':'interrupted','started_at':'2026-11-18T00:00:00Z','total':{'total_tokens':0,'measurement':'exact'},'by_agent':{'main':{'total_tokens':44},'child':{'total_tokens':999}}}))
    rows,warnings = _main_rows(sessions,datetime(2026,11,19,tzinfo=timezone.utc))
    assert rows[0]['usage']['total_tokens'] == 44
    assert rows[0]['usage']['measurement'] == 'partial'
    assert any('zero_total_conflicts' in w for w in warnings)
