"""Response accounting must not shrink when a provider settles a final turn."""
import json
from datetime import datetime, timezone

import pytest

from aka.agent.adapter import ClaudeAdapter
from aka.agent.session_usage import UsageAccumulator
from aka.agent.usage_summary import select_session_usage
from aka.workflow.light.long_horizon.token_checkpoints import _main_rows


def message(identity, count):
    return json.dumps({'type': 'assistant', 'message': {
        'id': identity, 'role': 'assistant',
        'usage': {'input_tokens': count, 'output_tokens': 5},
    }}) + '\n'


@pytest.mark.parametrize('terminal', [0, 3, 1000000])
def test_terminal_cannot_replace_responses(terminal):
    a = UsageAccumulator('claude')
    a.feed('provider/stdout.stream-json', message('one', 100))
    before = a.report()['total']['total_tokens']
    a.feed('provider/stdout.stream-json', json.dumps({
        'type': 'result', 'usage': {'input_tokens': terminal, 'output_tokens': 0},
    }) + '\n')
    report = a.report(finished=True)
    assert report['total']['total_tokens'] == before == 105
    assert report['terminal']['total_tokens'] == terminal
    assert report['accounting_basis'] == 'main_response_usage'
    assert select_session_usage(report)['usage'] == report['total']


def test_native_stream_duplicates_and_resume_history_count_once():
    a = UsageAccumulator('claude')
    a.feed('native/main.jsonl', message('old', 1000), previous=True)
    a.feed('provider/stdout.stream-json', message('old', 1000) + message('new', 100))
    a.feed('native/main.jsonl', message('new', 100))
    assert a.report(finished=True)['total']['total_tokens'] == 105


def test_missing_responses_do_not_use_provider_settlement():
    a = UsageAccumulator('claude')
    a.feed('provider/stdout.stream-json', '{"type":"result","usage":{"input_tokens":10,"output_tokens":5}}\n')
    assert a.report(finished=True)['total']['total_tokens'] is None


def test_adapter_without_capture_uses_responses():
    stream = message('one', 100) + message('two', 200)
    stream += '{"type":"result","usage":{"input_tokens":3,"output_tokens":0}}\n'
    events, total = ClaudeAdapter().normalize_stream(stream)
    assert total.total_tokens == 310
    assert [e.usage.total_tokens for e in events if e.kind == 'terminal_usage'] == [310]


def test_checkpoint_counts_completed_and_live_responses(tmp_path):
    sessions = tmp_path / 'e0002/sessions'
    for episode, state, main, terminal in [('e0001', 'completed', 52345967, 704848),
                                            ('e0002', 'running', 31919116, 0)]:
        path = tmp_path / episode / 'sessions/run-a/token-usage.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'backend': 'claude', 'state': state,
            'total': {'total_tokens': terminal}, 'accounting_basis': 'provider_result_usage',
            'by_agent': {'main': {'total_tokens': main}, 'child': {'total_tokens': 999}}}))
    rows, _ = _main_rows(sessions, datetime.now(timezone.utc))
    assert sum(r['usage']['total_tokens'] for r in rows) == 84265083
    assert all(r['accounting_basis'] == 'main_response_usage' for r in rows)


@pytest.mark.parametrize('purpose', ['main', 'claim-review', 'submission-review'])
def test_capture_and_result_keep_response_total(tmp_path, purpose):
    from aka.agent.session_capture import SessionCapture, captured_observation, _LAST_CAPTURE
    stream = message('first', 100) + message('second', 200)
    stream += '{"type":"result","usage":{"input_tokens":3,"output_tokens":0}}\n'
    capture = SessionCapture(tmp_path / purpose, backend='claude', command=['claude', 'test'],
                             context={}, native_environment={'HOME': str(tmp_path)})
    capture._chunks['stdout'].append(stream)
    capture._retained_chunks['stdout'].append(stream)
    capture.finish(exit_status=0)
    try:
        report = json.loads((capture.root / 'token-usage.json').read_text())
        assert report['total']['total_tokens'] == 310
        assert report['terminal']['total_tokens'] == 3
        adapter = ClaudeAdapter()
        events, _ = adapter.normalize_stream(stream)
        result = captured_observation(stream, events, adapter.capabilities)
        assert result.terminal_usage.total_tokens == 310
    finally:
        _LAST_CAPTURE.set(None)
