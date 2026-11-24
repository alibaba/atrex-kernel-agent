"""Long-running sessions archive complete output without retaining its bodies."""
import gc
import hashlib
import io
import json
import tracemalloc

from aka.agent.session_capture import SessionCapture
from aka.agent.session_tail import CaptureLimits
from aka.agent.session_transcript import render_conversation, write_conversation
from aka.agent.session_usage import estimate_usage_from_conversation


class GeneratedPipe:
    def __init__(self, count=1024):
        self.remaining = count
        self.digest = hashlib.sha256()

    def readline(self, limit):
        if not self.remaining:
            return ''
        self.remaining -= 1
        line = json.dumps({'type': 'system', 'subtype': 'thinking_tokens',
                           'n': self.remaining, 'padding': 'x' * 16384}) + '\n'
        self.digest.update(line.encode())
        return line

    def close(self):
        pass


def test_filtered_output_is_spooled_and_complete_not_retained(tmp_path):
    c = SessionCapture(tmp_path, backend='claude', command=['claude', 'prompt'], context={},
                       limits=CaptureLimits(file_bytes=1024, total_bytes=1024))
    pipe = GeneratedPipe()
    tracemalloc.start()
    try:
        c._read_pipe('stdout', pipe)
        c._sync_stdout_usage()
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        assert c.output_sizes()[0] > 16 * 1024 * 1024
        assert current < 1024 * 1024
        assert peak < 2 * 1024 * 1024
    finally:
        tracemalloc.stop()
    # Full functional output survives filtering and the diagnostic byte budget.
    assert hashlib.sha256(c._chunks['stdout'].text().encode()).digest() == pipe.digest.digest()
    c.finish(exit_status=0)
    assert c._chunks['stdout'].file.closed
    assert (c.root / 'conversation.jsonl').stat().st_size < 2000


def test_disk_projection_matches_existing_claude_dedup_and_error_order(tmp_path):
    assistant = {'type': 'assistant', 'message': {'id': 'a', 'content': [{'type': 'text', 'text': '你好'}]}}
    user = {'type': 'user', 'message': {'content': 'prompt'}}
    result = {'type': 'result', 'usage': {'input_tokens': 2, 'output_tokens': 3}}
    native = '\n'.join(map(json.dumps, [user, assistant, result])) + '\n'
    stream = '\n'.join(map(json.dumps, [assistant, {'type': 'error', 'message': 'failure'}, result])) + '\n'
    raw = tmp_path / 'native.jsonl'
    raw.write_text(native)
    kwargs = dict(backend='claude', session_id='s', prompt='prompt', state='failed',
                  exit_status=1, timed_out=False, raw_provider_capture_complete=True,
                  started_at='time', context={}, capture_errors=[])
    expected = render_conversation(stdout=stream, stderr='error\n',
        raw_provider_files=[('provider/claude-session.raw-jsonl', native.encode())], **kwargs)
    destination = tmp_path / 'conversation.jsonl'
    write_conversation(destination=destination, stdout=iter(stream.splitlines(True)), stderr=iter(['error\n']),
        raw_provider_files=[('provider/claude-session.raw-jsonl', raw)], **kwargs)
    assert destination.read_text() == expected


def test_large_final_projection_does_not_materialize_event_bodies(tmp_path):
    def lines():
        for i in range(512):
            yield json.dumps({'type': 'assistant', 'message': {'id': str(i),
                'content': [{'type': 'text', 'text': 'x' * 32768}]}}) + '\n'
    destination = tmp_path / 'conversation.jsonl'
    tracemalloc.start()
    try:
        write_conversation(destination=destination, backend='claude', session_id='s', prompt='p',
            stdout=lines(), stderr=(), raw_provider_files=[], state='completed', exit_status=0,
            timed_out=False, raw_provider_capture_complete=False)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 6 * 1024 * 1024
    finally:
        tracemalloc.stop()
    assert destination.stat().st_size > 16 * 1024 * 1024
    with destination.open() as stream:
        assert sum(1 for _ in stream) == 515  # header, prompt, events, terminal


def test_native_capture_stores_paths_and_incremental_usage(tmp_path, monkeypatch):
    native = tmp_path / 'native.jsonl'
    native.write_text('')
    c = SessionCapture(tmp_path / 'captures', backend='claude', command=['claude', 'p'], context={})
    monkeypatch.setattr(c, '_native_paths', lambda **_: {'provider/native.jsonl': (native, 0)})
    for i in range(2):
        with native.open('a') as f:
            f.write(json.dumps({'type': 'assistant', 'message': {'id': str(i),
                'usage': {'input_tokens': 10, 'output_tokens': 2}, 'content': 'hi'}}) + '\n')
        c.sync_native()
    assert c.native['provider/native.jsonl'].read_bytes() == native.read_bytes()
    c.finish(exit_status=0)
    assert json.loads((c.root / 'token-usage.json').read_text())['total']['total_tokens'] == 24


def test_streaming_estimate_matches_string_input(tmp_path):
    rows = [{'type': 'message', 'source': 'runtime_input', 'content': 'hello中文'},
            {'type': 'provider_event', 'path': 'provider/stdout.stream-json',
             'event': {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': '答复abc'}]}}}]
    text = '\n'.join(map(json.dumps, rows))
    assert estimate_usage_from_conversation(io.StringIO(text)) == estimate_usage_from_conversation(text)


def test_failed_spool_drains_live_pipe_and_marks_capture_incomplete(tmp_path, monkeypatch):
    import errno
    c = SessionCapture(tmp_path, backend='claude', command=['claude', 'p'], context={})
    pipe = GeneratedPipe(count=10)
    def fail(_):
        raise OSError(errno.ENOSPC, 'full disk')
    monkeypatch.setattr(c._chunks['stdout'], 'append', fail)
    c._read_pipe('stdout', pipe)
    assert pipe.remaining == 0
    assert c._output_error.errno == errno.ENOSPC
    assert any('output_spool' in error for error in c.errors)
    c.finish(exit_status=0)
    assert json.loads((c.root / 'token-usage.json').read_text())['capture_errors']
