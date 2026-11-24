"""Codex thread restoration must retain both native identity and native files."""
import json
from pathlib import Path
import uuid

import pytest

from aka.agent.native import NativeSession
from aka.agent.runtime import build_agent_runtime
from aka.agent.execution.home import SESSION_HOME_ROOT_ENV, SESSION_HOME_ID_ENV
from aka.agent.provider import Agent, AgentOptions


def response(thread_id):
    return '\n'.join((json.dumps({'type': 'thread.started', 'thread_id': thread_id}),
        json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 2, 'output_tokens': 1}})))


def test_reopened_codex_uses_supplied_thread_id(tmp_path):
    thread_id = str(uuid.uuid4())
    calls = []
    def execute(command, workspace, timeout, environment):
        calls.append(command)
        return response(thread_id), '', 0, False
    runtime = build_agent_runtime('codex', environment={'HOME': str(tmp_path)})
    session = NativeSession('codex', tmp_path, thread_id, execute, runtime=runtime)
    result = session.invoke('Continue', resume=True)
    assert calls[0][:3] == ['codex', 'exec', 'resume']
    assert calls[0][-2:] == [thread_id, 'Continue']
    assert '--color' not in calls[0]
    assert result.session_id == thread_id


def test_fresh_codex_does_not_advertise_placeholder_as_thread(tmp_path):
    placeholder, actual = str(uuid.uuid4()), str(uuid.uuid4())
    calls = []
    def execute(command, *args):
        calls.append(command)
        return response(actual), '', 0, False
    session = NativeSession('codex', tmp_path, placeholder, execute,
        runtime=build_agent_runtime('codex', environment={'HOME': str(tmp_path)}))
    assert session.session_id == ''
    assert session.invoke('Start').session_id == actual
    session.invoke('Continue', resume=True)
    assert placeholder not in calls[0]
    assert calls[1][-2] == actual


@pytest.mark.parametrize('sandbox', ['none', 'bwrap'])
def test_new_owner_resumes_native_files_in_stable_home(tmp_path, sandbox, monkeypatch):
    # Exercise sandbox Home preparation without launching Linux bwrap on macOS.
    if sandbox == "bwrap":
        monkeypatch.setattr("aka.agent.execution.bubblewrap.executable", lambda _: Path("/usr/bin/true"))
    root = tmp_path.resolve()
    workspace = root / 'workspace'
    workspace.mkdir()
    host = root / 'host'
    host.mkdir()
    home_root = root / 'private-homes'
    actual = str(uuid.uuid4())
    overrides = {SESSION_HOME_ROOT_ENV: str(home_root), SESSION_HOME_ID_ENV: 'optimization-session'}
    environment = {'HOME': str(host), 'PATH': '/usr/bin:/bin'}
    homes = []
    def execute(command, workspace, timeout, values):
        home = Path(values['CODEX_HOME'])
        homes.append(home)
        transcript = home / 'sessions' / f'rollout-{actual}.jsonl'
        if command[:3] == ['codex', 'exec', 'resume']:
            assert command[-2] == actual
            assert transcript.read_text() == 'original native conversation'
        else:
            transcript.parent.mkdir(parents=True)
            transcript.write_text('original native conversation')
        return response(actual), '', 0, False
    # Recreate the owner as on process restart. Only the persisted native ID
    # and trajectory Home binding survive; no transcript copying is performed.
    for resume, identity in ((False, str(uuid.uuid4())), (True, actual)):
        agent = Agent(AgentOptions(sandbox=sandbox), environment=environment)
        agent._execute = execute
        session = agent.open_session('codex', workspace, session_id=identity, environment=overrides)
        assert session.invoke('Continue' if resume else 'Start', resume=resume).session_id == actual
    assert homes[0] == homes[1]


def test_reviewer_codex_can_reopen_with_family_home(tmp_path):
    from aka.workflow.light.long_horizon.review_archive import reviewer_environment
    root = tmp_path.resolve() / 'review-1'
    workspace = root / 'workspace'
    workspace.mkdir(parents=True)
    host = tmp_path.resolve() / 'host'
    host.mkdir()
    actual = str(uuid.uuid4())
    homes = []
    for resume, identity in ((False, str(uuid.uuid4())), (True, actual)):
        agent = Agent(environment={'HOME': str(host), 'PATH': '/usr/bin:/bin'})
        def execute(command, cwd, timeout, env):
            homes.append(env['CODEX_HOME'])
            assert ('resume' in command) == resume
            return response(actual), '', 0, False
        agent._execute = execute
        session = agent.open_session('codex', workspace, session_id=identity,
            environment=reviewer_environment(root))
        assert session.invoke('review', resume=resume).session_id == actual
    assert homes[0] == homes[1]


def test_resume_keeps_known_id_when_backend_fails_before_thread_event(tmp_path):
    actual = str(uuid.uuid4())
    session = NativeSession('codex', tmp_path, actual,
        lambda *args: ('', 'provider unavailable', 1, False),
        runtime=build_agent_runtime('codex', environment={'HOME': str(tmp_path)}))
    result = session.invoke('Continue', resume=True)
    assert result.exit_status == 1
    assert session.session_id == actual
