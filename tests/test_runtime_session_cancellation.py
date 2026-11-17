"""Cancellation must stop the owned process, not merely revoke its tools."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
import os
from pathlib import Path
import signal
import sys
import tempfile
from threading import Event, RLock
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aka.agent.execution.process import spawn_session
from aka.agent.provider import Agent
from aka.agent.tools.client import call_tool, ToolSessionError
from aka.agent.tools.host import ToolHost
from aka.contracts.agent import AgentToolBinding
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.workflow.light.orchestration.supervisor_runtime import Capability, SupervisorRuntime
from aka.workflow.light.search_driver import EpisodeSearchDriver
from aka.workflow.light.support.errors import SessionRevokedError


def access_to(workspace):
    return FilesystemAccess((FilesystemMount(workspace, workspace, True),))


def runtime_for(agent):
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.agent = agent
    runtime.lock = RLock()
    runtime.closed = False
    runtime.capabilities = {}
    runtime.journals = {}
    runtime.config = SimpleNamespace(agate_tools=("env", "evaluate", "command"))
    runtime.temporary = tempfile.TemporaryDirectory()
    runtime._wiki_execution = SimpleNamespace(dispose=Mock())
    return runtime


def wait_until(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Agent did not reach the expected process state")


@pytest.mark.parametrize("action", ["close", "revoke", "interrupt"])
def test_runtime_cancellation_stops_real_agent_without_stopping_other_campaign(tmp_path, action):
    root = tmp_path.resolve()
    workspaces = [root / name for name in ("campaign-a", "campaign-b")]
    for workspace in workspaces:
        workspace.mkdir()
        (workspace / "kernel.py").write_text("candidate = 1\n")
    processes = {}

    def spawn(_command, **kwargs):
        # Exercise real native process ownership without a provider or GPU call.
        process = spawn_session([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes[Path(kwargs["cwd"])] = process
        return process

    agent = Agent(spawn=spawn, environment={"HOME": str(root), "PATH": os.defpath})
    runtimes = [runtime_for(agent) for _ in workspaces]
    with ExitStack() as cleanup:
        pool = cleanup.enter_context(ThreadPoolExecutor(max_workers=2))
        cleanup.callback(agent.close)
        for runtime in runtimes:
            cleanup.callback(runtime.close)
        sessions, futures = [], []
        for workspace, runtime in zip(workspaces, runtimes):
            grant = cleanup.enter_context(runtime.session(workspace, {}, access=access_to(workspace)))
            session = agent.open_session("claude", workspace,
                environment={**grant.environment, "ATREX_SESSION_CAPTURE": "0", "HOME": str(root)},
                access=grant.filesystem_access)
            sessions.append(session)
            futures.append(pool.submit(session.invoke, "wait for cancellation"))
        wait_until(lambda: len(processes) == 2 and all(
            context.control is not None and context.control.scope is not None
            for context in agent._invocations.values()))

        if action == "interrupt":
            driver = EpisodeSearchDriver.__new__(EpisodeSearchDriver)
            driver.cancel_event = Event()
            driver.children = {1: SimpleNamespace(close_runtime=runtimes[0].close)}
            driver.interrupt()
            assert driver.cancel_event.is_set()
        elif action == "revoke":
            runtimes[0].revoke(next(iter(runtimes[0].capabilities)))
        else:
            runtimes[0].close()

        stopped = futures[0].result(timeout=10)
        assert stopped.exit_status == -signal.SIGKILL
        assert not stopped.timed_out
        assert processes[workspaces[0]].poll() == -signal.SIGKILL
        assert processes[workspaces[1]].poll() is None
        assert not futures[1].done()
        assert (workspaces[0] / "kernel.py").read_text() == "candidate = 1\n"
        # The revoked grant cannot start another invocation through resume.
        with pytest.raises(PermissionError, match="session is unavailable"):
            sessions[0].invoke("must not restart", resume=True)
        runtimes[1].close()
        assert futures[1].result(timeout=10).exit_status == -signal.SIGKILL


def test_grant_cancellation_cancels_pending_operation_and_only_its_controller(tmp_path):
    workspace = tmp_path.resolve()
    entered, cancelled = Event(), Event()

    class Handler:
        def handle(self, payload, context):
            if payload == b"wait":
                entered.set()
                if context.cancel_event.wait(10):
                    cancelled.set()
            return payload

    host = ToolHost()
    with ExitStack() as cleanup:
        pool = cleanup.enter_context(ThreadPoolExecutor(max_workers=1))
        cleanup.callback(host.close)
        grants = [cleanup.enter_context(host.session(workspace, access=access_to(workspace),
            bindings=(AgentToolBinding("test.probe", Handler()),))) for _ in range(2)]
        controls = [SimpleNamespace(request=nullcontext, close=Mock(), kill=Mock()) for _ in grants]
        for grant, control in zip(grants, controls):
            cleanup.enter_context(host.bind_control(workspace, grant.environment, control))
        pending = pool.submit(call_tool, "test.probe", b"wait", workspace=workspace,
            environment=grants[0].environment)
        assert entered.wait(5)
        grants[0].cancel()
        assert cancelled.wait(5)
        controls[0].kill.assert_called_once()
        controls[1].kill.assert_not_called()
        # A reply may have raced with cancellation; either way the call settles.
        try:
            pending.result(timeout=5)
        except ToolSessionError:
            pass
        with pytest.raises(ToolSessionError):
            call_tool("test.probe", b"late", workspace=workspace, environment=grants[0].environment)
        assert call_tool("test.probe", b"peer", workspace=workspace,
            environment=grants[1].environment) == b"peer"
        grants[0].cancel()
        controls[0].kill.assert_called_once()


def test_runtime_closed_during_owner_grant_creation_cannot_yield_live_session(tmp_path):
    workspace = tmp_path.resolve()
    agent = Mock()
    runtime = runtime_for(agent)
    grant = SimpleNamespace(cancel=Mock(), environment={}, filesystem_access=access_to(workspace))

    @contextmanager
    def tool_session(*args, **kwargs):
        runtime.close()
        yield grant

    agent.tool_session = tool_session
    with pytest.raises(SessionRevokedError, match="while binding"):
        with runtime.session(workspace, {}, access=access_to(workspace)):
            pytest.fail("A closed Runtime exposed an Agent grant")
    grant.cancel.assert_called_once()
    assert runtime.capabilities == {}


def test_close_cancels_every_grant_even_if_one_cleanup_fails(tmp_path):
    runtime = runtime_for(Mock())
    grants = [SimpleNamespace(cancel=Mock()),
        SimpleNamespace(cancel=Mock(side_effect=OSError("cannot stop one process")))]
    for index, grant in enumerate(grants):
        runtime.capabilities[str(index)] = Capability(tmp_path, {}, agent_session=grant)
    temporary = Path(runtime.temporary.name)
    with pytest.raises(OSError, match="cannot stop"):
        runtime.close()
    for grant in grants:
        grant.cancel.assert_called_once()
    assert not temporary.exists()
    runtime._wiki_execution.dispose.assert_called_once()
    runtime.close()  # Cancellation remains idempotent.


def test_search_interrupt_attempts_every_lineage_on_cleanup_error():
    driver = EpisodeSearchDriver.__new__(EpisodeSearchDriver)
    driver.cancel_event = Event()
    children = [SimpleNamespace(close_runtime=Mock()),
        SimpleNamespace(close_runtime=Mock(side_effect=OSError("cleanup failed")))]
    driver.children = dict(enumerate(children))
    with pytest.raises(OSError, match="cleanup failed"):
        driver.interrupt()
    assert driver.cancel_event.is_set()
    for child in children:
        child.close_runtime.assert_called_once()
