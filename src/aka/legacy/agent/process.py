"""Legacy recovery ownership adapter for shared Agent process execution."""
from aka.agent.execution.process import *
from aka.agent.execution.process import run_bounded as _run_bounded
from aka.execution.processes import spawn_owned_session

from aka.legacy.workflow.agent_policy import (protected_gateway_identity, dependency_process_violation,
    python_import_roots, DEFAULT_PROTECTED_GATEWAY_SCREEN, DEFAULT_PROTECTED_GATEWAY_STATE_NAME)

def run_bounded(command, cwd, timeout, env=None):
    from aka.legacy.workflow.agent_host import unavailable, violation
    return _run_bounded(command, cwd, timeout, env, spawn=spawn_owned_session,
                        unavailable=unavailable, violation=violation)
