"""Historical environment names forwarded to their Workflow implementation."""
from aka.legacy.workflow.constants import CONVERT_PERF_TOL as _constants_CONVERT_PERF_TOL
from aka.legacy.task.candidate_workspace.policy import IMMUTABLE_BASELINE_PATHS as _constants_IMMUTABLE_BASELINE_PATHS
from aka.legacy.task.candidate_workspace.excludes import STALL_STATE_FILE as _constants_STALL_STATE_FILE
import aka.legacy.task.candidate_workspace.state as _workspace_state
import aka.legacy.agent.process as _agent_process
from aka.legacy.workflow.environment import (
    prepare_campaign, link_episode_runtime, episode_directives, run_sandbox,
    candidate_policy_violations, conversion_required, candidate_is_gluon,
    restored_stall, save_stall, latest_version,
)

# Re-exported for the rest of long_horizon, which reaches orchestrator only through this adapter.
run_bounded = _agent_process.run_bounded
peak_util = _workspace_state.peak_util
read_memory = _workspace_state.read_memory
speedup_vs_reference = _workspace_state.speedup_vs_reference
CONVERT_PERF_TOL = _constants_CONVERT_PERF_TOL
IMMUTABLE_BASELINE_PATHS = _constants_IMMUTABLE_BASELINE_PATHS
STALL_STATE_FILE = _constants_STALL_STATE_FILE

from aka.legacy.agent.native import (
    fresh_session_command,
    normalize_stream,
    resume_session_command,
    session_environment,
    session_id_from_stream,
    supports_same_session_resume,
)


def tokens_from_stream(stdout):
    from aka.legacy.agent.runtime import terminal_usage_from_stream

    return terminal_usage_from_stream(stdout).total_tokens or 0
