"""Workflow runtime contributions, combined with the selected owners' path declarations."""
from aka.contracts.workspace import WorkspaceExclusion

RUNTIME_DIR = '.atrex_long_horizon'
VERIFY_DIR = 'verification_artifacts/.atrex_long_horizon_verify'
LIVE_MEMORY_FILE = 'memory/live.json'
STALL_STATE_FILE = '.orchestrator_state.json'

WORKFLOW_EXCLUSIONS = (
    WorkspaceExclusion(RUNTIME_DIR, subtree=True),
    WorkspaceExclusion(VERIFY_DIR, subtree=True),
    WorkspaceExclusion(STALL_STATE_FILE),
    WorkspaceExclusion('framework_baseline_resume.json'),
    WorkspaceExclusion(LIVE_MEMORY_FILE),
)
EPISODE_EXCLUSIONS = tuple(WorkspaceExclusion(path, subtree=True)
                           for path in ('plans', 'profiles', '.humanize'))


def early_exclusions():
    from aka.resources.runtime_paths import RUNTIME_EXCLUSIONS as resources
    from aka.agent.runtime_paths import RUNTIME_EXCLUSIONS as agent
    from aka.evaluation.runtime_paths import RUNTIME_EXCLUSIONS as evaluation
    return (*resources, *agent, *evaluation)


def full_exclusions():
    return (*WORKFLOW_EXCLUSIONS, *early_exclusions(), *EPISODE_EXCLUSIONS)
