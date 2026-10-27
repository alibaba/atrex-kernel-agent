"""Workflow-owned Skills, instructions and explicitly selected reference projects."""
from importlib.resources import files
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution
from aka.contracts.content import WorkspaceFile

PROMPTS_DIR = Path(str(files("aka.legacy.workflow").joinpath("prompts")))
DEPENDENCY_REVIEW_PROMPT = PROMPTS_DIR / "dependency_review.md"


def contribute(ctx, config):
    root = Path(str(files('aka.legacy.workflow')))
    descriptions = {
        'gen-plan': 'Generate a reviewed implementation plan from a concrete draft.',
        'gpu-kernel-baseline': 'Implement and validate the initial GPU kernel baseline.',
        'gpu-kernel-episode-loop': 'Run the profile, research, implementation and verification episode loop.',
    }
    for name, description in descriptions.items():
        ctx.contribute('agent_skills', name, AgentSkill(name, root/'skills'/name, description))
    resources = [
        WorkspaceResourceContribution('workflow-baseline-agent', root/'agents/gpu-kernel-baseline.md',
                                      '.claude/agents/gpu-kernel-baseline.md'),
        WorkspaceResourceContribution('workflow-instructions', root/'resources/CLAUDE.md', visibility='runtime'),
        WorkspaceResourceContribution('workflow-iteration-schema', root/'resources/v_iteration.schema.json',
                                      'reference/v_iteration.schema.json'),
        WorkspaceResourceContribution('workflow-workspace-reference', root/'resources/README.md',
                                      'reference/README.md'),
    ]
    for index, source in enumerate(config.get('reference_projects', ())):
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError('selected Workflow reference project must be a directory')
        resources.append(WorkspaceResourceContribution(f'workflow-reference-{index}', source,
                                                       'reference-projects/'+source.name))
    for resource in resources:
        ctx.contribute('workspace_resources', resource.name, resource)


def initial_files(registry, *, sol=False, policy=None):
    installation = registry.resource_result()
    instructions = installation.resource('workflow-instructions').source
    result = [WorkspaceFile('CLAUDE.md', source=instructions) if policy is None
              else WorkspaceFile('CLAUDE.md', text=policy(instructions.read_text(encoding='utf-8')))]
    try:
        driver = installation.resource('execution-profile-driver')
    except KeyError:
        if sol:
            raise ValueError('SOL workspace requires the selected execution-profile-driver resource') from None
    else:
        result.append(WorkspaceFile('profile_driver.py', source=driver.source))
    return tuple(result)


def campaign_initial_files(campaign, *, sol=False):
    from .policy import workspace_policy_content
    return initial_files(campaign.plugin_registry, sol=sol, policy=lambda current:
        workspace_policy_content(current, campaign.optimization_mode, campaign.framework,
                                 source_checks_enabled=campaign.source_checks_enabled))


def profile_driver_files(registry):
    driver = registry.resource_result().resource('execution-profile-driver')
    return (WorkspaceFile('profile_driver.py', source=driver.source),)


def campaign_context_directories(campaign):
    """Declare the Legacy Task runtime projection, without inferring file ownership."""
    if campaign.candidate_workspace.current_revision(campaign.workspace):
        return ()
    return ("atrex-bench",) if campaign.problem.evaluation.runtime_root is not None else ()
