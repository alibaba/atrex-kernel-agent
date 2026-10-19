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


def initial_files(registry, *, sol=False):
    installation = registry.resource_result()
    result = [WorkspaceFile('CLAUDE.md', source=installation.resource('workflow-instructions').source)]
    try:
        driver = installation.resource('execution-profile-driver')
    except KeyError:
        if sol:
            raise ValueError('SOL workspace requires the selected execution-profile-driver resource') from None
    else:
        result.append(WorkspaceFile('profile_driver.py', source=driver.source))
    return tuple(result)
