"""Workflow inputs consume explicitly selected common Resources."""
from aka.resources.bundles import contribute_reference_projects, resource_path
from aka.contracts.content import WorkspaceFile

PROMPTS_DIR = resource_path("prompts/kernel-development")
DEPENDENCY_REVIEW_PROMPT = PROMPTS_DIR / "dependency_review.md"


def contribute(ctx, config):
    contribute_reference_projects(ctx, config.get("reference_projects", ()))


def initial_files(registry, *, sol=False, policy=None):
    installation = registry.resource_result()
    instructions = installation.resource('workflow-instructions').source
    text = instructions.read_text(encoding="utf-8")
    if policy is not None:
        text = policy(text)
    result = [WorkspaceFile("CLAUDE.md", text=text)]
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
