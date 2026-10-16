"""Legacy campaign construction and application argument conversion."""

from dataclasses import fields
from pathlib import Path
from aka.contracts.workflow import CampaignRequest, WorkflowSettings
from aka.contracts.workflow import CampaignRuntimeSettings as EvaluationSettings


def campaign_request(configuration):
    values = dict(configuration)
    name = values.pop("name")
    suffix = values.pop("workspace_suffix", "")
    campaign_name = name + ("_" + suffix if suffix else "")
    work_dir = values.pop("work_dir", "")
    workspace = (Path(work_dir) if work_dir else Path.cwd()) / (
        "kernel_opt_" + campaign_name
    )
    from aka.contracts.workflow import ProblemSource
    problem = values.pop("problem")
    if not isinstance(problem, ProblemSource):
        raise TypeError("campaign CLI requires an explicit ProblemSource")
    for policy, prefix in (
        ("framework_baseline_reviewers", "v1"),
        ("fast_episode_reviewers", "fast_episode"),
        ("full_episode_reviewers", "full_episode"),
    ):
        enabled = []
        for backend in ("codex", "qoder"):
            if values.pop(prefix + "_ask_" + backend, False):
                enabled.append(backend)
        if enabled:
            values[policy] = tuple(enabled)
    settings = {
        f.name: values.pop(f.name) for f in fields(WorkflowSettings) if f.name in values
    }
    evaluation = {}
    for f in fields(EvaluationSettings):
        old = (
            "sandbox_" + f.name
            if f.name
            in {
                "hardware",
                "profile",
                "url",
                "timeout",
                "ssh",
                "ssh_init",
                "ssh_gpu",
                "health_command",
            }
            else f.name
        )
        if old in values:
            evaluation[f.name] = values.pop(old)
    if values:
        raise TypeError(
            "unexpected campaign configuration: " + ", ".join(sorted(values))
        )
    return CampaignRequest(
        problem, workspace, campaign_name, WorkflowSettings(**settings),
        EvaluationSettings(**evaluation)
    )
