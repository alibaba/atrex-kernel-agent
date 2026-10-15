"""Legacy campaign construction and application argument conversion."""

from dataclasses import fields
from pathlib import Path
from aka.contracts.workflow import CampaignRequest, WorkflowSettings
from aka.contracts.problem import LoadedProblem
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
    values.pop("kernel_demo", "")
    values.pop("atrex_bench_root", "")
    from .cli_problem import constructor_problem_request
    from aka.contracts.workflow import ProblemSource

    supplied = values.pop("problem", None)
    if isinstance(supplied, LoadedProblem):
        problem = ProblemSource(supplied.directory, None)
    else:
        problem = supplied or constructor_problem_request(configuration)
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
