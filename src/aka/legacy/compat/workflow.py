"""Legacy campaign construction and application argument conversion."""

from dataclasses import fields
from pathlib import Path
from aka.contracts.workflow import CampaignRequest, WorkflowSettings
from aka.contracts.problem import LoadedProblem
from .configuration import EvaluationSettings, evaluation_settings


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
    from .problem import constructor_problem_request
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


def evaluation_configuration(configuration):
    values = {}
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
        if old in configuration:
            values[f.name] = configuration[old]
    return EvaluationSettings(**values)


def create_campaign(workflow, **configuration):
    configuration.pop("task_dependencies", None)
    settings = evaluation_configuration(configuration)
    with evaluation_settings(settings):
        request = campaign_request(configuration)
        supplied = configuration.get("problem")
        if isinstance(supplied, LoadedProblem):
            # Historical prepared-input callers stay outside the clean Interface.
            request = workflow._dependencies.prepare_request(request, supplied)
            return workflow._campaign(request, supplied)
        return workflow.campaign(request)


class BoundEvaluation:
    """Confine the old Campaign callback convention to the external adapter."""

    def __init__(self, implementation, context):
        self.implementation = implementation
        self.context = context

    def __getattr__(self, name):
        import aka.legacy.compat.legacy_evaluation.results as results

        if hasattr(results, "_" + name):
            return getattr(results, "_" + name)
        operation = getattr(self.implementation, name)
        return lambda *args, **kwargs: operation(self.context, *args, **kwargs)


class EvaluationContext:
    def __init__(self, campaign, implementation, settings=None, base=None):
        from .configuration import current_settings

        self.campaign = campaign
        self.implementation = implementation
        self.settings = settings or current_settings()
        self.base = base
        if (
            sum(
                bool(v)
                for v in (self.settings.ssh, self.settings.url, self.settings.profile)
            )
            > 1
        ):
            raise ValueError(
                "sandbox_ssh, sandbox_url, and sandbox_profile are mutually exclusive"
            )

    def __getattr__(self, name):
        if name == "base_campaign" and self.base is not None:
            return self.base
        key = name.removeprefix("sandbox_") if name.startswith("sandbox_") else name
        if hasattr(self.settings, key):
            return getattr(self.settings, key)
        if name in self.campaign.__dict__:
            return self.campaign.__dict__[name]
        if name == "_print_v0_evaluator_output":
            return self.implementation.print_v0_evaluator_output
        if name.startswith("_") and hasattr(self.implementation, name[1:]):
            operation = getattr(self.implementation, name[1:])
            return lambda *args, **kwargs: operation(self, *args, **kwargs)
        if name == "task_dependencies":
            if "_task_dependencies" in self.campaign.__dict__:
                return self.campaign._task_dependencies
            return dict(
                problem=self.campaign.problem_provider,
                source=self.campaign.source_provider,
                workspace=self.campaign.candidate_workspace,
            )
        return getattr(self.campaign, name)


def bind_evaluation(implementation, campaign, *, settings=None, base=None):
    return BoundEvaluation(
        implementation, EvaluationContext(campaign, implementation, settings, base)
    )
