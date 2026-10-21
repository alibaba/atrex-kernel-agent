from aka.legacy.task.candidate_workspace.plugin import CANDIDATE

name = "workspace"
provide = ("candidate",)
identity_packages = ("aka.light.workspace", "aka.legacy.task", "aka.legacy.execution", "aka.legacy.workflow")
Config = {"type": "object", "properties": {
    "inherit_state": {"type": "boolean"},
}, "additionalProperties": False}
Defaults = {"inherit_state": True}


def apply(ctx, config):
    from .provider import ManagedCandidateWorkspace
    ctx.provide(CANDIDATE, ManagedCandidateWorkspace(**config))
