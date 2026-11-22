"""Register existing workspace and candidate mechanics."""
from aka.core.keys import ServiceKey

CANDIDATE = ServiceKey("candidate", "CandidateWorkspace", module="aka.contracts.workspace", protocol_version=8)
name = "candidate-workspace"
provide = ("candidate",)
identity_packages = ("aka.task.candidate_workspace", "aka.task.git", "aka.task.io", "aka.task.source", "aka.contracts.content", "aka.contracts.workspace", "aka.contracts.filesystem")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from .provider import GitCandidateWorkspace
    ctx.provide(CANDIDATE, GitCandidateWorkspace())

