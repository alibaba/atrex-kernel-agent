"""Register existing workspace and candidate mechanics."""
from aka.core.keys import ServiceKey

CANDIDATE = ServiceKey("candidate", "CandidateWorkspace", module="aka.contracts.workspace")
name = "candidate-workspace"
provide = ("candidate",)
identity_files = ("plugin.py", "provider.py", "legacy.py", "snapshot.py", "policy.py", "baseline.py", "excludes.py", "runtime.py", "initial.py", "content.py", "init.sh")
identity_packages = ("aka.legacy.task.git", "aka.legacy.task.io", "aka.contracts.content")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from .provider import GitCandidateWorkspace
    ctx.provide(CANDIDATE, GitCandidateWorkspace())

