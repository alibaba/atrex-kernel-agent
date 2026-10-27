from aka.legacy.task.candidate_workspace.plugin import CANDIDATE

name = "workspace"
provide = ("candidate",)
identity_packages = ("aka.light.workspace", "aka.legacy.task", "aka.legacy.execution", "aka.legacy.workflow")
Config = {"type": "object", "properties": {
    "source_tree": {"type": "object", "properties": {
        "manifest": {"type": "string", "minLength": 1},
        "repository": {"type": "string", "minLength": 1},
        "directory": {"type": "string", "minLength": 1},
    }, "required": ["manifest"], "additionalProperties": False},
    "inherit_state": {"type": "boolean"},
    "file_system": {"type": "string", "enum": ["single_file", "source_tree", "cuda_manifest"]},
}, "additionalProperties": False}
Defaults = {"inherit_state": True}


def apply(ctx, config):
    from .provider import ManagedCandidateWorkspace
    ctx.provide(CANDIDATE, ManagedCandidateWorkspace(**config))
