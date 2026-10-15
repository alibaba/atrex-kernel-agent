"""Deprecated composition name for Workflow's internal startup entry.

Old profile files can retain this row while migrating to workflow.startup=true.
Inputs must already be canonical before Bootstrap, just as in the normal profile.
"""
from .processes import invocation_environment, use_launch_environment

name = "application"
provide = ("startup",)
inject = ("workflow",)
Config = {"type": "object", "properties": {"repo_root": {"type": "string", "minLength": 1}},
          "required": ["repo_root"], "additionalProperties": False}
interpolate = ("repo_root",)
identity_packages = ("aka.legacy.workflow", "aka.legacy.application", "aka.legacy.execution", "aka.contracts", "aka.bootstrap")


def apply(ctx, config):
    from aka.legacy.workflow.startup import WorkflowStartup
    ctx.provide("startup", WorkflowStartup(ctx.workflow))
