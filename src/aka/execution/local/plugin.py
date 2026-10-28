"""Select local command execution explicitly without replacing GPU providers."""
from .executor import LocalCommandExecutor

name = "local"
provide = ("executor",)
identity_packages = ("aka.execution.local", "aka.execution.filesystem_scope", "aka.contracts.execution", "aka.contracts.filesystem", "aka.contracts.filesystem_access")
Config = {"type": "object", "properties": {"bwrap": {"type": "string"}}, "additionalProperties": False}
Defaults = {"bwrap": "bwrap"}


def apply(ctx, config):
    ctx.provide("executor", LocalCommandExecutor(bwrap=config["bwrap"]))
