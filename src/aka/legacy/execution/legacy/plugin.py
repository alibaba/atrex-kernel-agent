"""Register the unchanged host runner as the legacy execution provider."""
from .executor import LegacyExecutor

name = "legacy"
provide = ("executor",)
identity_files = ("plugin.py", "executor.py", "runtime.py")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    ctx.provide("executor", LegacyExecutor())
