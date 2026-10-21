name = "execution"
provide = ("executor",)
identity_packages = ("aka.light.execution", "aka.light.supervisor", "aka.light.orchestrator")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from .provider import ReliableExecutor
    ctx.provide("executor", ReliableExecutor())
