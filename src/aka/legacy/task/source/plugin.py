"""Select initial kernel or SOL source without creating workspace state."""
from aka.core.keys import ServiceKey

SOURCE = ServiceKey("source", "SourceProvider", module="aka.contracts.workspace")
name = "source"
provide = ("source",)
identity_files = ("plugin.py", "kernel.py", "sol/source.py", "sol/render.py", "legacy.py")
identity_packages = ("aka.contracts.content",)
Config = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["kernel", "sol"]},
    "kernel_demo": {"type": "string"}, "operator_dir": {"type": "string"},
    "name": {"type": "string"}, "framework": {"type": "string"},
    "platform": {"type": "string"}, "gpu_wiki": {"type": "string"},
    "definition": {"type": "object"},
}, "required": [], "additionalProperties": False}


def validate_config(config):
    if not config:
        return
    if "kind" not in config:
        raise ValueError("configured source requires kind")
    fields = ("kernel_demo",) if config["kind"] == "kernel" else ("operator_dir", "name", "framework", "platform")
    if any(not config.get(name) for name in fields):
        raise ValueError("source configuration is missing " + ", ".join(name for name in fields if not config.get(name)))


def apply(ctx, config):
    if not config:
        from .legacy import LegacySourceProvider
        ctx.provide(SOURCE, LegacySourceProvider())
        return
    if config["kind"] == "kernel":
        from .kernel import KernelSourceProvider
        provider = KernelSourceProvider(config["kernel_demo"])
    else:
        from .sol.source import SolSourceProvider
        provider = SolSourceProvider(*(config[name] for name in ("operator_dir", "name", "framework", "platform")),
                                     gpu_wiki=config.get("gpu_wiki", ""), definition=config.get("definition"))
    ctx.provide(SOURCE, provider)

