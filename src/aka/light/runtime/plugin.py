"""Select the private measurement/Journal service independently of Agent isolation."""
from aka.core.keys import ServiceKey

TOOL_RUNTIME = ServiceKey("tool_runtime", "ToolRuntimeFactory", module="aka.light.runtime.provider")
name = "runtime"
provide = ("tool_runtime",)
inject = ("executor", "light_experiments", "light_acceptance", "light_journal", "wiki")
identity_packages = ("aka.light.runtime", "aka.light.evaluation", "aka.light.workspace", "aka.light.wiki",
                     "aka.contracts.wiki")
Config = {"type": "object", "properties": {
    "measurement_repetitions": {"type": "integer", "enum": [1, 3]},
    "wiki": {"type": "boolean"},
    "wiki_options": {"type": "object", "properties": {
        "model": {"type": "string"}, "effort": {"type": "string", "enum": ["low", "medium", "high", "max"]},
        "settings_file": {"type": "string"}, "base_url": {"type": "string"},
        "timeout": {"type": "number", "minimum": 0.001},
        "bridge_timeout": {"type": "integer", "minimum": 1},
        "queue_timeout": {"type": "number", "minimum": 0.001},
        "concurrency": {"type": "integer", "minimum": 1, "maximum": 16},
    }, "additionalProperties": False},
    "request_timeout": {"type": "number", "minimum": 0.001},
    "queue_timeout": {"type": "number", "minimum": 0.001},
}, "additionalProperties": False}
Defaults = {"measurement_repetitions": 1, "wiki": True, "queue_timeout": 60}


def apply(ctx, config):
    from .provider import ManagedRuntimeFactory
    from .services import RuntimeServices
    if not callable(getattr(ctx.executor, "run_gateway", None)):
        raise TypeError("light Tool Runtime requires an executor with run_gateway support")
    services = RuntimeServices(ctx.executor, ctx.light_experiments,
                               ctx.light_acceptance, ctx.light_journal)
    ctx.provide(TOOL_RUNTIME, ManagedRuntimeFactory(**config, services=services, wiki_provider=ctx.wiki))
