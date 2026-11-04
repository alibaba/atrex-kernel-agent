"""Own the selected Execution configuration for a Workflow composition."""
from pathlib import Path
from aka.core.keys import ServiceKey
from .assembly import EXECUTOR

EXECUTION_CONTEXT = ServiceKey("execution_context", "ExecutionContext", module="aka.contracts.execution")
MEASUREMENT_PARALLELISM = ServiceKey("measurement_parallelism", "int", module="builtins")

name = "execution"
provide = ("executor", "execution_context", "measurement_parallelism")
identity_packages = ("aka.execution",)
Config = {"type": "object", "properties": {
    "profile": {"type": "string"},
    "profiles_dir": {"type": "string"},
    "patches": {"type": "array", "items": {"type": "object"}},
}, "required": ["profile"], "additionalProperties": False}
interpolate = ("profiles_dir",)


def apply(ctx, config):
    import json
    import os
    from .selection import SELECTION_ENV, execution_scope

    selected = dict(config)
    if "profiles_dir" in selected:
        selected["profiles_dir"] = str(Path(selected["profiles_dir"]).expanduser().resolve())
    inherited = os.environ.get(SELECTION_ENV)
    if inherited and json.loads(inherited) != selected:
        raise ValueError("inherited Execution selection conflicts with its module configuration")
    scope = execution_scope(selection=selected)
    executor = scope.__enter__()
    ctx.effect(lambda: scope.__exit__(None, None, None), label="close Execution selection")
    from aka.contracts.execution import ExecutionContext
    from .selection import execution_environment
    from .configuration import selected_ssh
    ctx.provide(EXECUTOR, executor)
    ctx.provide(EXECUTION_CONTEXT, ExecutionContext(execution_environment(os.environ)))
    ctx.provide(MEASUREMENT_PARALLELISM, 1 if selected_ssh() else 4)
    from .selection import selected_agent_skills, selected_workspace_resources
    for kind, values in (("workspace_resources", selected_workspace_resources()),
                         ("agent_skills", selected_agent_skills())):
        for value in values:
            ctx.contribute(kind, value.name, value)
