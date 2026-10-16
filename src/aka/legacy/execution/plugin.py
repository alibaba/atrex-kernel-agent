"""Own the selected Execution configuration for a Workflow composition."""
from pathlib import Path

name = "execution"
identity_packages = ("aka.legacy.execution",)
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
    scope.__enter__()
    ctx.effect(lambda: scope.__exit__(None, None, None), label="close Execution selection")
