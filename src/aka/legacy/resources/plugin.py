"""Select the tool catalog within Resources for this composition lifetime."""
from pathlib import Path
name = "resources"
identity_packages = ("aka.legacy.resources",)
Config = {"type":"object", "properties":{"plugin_dir":{"type":"string"}}, "additionalProperties":False}
Defaults = {"plugin_dir":""}
interpolate = ("plugin_dir",)

def apply(ctx, config):
    from .plugins import _selected_catalog
    value = config.get("plugin_dir", "")
    selected = Path(value).expanduser().resolve() if value else None
    if selected is not None and not selected.is_dir():
        raise ValueError("selected tool catalog is not a directory")
    token = _selected_catalog.set(selected)
    ctx.effect(lambda: _selected_catalog.reset(token), label="restore tool catalog selection")
