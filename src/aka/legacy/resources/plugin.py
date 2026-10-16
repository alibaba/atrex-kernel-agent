"""Select explicit tool roots and borrow module contributions for this lifetime."""
from pathlib import Path

name = "resources"
identity_packages = ("aka.legacy.resources",)
Config = {"type": "object", "properties": {
    "plugin_dir": {"type": "string"},
    "plugins": {"type": "array", "items": {"type": "string", "minLength": 1}},
}, "additionalProperties": False}
Defaults = {"plugin_dir": "", "plugins": []}
interpolate = ("plugin_dir", "plugins")


def apply(ctx, config):
    from .contributions import _selected_scope
    from .plugins import _selected_catalog, _selected_plugins
    value = config.get("plugin_dir", "")
    catalog = Path(value).expanduser().resolve() if value else None
    if catalog is not None and not catalog.is_dir():
        raise ValueError("selected tool catalog is not a directory")
    plugins = tuple(Path(value).expanduser().resolve() for value in config.get("plugins", ()))
    if any(not (path / "plugin.json").is_file() for path in plugins):
        raise ValueError("selected tool plugin requires plugin.json")
    for variable, value in ((_selected_catalog, catalog), (_selected_plugins, plugins),
                            (_selected_scope, ctx.current_scope)):
        token = variable.set(value)
        ctx.effect(lambda variable=variable, token=token: variable.reset(token), label="restore resource selection")
