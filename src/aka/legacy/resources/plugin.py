"""Select the tool catalog within Resources for this composition lifetime."""
from pathlib import Path
name = "resources"
identity_packages = ("aka.legacy.resources",)
Config = {"type":"object", "properties":{"plugin_dir":{"type":"string"},
    "plugins":{"type":"array", "items":{"type":"string", "minLength":1}}}, "additionalProperties":False}
Defaults = {"plugin_dir":"", "plugins":[]}
interpolate = ("plugin_dir", "plugins")

def apply(ctx, config):
    from .plugins import PLUGIN_DIR, _selected_catalog, _selected_plugins
    value = config.get("plugin_dir", "")
    selected = Path(value).expanduser().resolve() if value else PLUGIN_DIR
    if not selected.is_dir():
        raise ValueError("selected tool catalog is not a directory")
    plugins = tuple(Path(value).expanduser().resolve() for value in config.get("plugins", ()))
    if any(not (path / "plugin.json").is_file() for path in plugins):
        raise ValueError("selected tool plugin requires plugin.json")
    token = _selected_catalog.set(selected)
    ctx.effect(lambda: _selected_catalog.reset(token), label="restore tool catalog selection")
    plugin_token = _selected_plugins.set(plugins)
    ctx.effect(lambda: _selected_plugins.reset(plugin_token), label="restore explicit plugin selection")
