"""Select the Wiki knowledge provider used by the Light Tool Runtime."""
from pathlib import Path

from aka.core.keys import ServiceKey

WIKI = ServiceKey("wiki", "WikiProvider", module="aka.contracts.wiki")
name = "wiki"
provide = ("wiki",)
identity_packages = ("aka.light.wiki", "aka.contracts.wiki")
Config = {"type": "object", "properties": {
    "root": {"type": "string"},
}, "additionalProperties": False}
Defaults = {"root": ""}
interpolate = ("root",)


def apply(ctx, config):
    from .local import LocalWikiProvider
    root = config["root"].strip()
    ctx.provide(WIKI, LocalWikiProvider(Path(root) if root else None))
