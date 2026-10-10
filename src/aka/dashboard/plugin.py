"""Provide a local dashboard Startup without starting it during assembly."""
from .startup import DashboardStartup

name = "dashboard"
provide = ("startup",)
Config = {
    "type": "object",
    "properties": {
        "host": {"type": "string", "enum": ["127.0.0.1"]},
        "port": {"type": "integer", "minimum": 0, "maximum": 65535},
        "refresh_ms": {"type": "integer", "minimum": 1},
    },
    "additionalProperties": False,
}
Defaults = {"host": "127.0.0.1", "port": 8765, "refresh_ms": 2000}
identity_files = (
    "__init__.py", "__main__.py", "plugin.py", "startup.py", "reader.py", "server.py", "demo.py",
    "static/index.html", "static/fonts/geist-sans.woff2", "static/fonts/geist-mono.woff2",
    "static/fonts/Geist-LICENSE.txt", "profiles/dashboard.json",
    "profiles/compositions/dashboard.json",
)
identity_packages = ("aka.contracts", "aka.bootstrap")


def apply(ctx, config):
    startup = DashboardStartup(**config)
    ctx.provide("startup", startup)
    ctx.effect(startup.close)
