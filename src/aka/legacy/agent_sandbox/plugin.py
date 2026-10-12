"""Construct a launch policy without spawning processes or creating Agent Homes."""
from pathlib import Path

from .policy import SandboxPolicy

name = "agent-sandbox"
provide = ("agent_sandbox",)
Defaults = {"mode": "none", "executable": "bwrap", "read_only_paths": []}
Config = {
    "type": "object",
    "properties": {
        "repo_root": {"type": "string", "minLength": 1},
        "mode": {"type": "string", "enum": ["none", "bwrap"]},
        "executable": {"type": "string", "minLength": 1},
        "read_only_paths": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
    "required": ["repo_root"], "additionalProperties": False,
}
interpolate = ("repo_root",)
identity_packages = ("aka.legacy.agent_sandbox", "aka.contracts")


def apply(ctx, config):
    ctx.provide("agent_sandbox", SandboxPolicy(
        Path(config["repo_root"]), config["mode"], config["executable"],
        tuple(config["read_only_paths"]),
    ))
