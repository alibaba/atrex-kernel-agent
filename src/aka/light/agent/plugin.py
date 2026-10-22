"""Replace the existing Agent service, never the startup or Workflow service."""
from aka.legacy.agent.plugin import AGENT

name = "agent"
provide = ("agent",)
identity_packages = ("aka.light.agent", "aka.light.workspace", "aka.contracts.agent", "aka.legacy.agent")
Config = {"type": "object", "properties": {
    "sandbox": {"type": "string", "enum": ["none", "bwrap"]},
    "capture": {"type": "boolean"},
    "timeout_seconds": {"type": "number", "minimum": 0},
    "bwrap_executable": {"type": "string", "minLength": 1},
    "proc_mode": {"type": "string", "enum": ["private", "projected"]},
    "read_only_paths": {"type": "array", "items": {"type": "string", "minLength": 1}},
}, "additionalProperties": False}
Defaults = {"sandbox": "none", "capture": False, "bwrap_executable": "bwrap", "read_only_paths": [], "proc_mode": "private", "timeout_seconds": 28800}


def validate_config(config):
    from pathlib import Path
    from aka.light.agent.sandbox import sandbox_executable
    from .provider import AgentOptions
    for value in config["read_only_paths"]:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.exists():
            raise ValueError("Agent read_only_paths must name existing absolute paths")
    # Explicit isolation fails before a Campaign is launched, including on macOS.
    sandbox_executable(AgentOptions(**{**config, "read_only_paths": tuple(config["read_only_paths"])}).environment())


def apply(ctx, config):
    from .provider import AgentOptions, LightAgent
    agent = LightAgent(AgentOptions(**{**config, "read_only_paths": tuple(config["read_only_paths"])}))
    ctx.effect(agent.close, label="close optional Agent")
    ctx.provide(AGENT, agent)
