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


def _options(config):
    from .provider import AgentOptions
    values = dict(config)
    values["read_only_paths"] = tuple(values.pop("read_only_paths"))
    return AgentOptions(**values)


def validate_config(config):
    from pathlib import Path
    from aka.light.agent.sandbox import sandbox_executable
    from aka.light.agent.preflight import check_projected_proc
    import subprocess

    for value in config["read_only_paths"]:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.exists():
            raise ValueError("Agent read_only_paths must name existing absolute paths")
    # Explicit isolation fails before a Campaign is launched, including on macOS.
    options = _options(config)
    sandbox_executable(options.environment())
    if options.sandbox == "bwrap" and options.proc_mode == "projected":
        try:
            check_projected_proc()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise ValueError(
                "light-sandbox with proc_mode=projected requires Linux, a working "
                "C compiler and development headers in the execution container; "
                "install the platform packages and run "
                "`python -m aka.light.agent.preflight` before launch"
            ) from exc


def apply(ctx, config):
    from .provider import LightAgent
    agent = LightAgent(_options(config))
    ctx.effect(agent.close, label="close optional Agent")
    ctx.provide(AGENT, agent)
