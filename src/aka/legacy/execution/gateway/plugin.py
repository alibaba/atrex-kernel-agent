"""Select the shared gateway implementation with provider-owned configuration."""
from aka.core.keys import ServiceKey
from .executor import GatewayExecutor
EXECUTOR = ServiceKey("executor", "Executor", module="aka.contracts.execution", protocol_version=2)
name = "gateway"
provide = ("executor",)
identity_packages = ("aka.legacy.execution.gateway", "aka.contracts.execution")
Config = {"type":"object", "properties":{
    "url":{"type":"string"}, "profile":{"type":"string"},
    "requirements_file":{"type":"string"}, "warmup_iters":{"type":"string"},
    "retry_delays":{"type":"array", "items":{"type":"integer", "minimum":0}},
    "dev_dependencies":{"type":"array", "items":{"type":"string"}}}, "additionalProperties":False}
Defaults = {"warmup_iters":"5", "retry_delays":[5,15,30], "dev_dependencies":[]}
def validate_config(config):
    if config.get("url") and config.get("profile"):
        raise ValueError("gateway url and profile are mutually exclusive")
    warmup = config["warmup_iters"]
    if warmup != "default" and (not warmup.isdecimal()):
        raise ValueError("warmup_iters must be default or a non-negative integer")
def apply(ctx, config):
    from pathlib import Path
    from aka.contracts.agent_resources import AgentSkill
    config = dict(config)
    requirement = config.get("requirements_file", "")
    if ":" in requirement and not requirement.startswith("/"):
        from importlib.resources import files
        module, relative = requirement.split(":",1)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("invalid requirements resource")
        config["requirements_file"] = str(files(module).joinpath(relative))
    ctx.provide(EXECUTOR, GatewayExecutor(config))
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parents[1] / "skills/aka-execution",
        "Prepare requests for the selected AKA Execution entry point and preserve complete results."))
