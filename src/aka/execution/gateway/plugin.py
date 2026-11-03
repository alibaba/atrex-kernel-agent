"""Select the shared gateway implementation with provider-owned configuration."""
from aka.core.keys import ServiceKey
from .executor import GatewayExecutor
from aka.execution.resources import PROFILING_CONFIG, contribute, validate_profiling
EXECUTOR = ServiceKey("executor", "Executor", module="aka.contracts.execution", protocol_version=2)
name = "gateway"
provide = ("executor",)
identity_packages = ("aka.execution", "aka.contracts.execution")
Config = {"type":"object", "properties":{
    "url":{"type":"string"}, "profile":{"type":"string"},
    "warmup_iters":{"type":"string"},
    "retry_delays":{"type":"array", "items":{"type":"integer", "minimum":0}},
    "idempotent_replay":{"type":"boolean"},
    "recovery_retry_delays":{"type":"array", "items":{"type":"number", "minimum":0}},
    "recovery_retry_budget_s":{"type":"number", "minimum":0},
    "dev_dependencies":{"type":"array", "items":{"type":"string"}}, "profiling": PROFILING_CONFIG}, "additionalProperties":False}
Defaults = {"warmup_iters":"default", "retry_delays":[], "dev_dependencies":[], "profiling":[],
            "idempotent_replay":False, "recovery_retry_delays":[], "recovery_retry_budget_s":0}
def validate_config(config):
    validate_profiling(config["profiling"])
    from aka.execution.job_recovery import recovery_policy
    recovery_policy(config)
    if config.get("idempotent_replay") and not config.get("url"):
        raise ValueError("stable-key replay requires a verified explicit gateway URL")
    if config.get("url") and config.get("profile"):
        raise ValueError("gateway url and profile are mutually exclusive")
    warmup = config["warmup_iters"]
    if warmup != "default" and (not warmup.isdecimal()):
        raise ValueError("warmup_iters must be default or a non-negative integer")
def apply(ctx, config):
    contribute(ctx, config["profiling"])
    from pathlib import Path
    from aka.contracts.agent_resources import AgentSkill
    ctx.provide(EXECUTOR, GatewayExecutor(config))
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parents[1] / "skills/aka-execution",
        "Prepare requests for the selected AKA Execution entry point and preserve complete results."))
