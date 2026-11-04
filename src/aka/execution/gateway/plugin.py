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
Defaults = {"dev_dependencies":[], "profiling":[],
            "idempotent_replay":False, "recovery_retry_delays":[], "recovery_retry_budget_s":0}
def validate_config(config):
    if "retry_delays" in config:
        raise ValueError("gateway.retry_delays is retired; configure experiment.retry_delays and retry_budget_s")
    validate_profiling(config["profiling"])
    from aka.execution.job_recovery import recovery_policy
    recovery_policy(config)
    if config.get("idempotent_replay") and not config.get("url"):
        raise ValueError("stable-key replay requires a verified explicit gateway URL")
    if config.get("url") and config.get("profile"):
        raise ValueError("gateway url and profile are mutually exclusive")
    if "warmup_iters" in config:
        raise ValueError("gateway.warmup_iters is retired; set --warmup on the Evaluation request")

def apply(ctx, config):
    contribute(ctx, config["profiling"])
    from pathlib import Path
    from aka.contracts.agent_resources import AgentSkill
    ctx.provide(EXECUTOR, GatewayExecutor(config))
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parents[1] / "skills/aka-execution",
        "Prepare requests for the selected AKA Execution entry point and preserve complete results."))
    ctx.contribute("agent_skills", "atrex-gpu-gateway", AgentSkill(
        "atrex-gpu-gateway", Path(__file__).resolve().parents[1] / "skills/atrex-gpu-gateway",
        "Author native candidate manifests and use the selected Gateway operations."))
