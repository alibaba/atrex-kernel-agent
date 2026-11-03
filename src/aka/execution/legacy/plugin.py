"""Register the unchanged host runner as the legacy execution provider."""
from .executor import LegacyExecutor
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill
from aka.execution.resources import PROFILING_CONFIG, contribute, validate_profiling

name = "legacy"
provide = ("executor",)
identity_files = ("plugin.py", "executor.py", "runtime.py")
identity_packages = ("aka.execution", "aka.contracts.execution")
Config = {"type": "object", "properties": {"gateway_skill_path": {"type": "string"}, "profiling": PROFILING_CONFIG,
          "idempotent_replay":{"type":"boolean"},
          "url":{"type":"string"}, "profile":{"type":"string"},
          "recovery_retry_delays":{"type":"array", "items":{"type":"number", "minimum":0}},
          "recovery_retry_budget_s":{"type":"number", "minimum":0}},
          "additionalProperties": False}
Defaults = {"gateway_skill_path": "", "profiling": [], "idempotent_replay":False,
            "recovery_retry_delays":[], "recovery_retry_budget_s":0}


def validate_config(config):
    validate_profiling(config["profiling"])
    from aka.execution.job_recovery import recovery_policy
    recovery_policy(config)
    if config.get("url") and config.get("profile"):
        raise ValueError("gateway url and profile are mutually exclusive")
    if config.get("idempotent_replay") and not config.get("url"):
        raise ValueError("stable-key replay requires a verified explicit gateway URL")
    if config["gateway_skill_path"]:
        AgentSkill("atrex-gpu-gateway", Path(config["gateway_skill_path"]).expanduser(),
                   "Gateway operations, request parameters, and client configuration.")


def apply(ctx, config):
    contribute(ctx, config["profiling"])
    ctx.provide("executor", LegacyExecutor(configuration=config))
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parent.parent / "skills" / "aka-execution",
        "Prepare requests for the selected AKA Execution entry point and preserve complete results."))
    skill = (Path(config["gateway_skill_path"]).expanduser().resolve() if config["gateway_skill_path"]
             else Path(__file__).resolve().parents[1] / "skills/atrex-gpu-gateway")
    ctx.contribute("agent_skills", "atrex-gpu-gateway", AgentSkill(
        "atrex-gpu-gateway", skill, "Gateway operations, request parameters, and client configuration."))
