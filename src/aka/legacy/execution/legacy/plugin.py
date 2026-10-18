"""Register the unchanged host runner as the legacy execution provider."""
from .executor import LegacyExecutor
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill
from aka.legacy.execution.resources import PROFILING_CONFIG, contribute, validate_profiling

name = "legacy"
provide = ("executor",)
identity_files = ("plugin.py", "executor.py", "runtime.py")
identity_packages = ("aka.legacy.execution", "aka.contracts")
Config = {"type": "object", "properties": {"gateway_skill_path": {"type": "string"}, "profiling": PROFILING_CONFIG},
          "additionalProperties": False}
Defaults = {"gateway_skill_path": "", "profiling": []}


def validate_config(config):
    validate_profiling(config["profiling"])
    if config["gateway_skill_path"]:
        AgentSkill("atrex-gpu-gateway", Path(config["gateway_skill_path"]).expanduser(),
                   "Gateway operations, request parameters, and client configuration.")


def apply(ctx, config):
    contribute(ctx, config["profiling"])
    ctx.provide("executor", LegacyExecutor())
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parent.parent / "skills" / "aka-execution",
        "Prepare requests for the selected AKA Execution entry point and preserve complete results."))
    if config["gateway_skill_path"]:
        ctx.contribute("agent_skills", "atrex-gpu-gateway", AgentSkill(
            "atrex-gpu-gateway", Path(config["gateway_skill_path"]).expanduser().resolve(),
            "Gateway operations, request parameters, and client configuration."))
