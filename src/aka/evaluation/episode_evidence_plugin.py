"""Provide pure retained measurement interpretation without an Executor."""
from aka.contracts.evaluation import EVALUATION_EPISODE_INTERPRETER
from .episode_evidence import LegacyEvaluationEpisodeInterpreter

name = "episode-evidence-plugin"
provide = ("evaluation_episode_interpreter",)
identity_packages = ("aka.evaluation", "aka.contracts.evaluation")
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


def apply(ctx, config):
    ctx.provide(EVALUATION_EPISODE_INTERPRETER, LegacyEvaluationEpisodeInterpreter())
