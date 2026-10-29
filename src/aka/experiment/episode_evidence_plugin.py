"""Provide read-only episode evidence without opening an execution operation."""
from aka.contracts.experiment import EXPERIMENT_EPISODE_EVIDENCE
from .episode_evidence import LocalExperimentEpisodeEvidenceReader

name = "episode-evidence-plugin"
provide = ("experiment_episode_evidence",)
identity_packages = ("aka.experiment", "aka.contracts.experiment")
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


def apply(ctx, config):
    ctx.provide(EXPERIMENT_EPISODE_EVIDENCE, LocalExperimentEpisodeEvidenceReader())
