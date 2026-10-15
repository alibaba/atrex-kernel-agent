"""Select measurement preparation without importing Core in business consumers."""
from aka.core.keys import ServiceKey
from aka.contracts.evaluation import EvaluationResources
from .evaluator import LegacyEvaluator

EVALUATOR_FACTORY = ServiceKey("evaluator_factory", "EvaluatorFactory",
                               module="aka.contracts.evaluation")
name = "evaluation"
provide = ("evaluator_factory",)
identity_packages = ("aka.legacy.evaluation", "aka.contracts.evaluation")
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


class LegacyEvaluatorFactory:
    resources = EvaluationResources()

    def create(self, configuration, *, candidate_workspace):
        return LegacyEvaluator(configuration, candidate_workspace=candidate_workspace)


def apply(ctx, config):
    ctx.provide(EVALUATOR_FACTORY, LegacyEvaluatorFactory())
