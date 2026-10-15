"""Select measurement preparation without importing Core in business consumers."""
from aka.core.keys import ServiceKey
from aka.contracts.evaluation import EvaluationResources
from .evaluator import LegacyEvaluator

EVALUATOR_FACTORY = ServiceKey("evaluator_factory", "EvaluatorFactory",
                               module="aka.contracts.evaluation")
name = "evaluation"
provide = ("evaluator_factory",)
identity_packages = ("aka.legacy.evaluation", "aka.contracts.evaluation")
Config = {"type": "object", "properties": {"shape_batch_size":{"type":"integer","minimum":1}}, "additionalProperties": False}
Defaults = {"shape_batch_size":4}


class LegacyEvaluatorFactory:
    def __init__(self, *, native_harness=None, shape_batch_size=4):
        self.resources = EvaluationResources(native_harness)
        self.shape_batch_size = shape_batch_size

    def create(self, configuration, *, candidate_workspace):
        return LegacyEvaluator(configuration, candidate_workspace=candidate_workspace)


def apply(ctx, config):
    ctx.provide(EVALUATOR_FACTORY, LegacyEvaluatorFactory(shape_batch_size=config["shape_batch_size"]))
