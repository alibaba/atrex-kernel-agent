"""Select measurement preparation without importing Core in business consumers."""
from aka.core.keys import ServiceKey
from .evaluator import LegacyEvaluator
from aka.contracts.evaluation import EvaluationConfiguration

EVALUATOR_FACTORY = ServiceKey("evaluator_factory", "EvaluatorFactory",
                               module="aka.contracts.evaluation")
EVALUATION_DEFAULTS = ServiceKey("evaluation_defaults", "EvaluationConfiguration", module="aka.contracts.evaluation")
name = "evaluation"
provide = ("evaluator_factory", "evaluation_defaults")
identity_packages = ("aka.evaluation", "aka.contracts.evaluation")
Config = {"type": "object", "properties": {
    "shape_batch_size": {"type": "integer", "minimum": 1},
    "comparison_method": {"type": "string", "enum": ["legacy-abba", "native-abba"]},
    "measurement_repetitions": {"type": "integer", "enum": [1, 3]},
}, "additionalProperties": False}
Defaults = {"shape_batch_size": 4, "comparison_method": "legacy-abba", "measurement_repetitions": 1}


class LegacyEvaluatorFactory:
    def __init__(self, *, shape_batch_size=4):
        self.shape_batch_size = shape_batch_size

    def create(self, configuration, *, candidate_workspace, evaluation_input):
        return LegacyEvaluator(configuration, candidate_workspace=candidate_workspace,
                               evaluation_input=evaluation_input)
    def create_command(self, configuration, *, evaluation_input):
        from .gateway_command import GatewayCommandEvaluator
        return GatewayCommandEvaluator(measurement_reader=configuration.measurement_reader,
            execution_capabilities=configuration.execution_capabilities,
            evaluation_input=evaluation_input, shape_batch_size=configuration.shape_batch_size,
            measurement_repetitions=configuration.measurement_repetitions)


def apply(ctx, config):
    from .resources import contribute_resources
    contribute_resources(ctx)
    ctx.provide(EVALUATOR_FACTORY, LegacyEvaluatorFactory(shape_batch_size=config["shape_batch_size"]))
    ctx.provide(EVALUATION_DEFAULTS, EvaluationConfiguration(**config))
