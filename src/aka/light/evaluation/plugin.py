from aka.core.keys import ServiceKey
EVALUATION = ServiceKey("light_evaluation", "Evaluation", module="aka.light.contracts")
name = "evaluation"
provide = ("light_evaluation",)
identity_packages = ("aka.light.evaluation", "aka.light.evaluation.gateway", "aka.light.evaluation.projection")
Config = {"type": "object", "properties": {}, "additionalProperties": False}

def apply(ctx, config):
    from .provider import GatewayEvaluation
    ctx.provide(EVALUATION, GatewayEvaluation())
