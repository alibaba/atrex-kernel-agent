from aka.core.keys import ServiceKey
ACCEPTANCE = ServiceKey("light_acceptance", "Acceptance", module="aka.light.contracts")
name = "acceptance"
provide = ("light_acceptance",)
identity_packages = ("aka.light.acceptance", "aka.light.acceptance.recorded_verifier", "aka.light.acceptance.verifier")
Config = {"type": "object", "properties": {}, "additionalProperties": False}

def apply(ctx, config):
    from .provider import RecordedAcceptance
    ctx.provide(ACCEPTANCE, RecordedAcceptance())
