"""Default application assembly; feature relationships belong here."""
from aka.light.runtime.contributions import ContributionRegistry, ToolBinding


def compose(execute, *, modules=("measurement",), instruction_overrides=None):
    if len(set(modules)) != len(modules) or set(modules) - {"measurement"}:
        raise ValueError("unknown or duplicate contribution module")
    registry = ContributionRegistry()
    if "measurement" in modules:
        from aka.light.measurement.contributions import declarations, handlers

        def measurement(context, argv, *, prepare_only=False):
            result = execute(context, "/v1/gateway/execute", {"argv": argv},
                             prepare_identity_only=prepare_only)
            if prepare_only or result is None:
                return result
            return result.get("agent_result", result)

        registry.register(declarations(), {name: ToolBinding(*binding)
                                          for name, binding in handlers(measurement).items()})
    return registry.seal(instruction_overrides)
