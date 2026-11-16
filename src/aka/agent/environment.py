"""Keep host transports and unselected backend credentials out of Agent processes."""
from .backends import DEFAULT_BACKEND_REGISTRY

HOST_TRANSPORT_PREFIXES = (
    "AGATE_", "ATREX_SANDBOX_", "ATREX_WIKI_", "GPU_WIKI_",
    "ATREX_GPU_WIKI_", "AKA_EXECUTION_", "AKA_GATEWAY_", "ATREX_AKA_RUNTIME_",
)
HOST_TRANSPORT_NAMES = frozenset({"ATREX_API_TOKEN", "ATREX_SERVER_URL", "ATREX_PRIVATE_REFERENCE_DIR"})
# These are public guard targets, not credentials. Only a host policy may set them.
POLICY_ENVIRONMENT_PREFIX = "ATREX_PROTECTED_GATEWAY_"
EXPLICIT_ROUTING = frozenset({
    "ATREX_SANDBOX_GPU", "ATREX_SANDBOX_TIMEOUT", "ATREX_SANDBOX_PROFILE", "ATREX_SANDBOX_URL",
    "ATREX_SANDBOX_SSH", "ATREX_SANDBOX_SSH_INIT", "ATREX_SANDBOX_SSH_GPU",
    "ATREX_SANDBOX_HEALTH_COMMAND", "ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND",
    "ATREX_SANDBOX_SSH_RUNTIME_BINDS", "ATREX_SANDBOX_REQUIREMENTS_FILE",
    "AKA_EXECUTION_SELECTION", "ATREX_PRIVATE_REFERENCE_DIR",
})


def host_transport(name):
    return name in HOST_TRANSPORT_NAMES or name.startswith(HOST_TRANSPORT_PREFIXES)


def filter_environment(values, backend, *, explicit=()):
    selected = set(getattr(backend, "environment_variables", ()))
    unselected = {name for runtime_id in DEFAULT_BACKEND_REGISTRY.ids
                  for name in DEFAULT_BACKEND_REGISTRY.create(runtime_id).environment_variables} - selected
    if any(host_transport(name) and name not in EXPLICIT_ROUTING or name in unselected for name in explicit):
        raise PermissionError("Agent cannot receive host transport or unselected backend environment")
    if any(name.startswith(POLICY_ENVIRONMENT_PREFIX) for name in explicit):
        raise PermissionError("Agent protection targets are owned by the host policy")
    return {name: value for name, value in values.items()
            if (not host_transport(name) or name in explicit and name in EXPLICIT_ROUTING)
            and name not in unselected and not name.startswith(POLICY_ENVIRONMENT_PREFIX)}
