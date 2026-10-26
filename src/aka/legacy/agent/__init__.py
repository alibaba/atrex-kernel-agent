"""Compatibility exports for the shared Agent and Workflow host policies."""
import aka.agent as _shared
from aka.legacy.workflow import agent_policy as _policy
__all__ = [*_shared.__all__, "DEFAULT_PROTECTED_GATEWAY_SCREEN", "DEFAULT_PROTECTED_GATEWAY_STATE_NAME",
           "dependency_process_violation", "protected_gateway_identity", "python_import_roots"]
def __getattr__(name):
    if hasattr(_policy, name):
        return getattr(_policy, name)
    return getattr(_shared, name)
