"""Public Agent-facing measurement tool selection; independent of execution internals."""
DEFAULT_AGATE_TOOLS = ("env", "evaluate", "command")
AGATE_TOOLS = (*DEFAULT_AGATE_TOOLS, "profile", "dev", "check", "disassemble")


def normalize_agate_tools(value):
    if not isinstance(value, (tuple, list)) or any(not isinstance(x, str) for x in value):
        raise ValueError("agate_tools must be a list of tool names")
    if tuple(value) == ("all",):
        return AGATE_TOOLS
    if len(set(value)) != len(value) or any(x not in AGATE_TOOLS for x in value):
        raise ValueError("agate_tools must contain unique names from " + ", ".join(AGATE_TOOLS) + "; or only all")
    return tuple(value)
