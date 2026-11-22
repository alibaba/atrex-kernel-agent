"""Agent-facing Runtime catalog, independent of Executor operation names."""
from .agate_tools import normalize_agate_tools


def runtime_tools(agate_tools, *, research_modules=(), submissions=False, wiki=False):
    kinds = list(normalize_agate_tools(agate_tools))
    if "directions" in research_modules:
        kinds.extend(("update-direction", "prioritize-direction"))
    if "experiments" in research_modules:
        kinds.append("record-experiment")
    if submissions:
        kinds.append("submit-kernel")
    if wiki:
        kinds.extend(("wiki-query", "wiki-search", "wiki-hardware"))
    return tuple(kinds)
