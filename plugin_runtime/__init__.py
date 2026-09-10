"""Local plugin APIs independent of AKA, GPU Wiki and coding-agent backends."""

from .registry import HostLayout, Plugin, PluginRegistry
from .schema import PluginError

__all__ = ["HostLayout", "Plugin", "PluginError", "PluginRegistry"]
