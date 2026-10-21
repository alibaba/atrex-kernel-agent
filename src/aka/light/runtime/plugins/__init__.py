"""AKA local plugin runtime."""

from aka.light.runtime.plugins.registry import Plugin, PluginRegistry
from aka.light.runtime.plugins.schema import PluginError

__all__ = ["Plugin", "PluginError", "PluginRegistry"]
