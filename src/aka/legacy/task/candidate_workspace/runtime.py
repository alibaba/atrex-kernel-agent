"""Preserve Task installation entry points while delegating resource ownership."""
from aka.legacy.resources.workspace import _install_atrex_bench_runtime, link_runtime

__all__ = ["link_runtime"]
