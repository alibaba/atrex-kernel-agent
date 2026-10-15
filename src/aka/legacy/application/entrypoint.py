"""Compatibility export; implementation lives outside the normal runtime path."""
import sys
from aka.legacy.compat import entrypoint as _implementation
sys.modules[__name__] = _implementation
