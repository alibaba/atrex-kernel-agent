"""Compatibility export; implementation lives outside the normal runtime path."""
import sys
from aka.legacy.compat import host as _implementation
sys.modules[__name__] = _implementation
