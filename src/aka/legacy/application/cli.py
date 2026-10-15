"""Compatibility export; implementation lives outside the normal runtime path."""
import sys
from aka.legacy.compat import command as _implementation
sys.modules[__name__] = _implementation
