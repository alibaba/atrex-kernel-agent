"""Compatibility module identity for the Agent-local backend registry."""

import sys
from . import backends as _implementation

sys.modules[__name__] = _implementation
