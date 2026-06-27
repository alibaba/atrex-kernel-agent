# SPDX-License-Identifier: MIT

"""Task17 project-local FlyDSL overlay package.

Only files present in this directory override the fixed AITER dependency. Missing
modules fall back to ``$AITER_BASE/aiter/ops/flydsl``.
"""

import os
from pathlib import Path


def _require_aiter_base() -> Path:
    value = os.environ.get("AITER_BASE")
    if not value:
        raise RuntimeError(
            "AITER_BASE must point to an aiter checkout. "
            "Example: export AITER_BASE=/path/to/aiter"
        )
    return Path(value).expanduser()


_base_flydsl = (
    _require_aiter_base()
    / "aiter"
    / "ops"
    / "flydsl"
)
if _base_flydsl.is_dir():
    _base = str(_base_flydsl.resolve())
    if _base not in __path__:
        __path__.append(_base)

from .moe_common import GateMode  # noqa: E402
from .utils import is_flydsl_available  # noqa: E402

__all__ = ["GateMode", "is_flydsl_available"]
