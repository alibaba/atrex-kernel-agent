# SPDX-License-Identifier: MIT

"""Task17 kernel overlay package with fallback to the fixed AITER dependency."""

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


_base_kernels = (
    _require_aiter_base()
    / "aiter"
    / "ops"
    / "flydsl"
    / "kernels"
)
if _base_kernels.is_dir():
    _base = str(_base_kernels.resolve())
    if _base not in __path__:
        __path__.append(_base)
