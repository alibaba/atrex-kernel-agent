"""Public, model-free checks for the Light Agent execution environment.

Run ``python -m aka.light.agent.preflight`` in the Agent's execution container.
This checks projected-proc shim compilation, not Bubblewrap namespace admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check_projected_proc() -> None:
    """Verify the packaged shim and compiler without launching an Agent."""
    if sys.platform != "linux":
        raise RuntimeError("projected-proc requires a Linux execution container")
    from .proc_view import _install_maps_shim

    with tempfile.TemporaryDirectory(prefix="aka-proc-preflight-") as directory:
        # A valid cache must not hide a missing compiler/header in this container.
        _install_maps_shim(Path(directory), force_rebuild=True)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    try:
        check_projected_proc()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(
            f"Projected-proc preflight failed ({type(exc).__name__}); "
            "check Linux, cc/C development headers, packaged proc_maps.c and "
            "writable temporary storage in this execution container",
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"check": "projected_proc_compile", "status": "ready"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
