"""Reconstruct a Workflow child from its frozen Bootstrap selection."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aka.bootstrap.host import run_selection
from aka.bootstrap.continuation import read, restore
import os
from aka.contracts.startup import Invocation
from aka.legacy.workflow.invocation import decode_launch


def main(argv=None):
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    decode_launch(arguments)
    payload = read(os.environ)
    if payload is None:
        raise ValueError("Workflow child requires a recorded Bootstrap selection")
    selection = restore(payload)
    root = Path(selection.composition.variables["repo_root"]).resolve()
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        return run_selection(selection, Invocation(arguments))
    finally:
        if added:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(main())
