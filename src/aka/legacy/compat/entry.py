"""Installed compatibility command and recovery entry; convert before Bootstrap."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv=None, *, repo_root=None):
    from aka.contracts.application import ApplicationRequest
    from aka.legacy.application.paths import checkout_root
    from aka.legacy.compat.host import run_application
    root = checkout_root() if repo_root is None else Path(repo_root).resolve()
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        return run_application(ApplicationRequest(root, argv))
    finally:
        if added:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(main())
