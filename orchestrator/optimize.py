#!/usr/bin/env python3
"""Historical CLI: adapt arguments before entering Bootstrap."""
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    source = root / "src"
    if (source / "aka" / "bootstrap" / "host.py").is_file():
        sys.path.insert(0, str(source))
    from aka.legacy.compat.entry import main as launch
    from aka.legacy.compat.launch import use_optimizer_entrypoint
    with use_optimizer_entrypoint(Path(__file__)):
        return launch(argv, repo_root=root)


if __name__ == "__main__":
    raise SystemExit(main())
