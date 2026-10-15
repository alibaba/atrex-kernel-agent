#!/usr/bin/env python3
"""Checkout command entry; transport implementation belongs to Execution."""
from pathlib import Path
import os, sys
root = Path(__file__).resolve().parent.parent
if (root / "src").is_dir(): sys.path.insert(0, str(root / "src"))
if str(root) not in sys.path: sys.path.insert(0, str(root))

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    from aka.legacy.execution.selection import selected_execution
    if selected_execution() != {"profile":"legacy"}:
        from aka.legacy.execution.cli import main as selected_main
        prefix = ["--repo-root", str(root)]
        options = arguments[:arguments.index("--")] if "--" in arguments else arguments
        if "--workspace" not in options and not any(v.startswith("--workspace=") for v in options):
            prefix += ["--workspace", str(Path.cwd())]
        return selected_main([*prefix, *arguments])
    from aka.legacy.execution.gateway.sandbox import main as transport_main
    return transport_main(arguments)

if __name__ == "__main__":
    raise SystemExit(main())
