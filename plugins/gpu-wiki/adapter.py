"""Adapt the existing Wiki CLI to the AKA plugin JSON stdin/stdout contract."""

from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def main() -> None:
    request = json.load(sys.stdin)
    plugin_root = Path(__file__).resolve().parent
    manifest = json.loads((plugin_root / "plugin.json").read_text())
    root = (plugin_root / manifest["resources"]["gpu-wiki"]).resolve()
    internal = (
        plugin_root / manifest["resources"]["internal-gpu-wiki"]["path"]
    ).resolve()
    if internal != (root.parent / "internal_gpu_wiki").resolve():
        raise ValueError(
            "Wiki resources must declare the public store and its internal_gpu_wiki sibling"
        )
    override = os.environ.get("ATREX_WIKI_STORE_ROOT")
    if override and Path(override).expanduser().resolve() != root:
        raise ValueError(
            "ATREX_WIKI_STORE_ROOT differs from the locked plugin resource; configure the plugin's resources explicitly"
        )
    # A request file avoids interpreting leading '-' in prose as CLI options.
    with tempfile.TemporaryDirectory(prefix="aka-wiki-plugin-") as directory:
        path = Path(directory) / "request.txt"
        path.write_text(request["request"], encoding="utf-8")
        sys.argv = [
            str(root / "tools/query_nl.py"),
            "--store-root",
            str(root),
            "--file",
            str(path),
        ]
        for field in ("max_records", "max_bytes", "exclude"):
            if field in request:
                sys.argv.extend(["--" + field.replace("_", "-"), str(request[field])])
        sys.path.insert(0, str(root / "tools"))
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
