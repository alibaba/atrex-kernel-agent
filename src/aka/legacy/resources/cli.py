#!/usr/bin/env python3
"""Discover and invoke enabled local AKA plugin tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from aka.legacy.resources.plugins import PluginError, PluginRegistry, decode_json, read_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", help="Select an independent local plugin catalog")
    parser.add_argument("--plugin", action="append", help="Include an explicit directory containing plugin.json")
    parser.add_argument(
        "--workspace", default=("." if (Path.cwd() / ".atrex_plugins/lock.json").is_file()
                                else os.environ.get("ATREX_PLUGIN_WORKSPACE", "."))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    call = commands.add_parser("call")
    call.add_argument("tool")
    call.add_argument(
        "--input", required=True, help="Request JSON file, or - for stdin"
    )
    args = parser.parse_args(argv)
    try:
        workspace = Path(args.workspace).resolve()
        registry = PluginRegistry(args.plugin_dir, plugins=args.plugin)
        registry.check_lock(workspace)
        if args.command == "list":
            from aka.legacy.agent.resources import discover_resources
            result = discover_resources(registry.resource_result(workspace))
        else:
            request = (
                decode_json(sys.stdin.read())
                if args.input == "-"
                else read_json(Path(args.input))
            )
            result = registry.call(args.tool, request, workspace, cwd=Path.cwd())
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (PluginError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "error": {
                        "code": getattr(exc, "code", "invalid_input"),
                        "message": str(exc),
                    }
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
