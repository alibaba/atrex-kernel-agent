#!/usr/bin/env python3
"""Discover and invoke enabled local AKA plugin tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.plugins import (
    DEFAULT_CONFIG,
    STATE_DIR,
    PluginError,
    PluginRegistry,
    decode_json,
    read_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Plugin configuration JSON (defaults to campaign lock or bundled defaults)",
    )
    parser.add_argument(
        "--workspace", default=os.environ.get("ATREX_PLUGIN_WORKSPACE", ".")
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
        lock = workspace / STATE_DIR / "lock.json"
        config = args.config or os.environ.get("ATREX_PLUGIN_CONFIG")
        if config is None and lock.exists():
            config = read_json(lock)["config"]
        registry = PluginRegistry(config or DEFAULT_CONFIG)
        registry.check_lock(workspace)
        if args.command == "list":
            result = {"tools": registry.catalog(), "skills": registry.skill_catalog()}
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
