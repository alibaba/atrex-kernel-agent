#!/usr/bin/env python3
"""List and invoke the Session's portable tool contributions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from sandbox import MAX_REQUEST_FILE_BYTES, proxy_command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", allow_abbrev=False)
    call = commands.add_parser("call", allow_abbrev=False)
    call.add_argument("tool")
    call.add_argument("--input", required=True, help="JSON object file, or - for stdin")
    args = parser.parse_args(argv)
    if args.command == "list":
        return proxy_command("/v1/contributions/catalog", {}, raw_json=True)
    try:
        catalog = json.loads(os.environ.get("ATREX_AKA_CONTRIBUTIONS", "{}"))
        identity = catalog["identity"]
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("invalid contribution catalog identity")
        if args.input == "-":
            raw = sys.stdin.buffer.read(MAX_REQUEST_FILE_BYTES + 1)
        else:
            with Path(args.input).open("rb") as source:
                raw = source.read(MAX_REQUEST_FILE_BYTES + 1)
        if len(raw) > MAX_REQUEST_FILE_BYTES:
            raise ValueError("tool input exceeds the request limit")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("tool input must be a JSON object")
    except (KeyError, OSError, ValueError, TypeError) as error:
        parser.error(f"{error}; use the tools in the current Supervisor Session")
    return proxy_command("/v1/contributions/invoke",
                         {"catalog_id": identity, "tool": args.tool, "arguments": value}, raw_json=True)


if __name__ == "__main__":
    raise SystemExit(main())
