#!/usr/bin/env python3
"""Discover and call operator-enabled plugins through the Session Runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sandbox import MAX_REQUEST_FILE_BYTES, proxy_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", allow_abbrev=False)
    call = commands.add_parser("call", allow_abbrev=False)
    call.add_argument("tool")
    call.add_argument("--input", required=True, help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    request = {"action": args.command}
    if args.command == "call":
        try:
            if args.input == "-":
                raw = sys.stdin.buffer.read(MAX_REQUEST_FILE_BYTES + 1)
            else:
                with Path(args.input).open("rb") as source:
                    raw = source.read(MAX_REQUEST_FILE_BYTES + 1)
            if len(raw) > MAX_REQUEST_FILE_BYTES:
                raise ValueError("Plugin input exceeds the request limit; shorten it before retrying")
            request.update(tool=args.tool, input=json.loads(raw))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    return proxy_command("/v1/plugins/execute", request)


if __name__ == "__main__":
    raise SystemExit(main())
