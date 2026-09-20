#!/usr/bin/env python3
"""Session-scoped HTTP client; GPU/Wiki authority stays in the Supervisor.

Existing GPU command syntax is preserved, for example:
  python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --no-memory
  python3 tools/sandbox.py --kind profile --no-sync -- python3 profile_driver.py
  python3 tools/sandbox.py --kind dev --input scratch/probe.py -- python3 scratch/probe.py
Additional typed operations: check, disassemble, env; Wiki: wiki-query, wiki-search,
wiki-hardware. Use --kind OPERATION --help for the operation's request options.
Read saved results with --kind record-read --record-id gateway-...;
copy source with --kind kernel-read --kernel-id kernel-... --output-path scratch/kernel.py;
list a Kernel's measurements with --kind kernel-records --kernel-id kernel-....
Journal writes: update-direction, record-experiment, episode-report --request-file FILE.
Journal indexes: list-directions, list-experiments --output-path scratch/FILE.
Journal reads: load-direction, load-experiment --record-id ID.
Journal tools require a registered Long Horizon Episode; see skills/runtime-records/.
No direct Agate fallback is performed when the Runtime is unavailable.
"""
from __future__ import annotations

import json
import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RUNTIME_URL_ENV = "ATREX_AKA_RUNTIME_URL"
RUNTIME_TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
# Wire contract with the Supervisor; keep this client independent of its modules.
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_REQUEST_FILE_BYTES = MAX_REQUEST_BYTES - 4 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
WIKI_KINDS = {"wiki-query": "query_nl", "wiki-search": "query_wiki", "wiki-hardware": "query_hardware"}
REQUEST_KINDS = {"update-direction": "direction_update", "record-experiment": "experiment_record", "episode-report": "episode_report"}
LIST_KINDS = {"list-directions": "directions_list", "list-experiments": "experiments_list"}
LOAD_KINDS = {"load-direction": ("direction_load", "direction_id"), "load-experiment": ("experiment_load", "experiment_id")}


def journal_request(kind: str, argv: list[str]) -> dict:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Runtime Journal; request schemas: skills/runtime-records/references/journal.md")
    parser.add_argument("--kind", choices=[*REQUEST_KINDS, *LIST_KINDS, *LOAD_KINDS], required=True)
    if kind in REQUEST_KINDS:
        parser.add_argument(
            "--request-file",
            required=True,
            help=(
                f"JSON object; at most {MAX_REQUEST_FILE_BYTES} bytes "
                "(2 MiB minus 4 KiB envelope reserve)"
            ),
        )
    elif kind in LIST_KINDS:
        parser.add_argument("--output-path", required=True, help="Workspace-relative scratch/ destination")
    else:
        parser.add_argument("--record-id", required=True)
    args = parser.parse_args(argv)
    if kind in REQUEST_KINDS:
        try:
            with Path(args.request_file).open("rb") as source:
                raw = source.read(MAX_REQUEST_FILE_BYTES + 1)
            if len(raw) > MAX_REQUEST_FILE_BYTES:
                raise ValueError(
                    f"request file exceeds {MAX_REQUEST_FILE_BYTES} bytes "
                    "(2 MiB minus 4 KiB reserved for the request envelope). "
                    "Shorten request text or lists before retrying; no request was sent"
                )
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request file must contain one JSON object")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return {"operation": REQUEST_KINDS[kind], "request": value}
    if kind in LIST_KINDS:
        return {"operation": LIST_KINDS[kind], "file": args.output_path}
    operation, field = LOAD_KINDS[kind]
    return {"operation": operation, field: args.record_id}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def proxy_command(path: str, payload: dict) -> int:
    url, token = os.environ.get(RUNTIME_URL_ENV, ""), os.environ.get(RUNTIME_TOKEN_ENV, "")
    if not url or not token:
        print("sandbox: this tool requires a live Supervisor Session; launch through optimize.py.", file=sys.stderr)
        return 75
    # Never send the Session bearer to an overridden remote URL, URL userinfo,
    # path or query. A Supervisor always advertises this exact loopback origin.
    try:
        address = urllib.parse.urlsplit(url)
        if (address.scheme != "http" or address.hostname != "127.0.0.1"
                or address.port is None or address.port == 0
                or address.username is not None or address.password is not None
                or address.path not in {"", "/"} or address.query or address.fragment):
            raise ValueError("invalid Runtime origin")
    except ValueError:
        print("sandbox: Runtime URL must be an HTTP 127.0.0.1 origin with an explicit port.", file=sys.stderr)
        return 75
    # File size alone cannot bound JSON re-encoding (e.g. Unicode escapes).
    # Check exactly the bytes sent, including the operation/request envelope.
    body = json.dumps(payload).encode()
    if len(body) > MAX_REQUEST_BYTES:
        print(
            f"sandbox: encoded HTTP request exceeds {MAX_REQUEST_BYTES} bytes (2 MiB), "
            "including JSON encoding and the request envelope. Shorten request text or "
            "lists before retrying; no request was sent.",
            file=sys.stderr,
        )
        return 2
    request = urllib.request.Request(
        url.rstrip("/") + path, data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    # Loopback traffic must not traverse an operator's HTTP proxy or follow a
    # redirect with the bearer token. No automatic retry of a possibly-run job.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=None) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        with error:
            raw = error.read(MAX_RESPONSE_BYTES + 1)
        try:
            value = json.loads(raw) if len(raw) <= MAX_RESPONSE_BYTES else None
        except (ValueError, UnicodeError):
            value = None
        print(json.dumps(value) if isinstance(value, dict) else f"sandbox: HTTP {error.code}", file=sys.stderr)
        # A confirmed pre-dispatch 429 and an unknown-outcome 503 both use
        # temporary-failure exit 75. The JSON repairable/code/next_action fields
        # distinguish safe backoff from escalation; never auto-resubmit here.
        return 2 if error.code in {400, 404, 413, 422} else 75
    except (OSError, urllib.error.URLError) as error:
        print(
            f"sandbox: Runtime unavailable ({error}); outcome may be unknown. "
            "Do not resubmit blindly; ask the operator to inspect the request.",
            file=sys.stderr,
        )
        return 75
    try:
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds limit")
        result = json.loads(raw)
        if not isinstance(result, dict) or type(result.get("exit_code")) is not int:
            raise ValueError("response omitted exit_code")
        for key, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
            value = result.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"invalid {key}")
            if value:
                print(value, end="" if value.endswith("\n") else "\n", file=stream)
        if result.get("truncated"):
            print("[sandbox] output shortened by Supervisor; ask the operator to inspect request diagnostics.", file=sys.stderr)
        return result["exit_code"]
    except (ValueError, UnicodeError) as error:
        print(f"sandbox: invalid Runtime response ({error}); outcome may be unknown.", file=sys.stderr)
        return 75


def proxy_wiki(tool: str, argv: list[str]) -> int:
    return proxy_command("/v1/wiki/query", {"tool": tool, "argv": argv})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        print(__doc__)
        return 0
    # Only recognize the full client-side dispatcher; the Supervisor validates
    # every argument using parsers with allow_abbrev=False.
    for index, value in enumerate(args):
        if value == "--":
            break
        kind = args[index + 1] if value == "--kind" and index + 1 < len(args) else (
            value.split("=", 1)[1] if value.startswith("--kind=") else None
        )
        if kind in {*REQUEST_KINDS, *LIST_KINDS, *LOAD_KINDS}:
            return proxy_command("/v1/journal/execute", journal_request(kind, args))
        if kind in WIKI_KINDS:
            count = 2 if value == "--kind" else 1
            return proxy_wiki(WIKI_KINDS[kind], args[:index] + args[index + count:])
    return proxy_command("/v1/gateway/execute", {"argv": args})


if __name__ == "__main__":
    raise SystemExit(main())
