#!/usr/bin/env python3
"""Session-scoped Agent tool client; GPU/Wiki authority stays in the Supervisor.

Existing GPU command syntax is preserved, for example:
  python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --no-memory
  python3 tools/sandbox.py --kind profile --no-sync -- python3 profile_driver.py
  python3 tools/sandbox.py --kind dev --input scratch/probe.py -- python3 scratch/probe.py
Additional typed operations: check, disassemble, env; Wiki: wiki-query, wiki-search,
wiki-hardware. Use --kind OPERATION --help for the operation's request options.
Read saved results with --kind record-read --record-id gateway-...;
copy source with --kind kernel-read --kernel-id kernel-... --output-path scratch/kernel.py;
list a Kernel's measurements with --kind kernel-records --kernel-id kernel-....
Find each Shape's fastest measured Kernel with --kind kernel-pareto-frontier (no GPU job).
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
import stat
import sys
from pathlib import Path

# Wire contract with the Supervisor; keep this client independent of its modules.
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_REQUEST_FILE_BYTES = MAX_REQUEST_BYTES - 4 * 1024
# Canonical result payload limit, matching the Supervisor's compact ASCII JSON.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
WIKI_KINDS = {"wiki-query": "query_nl", "wiki-search": "query_wiki", "wiki-hardware": "query_hardware"}
REQUEST_KINDS = {"update-direction": "direction_update", "record-experiment": "experiment_record", "episode-report": "episode_report"}
LIST_KINDS = {"list-directions": "directions_list", "list-experiments": "experiments_list"}
LOAD_KINDS = {"load-direction": ("direction_load", "direction_id"), "load-experiment": ("experiment_load", "experiment_id")}


def client_error(code: str, message: str, instruction: str, *, repairable=False, path="$", **details) -> None:
    """Keep the standalone tool client independent of private Supervisor modules."""
    print(json.dumps({
        "ok": False, "repairable": repairable, "detail": message,
        "error": {"code": code, "message": message, "next_action": instruction, **details},
        "issues": [{"path": path, "code": code, "message": message}],
        "recovery": [{"instruction": instruction}],
    }), file=sys.stderr)


class _ClientParser(argparse.ArgumentParser):
    def error(self, message):
        client_error("invalid_request", message,
                     "Correct the CLI arguments or JSON request file and retry; no request was sent. "
                     "Use --kind OPERATION --help and skills/runtime-records/references/journal.md.",
                     repairable=True)
        raise SystemExit(2)


def journal_request(kind: str, argv: list[str]) -> dict:
    parser = _ClientParser(allow_abbrev=False, description="Runtime Journal; request schemas: skills/runtime-records/references/journal.md")
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


def proxy_command(path: str, payload: dict, *, record_read=False) -> int:
    client_path = os.environ.get("AKA_AGENT_TOOL_CLIENT_PATH")
    if client_path:
        sys.path.insert(0, client_path)
        from tools.client import call_tool, ToolSessionError
    else:
        from aka.agent.tools.client import call_tool, ToolSessionError
    names = {"/v1/gateway/execute": "workflow.gateway", "/v1/journal/execute": "workflow.journal",
             "/v1/wiki/query": "workflow.wiki", "/v1/plugins/execute": "workflow.plugins"}
    body = json.dumps(payload).encode()
    if len(body) > MAX_REQUEST_BYTES:
        client_error("request_too_large", "Encoded request exceeds 2 MiB.",
                     "Shorten the request; no operation was sent.", repairable=True)
        return 2
    try:
        raw = call_tool(names[path], body, workspace=Path.cwd())
    except ToolSessionError as error:
        if error.payload is None:
            client_error(error.code, "The tool session or operation outcome could not be confirmed.",
                         "Inspect saved records and ask the operator; do not resubmit blindly.")
            return 75
        raw = error.payload
    try:
        envelope = json.loads(raw)
        if (not isinstance(envelope, dict) or type(envelope.get("status")) is not int
                or not isinstance(envelope.get("result"), dict)):
            raise ValueError("invalid tool reply")
        execution_exit_code = envelope.get("execution_exit_code")
        status = envelope["status"]
        result = envelope["result"]
        if status >= 400:
            print(json.dumps(result), file=sys.stderr)
            return 2 if status in {400, 404, 413, 422} else 75
        raw = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, UnicodeError):
        client_error("invalid_runtime_response", "The operation reply is invalid.",
                     "Inspect saved records; do not resubmit blindly.")
        return 75
    try:
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds limit")
        result = json.loads(raw)
        if (isinstance(result, dict) and isinstance(result.get("operation"), str)
                and result.get("status") in {"completed", "failed", "cancelled", "unknown"}
                and isinstance(result.get("result"), dict)):
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
            # Reading a failed measurement succeeds; it is not a failed query.
            if record_read:
                return 0
            if result["status"] == "unknown" or result["result"].get("outcome") == "unknown":
                return 75
            # Preserve the executor's tempfail/candidate exit classification
            # without putting the compatibility stream envelope back in JSON.
            if execution_exit_code is not None:
                code = int(execution_exit_code)
                if not -255 <= code <= 255:
                    raise ValueError("invalid execution exit code")
                if code:
                    return code
            detail = result["result"]
            if detail.get("outcome") == "unknown" or (
                isinstance(detail.get("error"), dict) and detail["error"].get("error_class") == "infra"
            ):
                return 75
            return 0 if result["status"] == "completed" and detail.get("correct") is not False else 1
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
        client_error("invalid_runtime_response", f"Invalid Runtime response ({error}); outcome may be unknown.",
                     "Inspect saved records and ask the operator to reconcile the request. Do not resubmit blindly.")
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
    record_read = False
    for index, value in enumerate(args):
        if value == "--":
            break
        kind = args[index + 1] if value == "--kind" and index + 1 < len(args) else (
            value.split("=", 1)[1] if value.startswith("--kind=") else None
        )
        if kind is not None:
            record_read = kind == "record-read"
        if kind in {*REQUEST_KINDS, *LIST_KINDS, *LOAD_KINDS}:
            return proxy_command("/v1/journal/execute", journal_request(kind, args))
        if kind in WIKI_KINDS:
            count = 2 if value == "--kind" else 1
            return proxy_wiki(WIKI_KINDS[kind], args[:index] + args[index + count:])
    return proxy_command("/v1/gateway/execute", {"argv": args}, record_read=record_read)


if __name__ == "__main__":
    raise SystemExit(main())
