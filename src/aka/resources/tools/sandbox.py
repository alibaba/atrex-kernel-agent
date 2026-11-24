#!/usr/bin/env python3
"""Session-scoped Runtime client. Use --help for this Campaign's enabled tools,
and --kind OPERATION --help for its request schema. Wiki, research and submission
operations follow their independently installed Skills. Runtime returns the full
canonical receipt and saves observations in records/. No direct Agate fallback.
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
WIKI_KINDS = {"wiki-query": "query_nl", "wiki-search": "query_wiki", "wiki-hardware": "query_hardware"}
REQUEST_KINDS = {
    "update-direction": "direction_update",
    "prioritize-direction": "direction_prioritize",
    "record-experiment": "experiment_record",
    "submit-kernel": "kernel_submit",
}


def client_error(code: str, message: str, instruction: str, *, repairable=False, path="$", **details) -> None:
    """Keep the standalone tool client independent of private Supervisor modules."""
    print(json.dumps({
        "ok": False, "repairable": repairable, "detail": message,
        "error": {"code": code, "message": message, "next_action": instruction, **details},
        "issues": [{"path": path, "code": code, "message": message}],
    }), file=sys.stderr)


class _ClientParser(argparse.ArgumentParser):
    def error(self, message):
        client_error("invalid_request", message,
                     "Correct the CLI arguments or JSON request file and retry; no request was sent. "
                     "Use --kind OPERATION --help and .agents/skills/runtime-records/references/journal.md.",
                     repairable=True)
        raise SystemExit(2)


def journal_request(kind: str, argv: list[str]) -> dict:
    parser = _ClientParser(allow_abbrev=False, description="Runtime research and submission mutations; request schemas: .agents/skills/runtime-records/references/journal.md")
    parser.add_argument("--kind", choices=[kind], required=True)
    parser.add_argument(
        "--request-file",
        required=True,
        help=(
            f"JSON object; at most {MAX_REQUEST_FILE_BYTES} bytes "
            "(2 MiB minus 4 KiB envelope reserve)"
        ),
    )
    args = parser.parse_args(argv)
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


def proxy_command(path: str, payload: dict) -> int:
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
        # Bind to the installed client, normalizing ../tools invocations without
        # following the client file into a shared resource installation.
        workspace = Path(os.path.abspath(__file__)).parent.parent
        raw = call_tool(names[path], body, workspace=workspace)
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
        json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, UnicodeError):
        client_error("invalid_runtime_response", "The operation reply is invalid.",
                     "Inspect saved records; do not resubmit blindly.")
        return 75
    try:
        if (isinstance(result, dict) and isinstance(result.get("operation"), str)
                and result.get("status") in {"completed", "failed", "cancelled", "unknown"}
                and isinstance(result.get("result"), dict)):
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
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
        raise ValueError("response omitted the canonical operation/status/result receipt")
    except (ValueError, UnicodeError) as error:
        client_error("invalid_runtime_response", f"Invalid Runtime response ({error}); outcome may be unknown.",
                     "Inspect saved records and ask the operator to reconcile the request. Do not resubmit blindly.")
        return 75


def proxy_wiki(tool: str, argv: list[str]) -> int:
    return proxy_command("/v1/wiki/query", {"tool": tool, "argv": argv})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    kinds = json.loads(os.environ.get("AKA_RUNTIME_TOOLS",
        os.environ.get("AKA_AGATE_TOOLS", '["env", "evaluate", "command"]')))
    if args in (["--help"], ["-h"]):
        print("Enabled Runtime tools: " + ", ".join(kinds))
        print("Use --kind OPERATION --help for enabled tool options. "
              "Research and submit-kernel use --request-file; read the runtime-records Skill for JSON fields.")
        print("Evaluate/Command optionally accept --predictions JSON before --: "
              "a JSON array of {scope, prediction}, frozen before new execution.")
        return 0
    # Only recognize the full client-side dispatcher; the Supervisor validates
    # every argument using parsers with allow_abbrev=False.
    for index, value in enumerate(args):
        if value == "--":
            break
        kind = args[index + 1] if value == "--kind" and index + 1 < len(args) else (
            value.split("=", 1)[1] if value.startswith("--kind=") else None
        )
        if kind and "AKA_RUNTIME_TOOLS" in os.environ and kind not in kinds:
            client_error("runtime_tool_disabled", f"Runtime tool {kind!r} is not enabled.",
                         "Use python3 tools/sandbox.py --help for enabled tools.", available_kinds=kinds)
            return 2
        if kind in REQUEST_KINDS:
            return proxy_command("/v1/journal/execute", journal_request(kind, args))
        if kind in WIKI_KINDS:
            count = 2 if value == "--kind" else 1
            return proxy_wiki(WIKI_KINDS[kind], args[:index] + args[index + count:])
    predictions = None
    predictions_supplied = False
    clean = []
    index = 0
    while index < len(args):
        word = args[index]
        if word == "--":
            clean.extend(args[index:])
            break
        if word == "--predictions" or word.startswith("--predictions="):
            if predictions_supplied:
                raise ValueError("Specify --predictions only once")
            predictions_supplied = True
            if word == "--predictions":
                index += 1
                if index == len(args):
                    raise ValueError("--predictions requires a JSON array")
                raw_predictions = args[index]
            else:
                raw_predictions = word.split("=", 1)[1]
            predictions = json.loads(raw_predictions)
        else:
            clean.append(word)
        index += 1
    payload = {"argv": clean}
    if predictions_supplied:
        payload["predictions"] = predictions
    return proxy_command("/v1/gateway/execute", payload)


if __name__ == "__main__":
    raise SystemExit(main())
