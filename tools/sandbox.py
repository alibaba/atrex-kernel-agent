#!/usr/bin/env python3
"""Agent CLI for Supervisor-owned GPU execution, Wiki, Journal, and reports.

This command intentionally contains no Agate client, evaluator packaging,
retry policy, result projection, or evidence persistence. The Agent supplies
only the requested operation arguments; a short-lived capability binds the
request to its exact workspace, and the Supervisor returns the projected CLI
result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RUNTIME_URL_ENV = "ATREX_AKA_RUNTIME_URL"
RUNTIME_TOKEN_ENV = "ATREX_AKA_RUNTIME_TOKEN"
MAX_RUNTIME_RESPONSE_BYTES = 1024 * 1024
_UNKNOWN_OUTCOME = (
    "The request outcome may be unknown. Inspect available records before resubmitting a "
    "mutation or GPU request; do not retry blindly. If the Runtime remains unavailable, "
    "report the blocker to the operator. Do not change credentials or bypass this client."
)


def _client_error(message: str, *, code: str = "runtime_response_invalid") -> None:
    print(json.dumps({
        "ok": False,
        "repairable": False,
        "error": {"code": code, "message": message, "next_action": _UNKNOWN_OUTCOME},
    }), file=sys.stderr)


def _runtime_endpoint(path: str) -> str | None:
    base = os.environ.get(RUNTIME_URL_ENV, "").strip().rstrip("/")
    token = os.environ.get(RUNTIME_TOKEN_ENV, "").strip()
    if not base or not token:
        return None
    return base + path


def proxy_command(path: str, payload: dict[str, Any]) -> int | None:
    """Execute one Runtime request, preserving the wrapped CLI's exit contract.

    ``None`` means that no Supervisor Runtime is configured; the CLI rejects
    the request. Failures never fall back to direct Agate/Wiki access.
    """
    endpoint = _runtime_endpoint(path)
    if endpoint is None:
        return None
    token = os.environ[RUNTIME_TOKEN_ENV].strip()
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            raw = response.read(MAX_RUNTIME_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(MAX_RUNTIME_RESPONSE_BYTES + 1)
        finally:
            exc.close()
        try:
            result = json.loads(raw) if len(raw) <= MAX_RUNTIME_RESPONSE_BYTES else None
        except (ValueError, UnicodeError):
            result = None
        if isinstance(result, dict) and isinstance(result.get("error"), dict):
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        else:
            _client_error(
                f"supervisor runtime rejected the request: HTTP {exc.code}",
                code="runtime_http_error",
            )
        return 2 if exc.code in {400, 404, 413, 422} else 75
    except (OSError, urllib.error.URLError) as exc:
        _client_error(f"supervisor runtime unavailable: {exc}", code="runtime_unavailable")
        return 75
    if len(raw) > MAX_RUNTIME_RESPONSE_BYTES:
        _client_error("supervisor runtime response exceeded the client limit")
        return 2
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        _client_error("supervisor runtime returned invalid JSON")
        return 2
    if not isinstance(result, dict):
        _client_error("supervisor runtime returned a non-object response")
        return 2
    stdout = result.get("stdout")
    stderr = result.get("stderr")
    if isinstance(stdout, str) and stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if isinstance(stderr, str) and stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    truncated = result.get("truncated")
    if isinstance(truncated, dict):
        omitted_stdout = truncated.get("stdout_bytes_omitted")
        omitted_stderr = truncated.get("stderr_bytes_omitted")
        details = []
        if isinstance(omitted_stdout, int) and omitted_stdout > 0:
            details.append(f"stdout={omitted_stdout} bytes")
        if isinstance(omitted_stderr, int) and omitted_stderr > 0:
            details.append(f"stderr={omitted_stderr} bytes")
        if details:
            print(
                "[supervisor runtime] output was context-bounded; omitted "
                + ", ".join(details)
                + "; the complete response remains in private evidence",
                file=sys.stderr,
            )
    exit_code = result.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        _client_error("supervisor runtime response omitted a valid exit_code")
        return 2
    return exit_code


def proxy_gateway(argv: list[str]) -> int | None:
    return proxy_command("/v1/gateway/execute", {"argv": argv})


def proxy_wiki(tool: str, argv: list[str]) -> int | None:
    return proxy_command("/v1/wiki/query", {"tool": tool, "argv": argv})


def proxy_journal(operation: str, **payload: Any) -> int | None:
    return proxy_command(
        "/v1/journal/execute",
        {"operation": operation, **payload},
    )



_REQUEST_KINDS = {
    "update-direction": "direction_update",
    "record-experiment": "experiment_record",
    "episode-report": "episode_report",
}
_LIST_KINDS = {
    "list-directions": "directions_list",
    "list-experiments": "experiments_list",
}
_LOAD_KINDS = {
    "load-direction": ("direction_load", "direction_id"),
    "load-experiment": ("experiment_load", "experiment_id"),
}
_JOURNAL_KINDS = (*_REQUEST_KINDS, *_LIST_KINDS, *_LOAD_KINDS)
_WIKI_KINDS = {
    "wiki-query": "query_nl",
    "wiki-search": "query_wiki",
    "wiki-hardware": "query_hardware",
}


def _journal_request(kind: str, argv: list[str]) -> tuple[str, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--kind", choices=_JOURNAL_KINDS, required=True)
    if kind in _REQUEST_KINDS:
        parser.add_argument("--request-file", required=True, help="JSON request file.")
    elif kind in _LIST_KINDS:
        parser.add_argument(
            "--output-path", required=True, help="Workspace-relative scratch/ destination."
        )
    else:
        parser.add_argument("--record-id", required=True, help="Direction or Experiment ID.")
    args = parser.parse_args(argv)
    if kind in _REQUEST_KINDS:
        try:
            request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(f"request file is missing or invalid: {args.request_file}: {exc}")
        if not isinstance(request, dict):
            parser.error("request file must contain one JSON object")
        return _REQUEST_KINDS[kind], {"request": request}
    if kind in _LIST_KINDS:
        return _LIST_KINDS[kind], {"file": args.output_path}
    operation, field = _LOAD_KINDS[kind]
    return operation, {field: args.record_id}


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--kind", default="auto")
    # A Dev command may have its own --kind/--help flags. Never inspect past --.
    option_arguments = arguments[: arguments.index("--")] if "--" in arguments else arguments
    selected, remaining = selector.parse_known_args(option_arguments)
    if selected.kind in _JOURNAL_KINDS:
        operation, payload = _journal_request(selected.kind, arguments)
        result = proxy_journal(operation, **payload)
    elif selected.kind in _WIKI_KINDS:
        # Strip only our selector; the Supervisor-owned Wiki CLI validates its
        # own flags. Preserve -- and any following literal query arguments.
        result = proxy_wiki(
            _WIKI_KINDS[selected.kind], remaining + arguments[len(option_arguments):]
        )
    elif selected.kind == "auto" and any(flag in option_arguments for flag in ("-h", "--help")):
        print(
            "Usage: python3 tools/sandbox.py --kind OPERATION [options]\n\n"
            "GPU: run, profile, check, disassemble, dev, env\n"
            "Wiki: wiki-query (natural language), wiki-search (structured experience), "
            "wiki-hardware (hardware facts)\n"
            "Gateway/Kernel history: record-read --record-id ID "
            "[--view source|gateway-records] [--output-path scratch/FILE]\n"
            "Journal writes: update-direction, record-experiment, episode-report "
            "--request-file FILE\n"
            "Journal indexes: list-directions, list-experiments --output-path scratch/FILE\n"
            "Journal reads: load-direction, load-experiment --record-id ID\n\n"
            "Use --kind OPERATION --help for operation-specific options. "
            "Operations require a Supervisor Runtime capability."
        )
        return 0
    else:
        result = proxy_gateway(arguments)
    if result is None:
        _client_error(
            "sandbox: a Supervisor Runtime capability is required; direct GPU or Wiki access "
            "is unavailable. "
            "Ask the operator to start or restore this Agent session.",
            code="runtime_capability_missing",
        )
        return 75
    return result


if __name__ == "__main__":
    raise SystemExit(main())
