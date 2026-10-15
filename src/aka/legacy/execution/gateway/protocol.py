"""Gateway response classification."""
from __future__ import annotations

import json

import re

import subprocess

import sys

from pathlib import Path

from typing import Callable

PROTOCOL_FAILURE_SCHEMA = "optimization.gateway-protocol-failure/v1"

SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

def _response_failure(stdout: str) -> tuple[str, str] | None:
    if not stdout.strip():
        return "empty", "empty-stdout"
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return "text", "invalid-json"
    if not isinstance(payload, dict):
        return f"json-{type(payload).__name__}", "json-not-object"
    if "job_id" not in payload:
        return "json-object", "missing-job-id"
    if not isinstance(payload.get("job_id"), str):
        return "json-object", "job-id-not-string"
    return None

def _safe_job_id(
    process: subprocess.CompletedProcess[str],
    recover: Callable[[subprocess.CompletedProcess[str]], str | None],
) -> str | None:
    try:
        payload = json.loads(process.stdout or "")
    except (TypeError, json.JSONDecodeError):
        payload = None
    candidate = payload.get("job_id") if isinstance(payload, dict) else None
    if not isinstance(candidate, str):
        candidate = recover(process)
    if isinstance(candidate, str) and SAFE_JOB_ID.fullmatch(candidate):
        return candidate
    return None

def protocol_failure(
    process: subprocess.CompletedProcess[str],
    recover: Callable[[subprocess.CompletedProcess[str]], str | None],
) -> dict[str, object] | None:
    """Classify an invalid Agate response without retaining response contents."""
    failure = _response_failure(process.stdout or "")
    if failure is None and process.returncode != 0:
        failure = ("json-object", "nonzero-return-code")
    if failure is None:
        return None
    response_type, reason = failure
    diagnostic: dict[str, object] = {
        "schema": PROTOCOL_FAILURE_SCHEMA,
        "agate_return_code": int(process.returncode),
        "response_type": response_type,
        "reason": reason,
        "failure_class": "protocol",
    }
    # Agate writes HTTP admission errors to stderr, leaving stdout empty.
    # Retain only status and fixed categories; evaluator payloads stay private.
    http_error = re.search(r"agate:\s*\[(\d{3})\]", process.stderr or "")
    if http_error:
        status = int(http_error.group(1))
        diagnostic["http_status"] = status
        diagnostic["failure_class"] = (
            "infrastructure" if status in {408, 429} or status >= 500
            else "authentication" if status in {401, 403}
            else "request"
        )
        stderr = process.stderr or ""
        if "source validation failed" in stderr:
            diagnostic["source_validation_failed"] = True
            diagnostic["candidate_rejected"] = bool(re.search(r"- candidate:", stderr))
            diagnostic["reference_rejected"] = bool(re.search(r"- reference[.:]", stderr))
            diagnostic["io_contract_rejected"] = "- io_contract:" in stderr
            diagnostic["source_policy_rules"] = [
                code for message, code in (
                    ("Forbidden name: getattr", "forbidden-getattr"),
                    ("Blocked import: importlib", "blocked-importlib"),
                ) if message in stderr
            ]
    job_id = _safe_job_id(process, recover)
    if job_id is not None:
        diagnostic["job_id"] = job_id
    return diagnostic
