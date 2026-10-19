"""Gateway response classification."""
from __future__ import annotations

from dataclasses import dataclass
import json

import re

import subprocess

import sys

from pathlib import Path

from typing import Callable

PROTOCOL_FAILURE_SCHEMA = "optimization.gateway-protocol-failure/v1"
FAILURE_PREFIX = "[sandbox] gateway_protocol_failure="
RESULT_PREFIX = "[test_kernel] RESULT_JSON="

SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class GatewayOutcome:
    """One evaluator outcome assembled from every safe transport diagnostic."""

    has_result: bool
    primary_diagnostic: dict[str, object] | None
    secondary_diagnostics: tuple[dict[str, object], ...] = ()

    def failure_record(self) -> dict[str, object] | None:
        """Return the primary failure with its causal context preserved."""
        if self.has_result or self.primary_diagnostic is None:
            return None
        record = dict(self.primary_diagnostic)
        record["candidate_repairable"] = bool(
            record.get("failure_class") == "request"
            and not record.get("reference_rejected")
            and (record.get("candidate_rejected") or record.get("io_contract_rejected"))
        )
        if self.secondary_diagnostics:
            record["secondary_diagnostics"] = [
                dict(item) for item in self.secondary_diagnostics
            ]
        return record


def classify_gateway_outcome(
    process: subprocess.CompletedProcess[str],
) -> GatewayOutcome:
    """Classify a complete gateway attempt chain through one stable interface."""
    has_result = any(
        line.startswith(RESULT_PREFIX)
        for line in (process.stdout or "").splitlines()
    )
    diagnostics: list[dict[str, object]] = []
    for line in (process.stderr or "").splitlines():
        if not line.startswith(FAILURE_PREFIX):
            continue
        try:
            payload = json.loads(line[len(FAILURE_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            diagnostics.append(payload)
    if has_result or not diagnostics:
        return GatewayOutcome(has_result, None)
    primary_index = len(diagnostics) - 1
    for classes in (
        {"cancelled"},
        {"authentication"},
        {"request"},
        {"infrastructure"},
    ):
        decisive = next(
            (
                index
                for index, item in enumerate(diagnostics)
                if item.get("failure_class") in classes
            ),
            None,
        )
        if decisive is not None:
            primary_index = decisive
            break
    return GatewayOutcome(
        False,
        diagnostics[primary_index],
        tuple(item for index, item in enumerate(diagnostics) if index != primary_index),
    )

def terminal_cancellation(job: dict | None) -> bool:
    """An explicit cancellation outcome must survive evaluator redaction."""
    return bool(job and job.get("status") == "cancelled"
                and (job.get("result") or job.get("error")))

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
    try:
        job = json.loads(process.stdout or "")
    except (TypeError, json.JSONDecodeError):
        job = None
    if isinstance(job, dict) and terminal_cancellation(job):
        diagnostic = {
            "schema": PROTOCOL_FAILURE_SCHEMA,
            "agate_return_code": int(process.returncode),
            "response_type": "json-object",
            "reason": "terminal-cancelled",
            "failure_class": "cancelled",
        }
        job_id = _safe_job_id(process, recover)
        if job_id is not None:
            diagnostic["job_id"] = job_id
        return diagnostic
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
