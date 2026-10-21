"""Operator-only evidence saved before the numerical receipt is projected."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .measurement_records import JOB_ROOT_ENV, private_write

MAX_STDOUT_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024


def _bounded(value, limit):
    data = str(value or "").encode("utf-8", errors="replace")
    if len(data) <= limit:
        text = data.decode("utf-8")
    else:
        marker = b"\n<diagnostic truncated>\n"
        budget = max(0, limit - len(marker))
        head, tail = budget // 2, budget - budget // 2
        text = (data[:head].decode("utf-8", errors="ignore") + marker[:limit].decode()
                + (data[-tail:].decode("utf-8", errors="ignore") if tail else ""))
    return {"text": text, "bytes": len(data), "truncated": len(data) > limit}


def record_probe_diagnostics(job: dict) -> None:
    """Do not publish this file to scratch, Agent results, or record-read.

    JOB_ROOT_ENV is injected by RecordedExperiments and resides outside the
    Agent workspace. Standalone commands without that authority do not write.
    Save malformed/truncated receipts and failed jobs too, not only successful
    JSON: those are the cases for which raw evidence is most useful.
    """
    root = os.environ.get(JOB_ROOT_ENV)
    if not root:
        return
    result = job.get("result")
    result = result if isinstance(result, dict) else {}
    private_write(Path(root) / "numerical-diagnostics.json", {
        "schema_version": 1,
        "visibility": "operator-private",
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "exit_code": result.get("exit_code"),
        "gateway_error": _bounded(json.dumps(job.get("error"), ensure_ascii=False), MAX_STDERR_BYTES),
        "stdout": _bounded(result.get("stdout"), MAX_STDOUT_BYTES),
        "stderr": _bounded(result.get("stderr"), MAX_STDERR_BYTES),
    })
