"""Gateway retry and worker dependency configuration."""
from __future__ import annotations
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass

from aka.contracts.execution import JobResult


@dataclass
class TransportCapture:
    args: object
    returncode: int
    stdout: str
    stderr: str
    outcome: JobResult


def _workspace_arg(argv):
    options = argv[:argv.index("--")] if "--" in argv else argv
    for index, value in enumerate(options):
        if value == "--workspace" and index + 1 < len(options):
            return Path(options[index+1]).resolve()
        if value.startswith("--workspace="):
            return Path(value.split("=", 1)[1]).resolve()
    return None
FAILURE_FILE = ".atrex_long_horizon/ci_gateway_failure.json"

UPSTREAM_EVAL_RETRIES_EXHAUSTED = (
    "[sandbox] eval retries exhausted; dev fallback disabled"
)

TRANSIENT_MARKERS = (
    "generalized gateway response unavailable",
    "generalized evaluation failed",
    "sandbox response did not contain an artifact frame",
    "502 bad gateway",
    "503 service temporarily unavailable",
    "gateway http 502",
    "gateway http 503",
    "gateway http 504",
    "ray job submit failed",
    "probe worker discovery failed",
    "dashboard_unreachable",
    "submit_failed",
    "connection reset",
    "context deadline exceeded",
)

TYPED_DELAYS = ()

DEV_DELAYS = ()

def _patch_dev_command_script(command_path: Path, workspace: Path | None) -> None:
    """Map the same Task requirements to dev's raw command transport."""
    import os, shlex
    requirements = os.environ.get("ATREX_SANDBOX_REQUIREMENTS_FILE")
    if requirements:
        from .sandbox import (
            _candidate_languages,
            _json_object,
            _runtime_requirements,
        )

        solution = (
            _json_object(workspace / "solution.json") or {}
            if workspace is not None
            else {}
        )
        candidate_sources = (
            tuple(
                path.read_text(encoding="utf-8")
                for path in (workspace / "kernel.py", workspace / "test_kernel.py")
                if path.is_file()
            )
            if workspace is not None
            else ()
        )
        dependencies = _runtime_requirements(
            requirements,
            candidate_languages=_candidate_languages(
                solution, candidate_sources=candidate_sources
            ),
        )
    else:
        dependencies = json.loads(os.environ.get("AKA_GATEWAY_DEV_DEPENDENCIES", "[]"))
    if not dependencies:
        return
    source = command_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    install = "python3 -m pip install --disable-pip-version-check --no-cache-dir --no-deps " + shlex.join(dependencies)
    command_path.write_text("\n".join([lines[0], lines[1], install, *lines[2:]]) + "\n", encoding="utf-8")

def _patch_dev_command_file(agate: list[str], workspace: Path | None) -> None:
    for index, value in enumerate(agate):
        if value == "__atrex_command.sh" and index + 1 < len(agate):
            value = agate[index + 1]
        elif value.startswith("__atrex_command.sh="):
            value = value.split("=", 1)[1]
        else:
            continue
        _patch_dev_command_script(Path(value), workspace)
        return

def _kind(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "--kind" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--kind="):
            return value.split("=", 1)[1]
    return "auto"

def _failed_transiently(process: subprocess.CompletedProcess[str]) -> bool:
    from .protocol import classify_gateway_outcome, terminal_cancellation

    outcome = classify_gateway_outcome(process)
    if process.returncode == 0 or outcome.has_result:
        return False
    try:
        job = json.loads(process.stdout or "")
    except (TypeError, json.JSONDecodeError):
        job = None
    if isinstance(job, dict) and terminal_cancellation(job):
        return False
    output = ((process.stdout or "") + "\n" + (process.stderr or "")).lower()
    if UPSTREAM_EVAL_RETRIES_EXHAUSTED in output:
        return False
    diagnostic = outcome.failure_record()
    if diagnostic and diagnostic.get("failure_class") in {"request", "authentication", "cancelled"}:
        return False
    # A result-less transport error alone does not prove that submission failed.
    # Retry only a backend-confirmed terminal job or explicit admission rejection.
    if not isinstance(process, TransportCapture):
        return False
    if process.outcome.facts.disposition != "not_submitted":
        return False
    return any(marker in output for marker in TRANSIENT_MARKERS)

def _record_gateway_failure(argv: list[str], process: subprocess.CompletedProcess[str]) -> None:
    from .protocol import classify_gateway_outcome

    workspace = _workspace_arg(argv)
    if workspace is None:
        return
    path = workspace / FAILURE_FILE
    outcome = classify_gateway_outcome(process)
    if process.returncode == 0 or outcome.has_result:
        path.unlink(missing_ok=True)
        return
    diagnostic = outcome.failure_record()
    if diagnostic is None and not (
        _failed_transiently(process)
        or UPSTREAM_EVAL_RETRIES_EXHAUSTED in (process.stderr or "")
    ):
        return
    diagnostic = dict(diagnostic or {})
    # Preserve the exact single-diagnostic record produced by older adapters.
    # The enriched field is required when several attempts form one causal chain.
    if not outcome.secondary_diagnostics:
        diagnostic.pop("candidate_repairable", None)
    diagnostic.setdefault("failure_class", "infrastructure")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(diagnostic) + "\n")
    temporary.replace(path)

def run_with_recovery(
    argv: list[str],
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    sleep: Callable[[float], None] = time.sleep,
    typed_delays: tuple[float, ...] = TYPED_DELAYS,
    dev_delays: tuple[float, ...] = DEV_DELAYS,
) -> subprocess.CompletedProcess[str]:
    """Dispatch once; original-job reads and verified stable-key replay own retries.

    Retain the historical call signature for external consumers. A whole launcher
    retry cannot safely bind admission waiting without truncating measurement
    budgets, and plain CLI submission provides no stable idempotency key.
    """
    return runner(argv)
