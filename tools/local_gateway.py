#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trusted localhost scheduler compatible with the public agate jobs API.

The server intentionally implements the subset used by ``tools/sandbox.py``:

* ``GET /healthz``
* ``GET /v1/env`` and ``GET /v1/env/local``
* ``POST /v1/jobs/eval`` (``agate run``)
* ``POST /v1/jobs/profile``
* ``POST /v1/jobs/dev``
* ``GET /v1/jobs`` and ``GET /v1/jobs/<job_id>``
* ``POST /v1/jobs/<job_id>/cancel``

Jobs are persisted in SQLite and consumed FIFO.  The default concurrency is
one, so commands targeting a single local GPU are serialized.  The process is
an interface-compatible execution adapter, not a security sandbox: submitted
commands run as the server user and must be trusted.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
VALID_STATUSES = frozenset({"queued", "running", *TERMINAL_STATUSES})
SUPPORTED_KINDS = frozenset({"eval", "profile", "dev"})
KNOWN_KINDS = frozenset({"eval", "profile", "dev", "disassemble"})
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_BODY_LIMIT = 32 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 32 * 1024 * 1024
DEFAULT_JOB_TIMEOUT = 600
MAX_JOB_TIMEOUT = 600
MAX_SOURCE_BYTES = 24 * 1024 * 1024
MAX_PROFILE_COUNTERS = 256
PROFILE_LEVELS = frozenset({"survey", "sol", "deep"})
PROFILE_TOOLS = frozenset({"ncu", "rocprofv3"})


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _error(reason: str, message: str, trace_id: str | None = None, **details: Any) -> dict[str, Any]:
    return {
        "error_class": "local_gateway",
        "reason": reason,
        "message": message,
        "details": details,
        "trace_id": trace_id,
    }


def _job_prefix(kind: str) -> str:
    return {"dev": "dv", "eval": "ev", "profile": "pf", "disassemble": "ds"}.get(kind, "jb")


class JobStore:
    """Thread-safe SQLite persistence for the FIFO scheduler."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        path.chmod(0o600)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    trace_id TEXT,
                    user_name TEXT,
                    idempotency_key TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    pid INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                )
                """
            )
            self._db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency
                ON jobs(kind, idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            now = time.time()
            restart_error = _json_dumps(
                _error("scheduler_restarted", "local scheduler stopped while the job was running")
            )
            self._db.execute(
                """
                UPDATE jobs
                SET status='failed', error_json=?, updated_at=?, finished_at=?, pid=NULL
                WHERE status='running'
                """,
                (restart_error, now, now),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def create(self, kind: str, request: dict[str, Any], trace_id: str) -> tuple[dict[str, Any], bool]:
        idempotency_key = request.get("idempotency_key")
        with self._lock:
            if idempotency_key:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE kind=? AND idempotency_key=?",
                    (kind, idempotency_key),
                ).fetchone()
                if row is not None:
                    return self._view(row), False

            now = time.time()
            job_id = f"{_job_prefix(kind)}_{secrets.token_hex(6)}"
            self._db.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, request_json, trace_id, user_name,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    _json_dumps(request),
                    trace_id,
                    request.get("author"),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            self._db.commit()
            return self.get(job_id), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._view(row) if row is not None else None

    def request(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT request_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return _json_loads(row[0]) if row is not None else None

    def list(
        self,
        *,
        kind: str | None = None,
        user: str | None = None,
        status: str | None = None,
        limit: int = 50,
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (("kind", kind), ("user_name", user), ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "DESC" if descending else "ASC"
        values.append(limit)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at {order} LIMIT ?", values
            ).fetchall()
            return [self._view(row) for row in rows]

    def claim_next(self) -> tuple[str, str, dict[str, Any]] | None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT job_id, kind, request_json FROM jobs "
                "WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                self._db.commit()
                return None
            now = time.time()
            changed = self._db.execute(
                """
                UPDATE jobs SET status='running', attempts=attempts+1,
                    started_at=?, updated_at=?
                WHERE job_id=? AND status='queued'
                """,
                (now, now, row["job_id"]),
            ).rowcount
            self._db.commit()
            if changed != 1:
                return None
            return row["job_id"], row["kind"], _json_loads(row["request_json"])

    def set_pid(self, job_id: str, pid: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET pid=?, updated_at=? WHERE job_id=? AND status='running'",
                (pid, time.time(), job_id),
            )
            self._db.commit()

    def complete(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> bool:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"terminal status required, got {status!r}")
        now = time.time()
        with self._lock:
            changed = self._db.execute(
                """
                UPDATE jobs SET status=?, result_json=?, error_json=?, pid=NULL,
                    updated_at=?, finished_at=?
                WHERE job_id=? AND status='running'
                """,
                (
                    status,
                    _json_dumps(result) if result is not None else None,
                    _json_dumps(error) if error is not None else None,
                    now,
                    now,
                    job_id,
                ),
            ).rowcount
            self._db.commit()
            return changed == 1

    def cancel(self, job_id: str) -> tuple[dict[str, Any] | None, bool]:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None, False
            was_running = row["status"] == "running"
            if row["status"] not in TERMINAL_STATUSES:
                now = time.time()
                self._db.execute(
                    """
                    UPDATE jobs SET status='cancelled', pid=NULL, updated_at=?, finished_at=?
                    WHERE job_id=?
                    """,
                    (now, now, job_id),
                )
                self._db.commit()
                row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._view(row), was_running

    def fail_running(self, reason: str, message: str) -> None:
        now = time.time()
        error = _json_dumps(_error(reason, message))
        with self._lock:
            self._db.execute(
                """
                UPDATE jobs SET status='failed', error_json=?, pid=NULL,
                    updated_at=?, finished_at=? WHERE status='running'
                """,
                (error, now, now),
            )
            self._db.commit()

    @staticmethod
    def _view(row: sqlite3.Row) -> dict[str, Any]:
        request = _json_loads(row["request_json"])
        mode_enforced = None
        if row["kind"] == "eval":
            mode_enforced = request.get("mode", "full") in {"full", "correctness_only"}
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "mode": request.get("mode"),
            "lock_clocks": request.get("lock_clocks"),
            "mode_enforced": mode_enforced,
            "result": _json_loads(row["result_json"]),
            "error": _json_loads(row["error_json"]),
            "trace_id": row["trace_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _safe_destination(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() in ("", "."):
        raise ValueError(f"file destination must be a safe relative path: {value!r}")
    return path


def _validate_payload_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _validate_target(payload: dict[str, Any]) -> None:
    spec = payload.get("spec")
    targets = spec.get("target_hardware") if isinstance(spec, dict) else None
    if not isinstance(targets, list) or not targets or not all(isinstance(v, str) for v in targets):
        raise ValueError("spec.target_hardware must be a non-empty string array")
    num_gpus = spec.get("num_gpus", 1)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("spec.num_gpus must be a positive integer")
    if num_gpus > 1:
        raise ValueError("localhost scheduler does not support spec.num_gpus greater than 1")


def _validate_timeout(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_JOB_TIMEOUT:
        raise ValueError(f"{field} must be an integer in the range 1..{MAX_JOB_TIMEOUT}")
    return value


def _validate_env_vars(value: Any) -> dict[str, str]:
    env_vars = value or {}
    if not isinstance(env_vars, dict) or not all(
        isinstance(k, str) and ENV_NAME_RE.fullmatch(k) and isinstance(v, str)
        for k, v in env_vars.items()
    ):
        raise ValueError("env_vars must map valid environment names to strings")
    return env_vars


def _validate_idempotency_key(payload: dict[str, Any]) -> None:
    idempotency_key = payload.get("idempotency_key")
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not idempotency_key or len(idempotency_key) > 256
    ):
        raise ValueError("idempotency_key must be a non-empty string of at most 256 characters")


def _validate_requirements(value: Any) -> list[str]:
    requirements = value or []
    if not isinstance(requirements, list) or len(requirements) > 128 or not all(
        isinstance(item, str) and item.strip() and len(item) <= 2048
        for item in requirements
    ):
        raise ValueError("requirements must be an array of at most 128 non-empty strings")
    return requirements


def _validate_dev_request(payload: Any) -> dict[str, Any]:
    payload = _validate_payload_object(payload)
    _validate_target(payload)
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if len(command.encode("utf-8")) > 1024 * 1024:
        raise ValueError("command exceeds the 1 MiB limit")

    _validate_timeout(payload.get("timeout_s", DEFAULT_JOB_TIMEOUT), "timeout_s")
    _validate_env_vars(payload.get("env_vars"))

    files = payload.get("files") or {}
    if not isinstance(files, dict) or len(files) > 128:
        raise ValueError("files must be an object with at most 128 entries")
    total_bytes = 0
    for name, content in files.items():
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError("files must map relative path strings to text contents")
        _safe_destination(name)
        total_bytes += len(content.encode("utf-8"))
    if total_bytes > 24 * 1024 * 1024:
        raise ValueError("uploaded files exceed the 24 MiB request limit")

    _validate_idempotency_key(payload)
    return payload


def _number_option(options: dict[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"options.{name} must be a finite number")
    if value < 0:
        raise ValueError(f"options.{name} must be non-negative")
    return float(value)


def _validate_typed_request(payload: Any, kind: str) -> dict[str, Any]:
    payload = _validate_payload_object(payload)
    _validate_target(payload)

    candidate = payload.get("candidate")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("candidate must be a non-empty Python source string")

    reference = payload.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    operator = reference.get("operator")
    if not isinstance(operator, str) or not operator.strip() or len(operator) > 512:
        raise ValueError("reference.operator must be a non-empty string")
    for field in ("reference_py", "input_py"):
        value = reference.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"reference.{field} must be a non-empty Python source string")
    shapes = reference.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("reference.shapes must be a non-empty object")
    for field in ("metadata", "roofline"):
        if field in reference and not isinstance(reference[field], dict):
            raise ValueError(f"reference.{field} must be an object")

    source_bytes = sum(
        len(value.encode("utf-8"))
        for value in (candidate, reference["reference_py"], reference["input_py"])
    )
    source_bytes += len(_json_dumps(shapes).encode("utf-8"))
    source_bytes += len(_json_dumps(reference.get("metadata", {})).encode("utf-8"))
    source_bytes += len(_json_dumps(reference.get("roofline", {})).encode("utf-8"))
    if source_bytes > MAX_SOURCE_BYTES:
        raise ValueError("candidate and reference bundle exceed the 24 MiB limit")

    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    num_cases = options.get("num_correctness_cases", 1)
    if isinstance(num_cases, bool) or not isinstance(num_cases, int) or not 1 <= num_cases <= 1000:
        raise ValueError("options.num_correctness_cases must be an integer in 1..1000")
    bench_iters = options.get("bench_iters", 100)
    if isinstance(bench_iters, bool) or not isinstance(bench_iters, int) or not 1 <= bench_iters <= 1_000_000:
        raise ValueError("options.bench_iters must be an integer in 1..1000000")
    _number_option(options, "atol", 1e-2)
    _number_option(options, "rtol", 0.05)
    if "correctness_max_rel_l2" in options:
        _number_option(options, "correctness_max_rel_l2", 0.0)
    _validate_timeout(options.get("timeout_s", DEFAULT_JOB_TIMEOUT), "options.timeout_s")

    _validate_env_vars(payload.get("env_vars"))
    _validate_requirements(payload.get("requirements"))
    _validate_idempotency_key(payload)
    lock_clocks = payload.get("lock_clocks", False)
    if not isinstance(lock_clocks, bool):
        raise ValueError("lock_clocks must be a boolean")

    if kind == "eval":
        mode = payload.get("mode", "full")
        if mode not in {"full", "correctness_only"}:
            raise ValueError("mode must be 'full' or 'correctness_only'")
    elif kind == "profile":
        level = payload.get("level", "sol")
        if level not in PROFILE_LEVELS:
            raise ValueError("level must be one of: survey, sol, deep")
        profiler = payload.get("profiler")
        if profiler is not None and profiler not in PROFILE_TOOLS:
            raise ValueError("profiler must be 'ncu' or 'rocprofv3'")
        counters = payload.get("counters") or []
        if not isinstance(counters, list) or len(counters) > MAX_PROFILE_COUNTERS or not all(
            isinstance(counter, str) and counter.strip() and len(counter) <= 512
            for counter in counters
        ):
            raise ValueError(
                f"counters must be an array of at most {MAX_PROFILE_COUNTERS} non-empty strings"
            )
        kernel_regex = payload.get("kernel_regex")
        if kernel_regex is not None:
            if not isinstance(kernel_regex, str) or not kernel_regex or len(kernel_regex) > 1024:
                raise ValueError("kernel_regex must be a non-empty string of at most 1024 characters")
            try:
                re.compile(kernel_regex)
            except re.error as exc:
                raise ValueError(f"kernel_regex is invalid: {exc}") from exc
        if level == "deep" and not kernel_regex:
            raise ValueError("level='deep' requires kernel_regex")
        top_kernels = payload.get("top_kernels")
        if top_kernels is not None and (
            isinstance(top_kernels, bool)
            or not isinstance(top_kernels, int)
            or not 1 <= top_kernels <= 1000
        ):
            raise ValueError("top_kernels must be an integer in 1..1000")
    else:
        raise ValueError(f"unsupported typed kind {kind!r}")
    return payload


def _find_atrex_bench_runner(explicit_root: Path | None = None) -> Path:
    roots: list[Path] = []
    if explicit_root is not None:
        roots.append(explicit_root)
    configured = os.environ.get("ATREX_BENCH_ROOT")
    if configured:
        roots.append(Path(configured))
    try:
        spec = importlib.util.find_spec("atrex_bench")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        roots.append(package_dir.parent.parent)
    for root in roots:
        runner = root.expanduser().resolve() / "scripts" / "run_eval.py"
        if runner.is_file():
            return runner
    searched = ", ".join(str(root) for root in roots) or "the active Python environment"
    raise FileNotFoundError(
        "atrex-bench scripts/run_eval.py was not found; install atrex-bench editable "
        f"or set ATREX_BENCH_ROOT (searched {searched})"
    )


CORRECTNESS_ONLY_RUNNER = r'''#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-eval", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--atol", type=float, required=True)
parser.add_argument("--rtol", type=float, required=True)
parser.add_argument("--correctness-max-rel-l2", type=float)
parser.add_argument("--num-correctness-cases", type=int, required=True)
parser.add_argument("--candidate-timeout-s", type=float, required=True)
parser.add_argument("--clock-locked", action="store_true")
parser.add_argument("--shape-id")
args = parser.parse_args()
if args.correctness_max_rel_l2 is not None:
    os.environ["ATREX_CORRECTNESS_MAX_REL_L2"] = str(args.correctness_max_rel_l2)

spec = importlib.util.spec_from_file_location("atrex_local_run_eval", args.run_eval)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import evaluator: {args.run_eval}")
run_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_eval
spec.loader.exec_module(run_eval)
run_eval = sys.modules.get(spec.name, run_eval)

if args.shape_id is not None:
    result = run_eval.check_correctness(
        args.reference / "reference.py",
        args.candidate,
        shape_id=args.shape_id,
        atol=args.atol,
        rtol=args.rtol,
        num_correctness_cases=args.num_correctness_cases,
        candidate_timeout_s=args.candidate_timeout_s,
    )
    target = args.output / ".correctness_only_shapes" / f"{args.shape_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run_eval._correctness_to_payload(result)), encoding="utf-8")
    raise SystemExit(0)

def correctness_only_shape(**kwargs):
    shape_id = str(kwargs["shape_id"])
    result_path = args.output / ".correctness_only_shapes" / f"{shape_id}.json"
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-eval", str(args.run_eval),
        "--candidate", str(args.candidate),
        "--reference", str(args.reference),
        "--output", str(args.output),
        "--atol", str(kwargs["atol"]),
        "--rtol", str(kwargs["rtol"]),
        "--num-correctness-cases", str(kwargs["num_correctness_cases"]),
        "--candidate-timeout-s", str(kwargs["candidate_timeout_s"]),
        "--shape-id", shape_id,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0 and result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            run_eval._correctness_from_payload(payload),
            run_eval.PerformanceShapeResult(),
            None,
        )
    reason = run_eval._summarize_subworker_failure(
        returncode=completed.returncode, stderr=completed.stderr
    )
    return (
        run_eval.CorrectnessShapeResult(status="failed", reason=reason),
        run_eval.PerformanceShapeResult(),
        None,
    )

run_eval._run_single_shape_subprocess = correctness_only_shape
args.output.mkdir(parents=True, exist_ok=True)
payload = run_eval._run_eval_worker(
    input_path=args.candidate,
    reference_dir=args.reference,
    artifact_dir=args.output,
    atol=args.atol,
    rtol=args.rtol,
    num_correctness_cases=args.num_correctness_cases,
    warmup_iters=0,
    bench_iters=1,
    checkpoint_dir=None,
    config_version="local-gateway-correctness-only-v1",
    clock_locked=args.clock_locked,
    require_clock_locked=False,
    collect_kernel_events=False,
    candidate_timeout_s=args.candidate_timeout_s,
    perf_timeout_s=0,
)
payload["eval_mode"] = "correctness_only"
runner_config = payload.get("runner_config")
if isinstance(runner_config, dict):
    runner_config["mode"] = "correctness_only"
payload["performance"] = {"shapes": {
    str(shape_id): {"input_artifact": None, "samples": [], "kernel_events": [],
                    "benchmark_mode": "eager", "capture_time_ms": None,
                    "cache_flush_mb": None, "graph_correctness": None,
                    "error": None, "observed_kernels": None}
    for shape_id in (payload.get("correctness", {}).get("shapes", {}) or {})
}}
(args.output / "eval_result.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
)
'''


PROFILE_DRIVER = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from atrex_bench.eval._runtime import (
    clone_model_inputs,
    import_module_from_path,
    instantiate_model_module,
    load_shape_call_inputs,
    load_shape_init_inputs,
    load_shape_spec,
    resolve_input_module,
    sync_device,
    validate_reference_module,
)

root = Path(__file__).resolve().parent
reference_path = root / "reference" / "reference.py"
candidate_path = root / "candidate.py"
shapes = json.loads((root / "reference" / "shapes.json").read_text(encoding="utf-8"))
shape_id = os.environ.get("ATREX_PROFILE_SHAPE_ID") or sorted(
    (str(value) for value in shapes), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
)[0]
device = torch.device("cuda")
reference_module = import_module_from_path(reference_path, "atrex_profile_reference")
validate_reference_module(reference_module)
input_module = resolve_input_module(reference_path, reference_module, module_prefix="atrex_profile_input")
shape = load_shape_spec(reference_path, shape_id)
init_inputs = load_shape_init_inputs(shape, device)
candidate = instantiate_model_module(
    candidate_path, device, "atrex_profile_candidate", init_inputs=init_inputs
).model
call_inputs = load_shape_call_inputs(input_module, shape, device)

with torch.inference_mode():
    for _ in range(int(os.environ.get("ATREX_PROFILE_WARMUP", "3"))):
        current = clone_model_inputs(call_inputs)
        candidate(*current.args, **current.kwargs)
    sync_device(device)
    torch.cuda.cudart().cudaProfilerStart()
    current = clone_model_inputs(call_inputs)
    candidate(*current.args, **current.kwargs)
    sync_device(device)
    torch.cuda.cudart().cudaProfilerStop()
'''


NCU_SOL_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def _float_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace(",", "").replace("%", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _duration_ns(value: float, unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"s", "second", "seconds"}:
        return value * 1_000_000_000.0
    if normalized in {"ms", "msecond", "millisecond", "milliseconds"}:
        return value * 1_000_000.0
    if normalized in {"us", "usecond", "microsecond", "microseconds"}:
        return value * 1_000.0
    return value


def _parse_ncu_csv(path: Path, *, level: str, top_kernels: int | None) -> dict[str, Any]:
    rows = list(csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    header_index = next(
        (index for index, row in enumerate(rows) if "Kernel Name" in row and "Metric Name" in row),
        None,
    )
    result: dict[str, Any] = {"profiler": "ncu", "level": level, "kernels": []}
    if header_index is None:
        result["note"] = "no kernels captured by ncu"
        return result
    header = [column.strip() for column in rows[header_index]]
    grouped: dict[str, dict[str, Any]] = {}
    for raw in rows[header_index + 1:]:
        if len(raw) < len(header):
            continue
        record = dict(zip(header, raw))
        name = (record.get("Kernel Name") or "").strip()
        metric = (record.get("Metric Name") or "").strip()
        value = _float_value(record.get("Metric Value"))
        if not name or not metric or value is None:
            continue
        unit = (record.get("Metric Unit") or "").strip()
        kernel = grouped.setdefault(
            name,
            {"name": name, "duration": 0.0, "duration_unit": "ns", "_metrics": {}},
        )
        kernel["_metrics"].setdefault(metric, []).append((value, unit))
        if metric.startswith("gpu__time_duration"):
            kernel["duration"] += _duration_ns(value, unit)

    kernels: list[dict[str, Any]] = []
    for kernel in grouped.values():
        raw_metrics = kernel.pop("_metrics")

        def metric_value(prefixes: tuple[str, ...]) -> float | None:
            for metric_name, values in raw_metrics.items():
                if metric_name in prefixes or any(metric_name.startswith(prefix) for prefix in prefixes):
                    return sum(item[0] for item in values) / len(values)
            return None

        compute = metric_value(("sm__throughput", "Compute (SM) Throughput"))
        memory = metric_value(("gpu__compute_memory_throughput", "Memory Throughput"))
        dram = metric_value(("dram__throughput", "DRAM Throughput"))
        occupancy = metric_value(("sm__warps_active", "Achieved Occupancy"))
        if level != "survey":
            kernel.update(
                compute_sol_pct=compute,
                mem_sol_pct=memory,
                dram_pct=dram,
                occupancy_pct=occupancy,
                bound=("compute" if (compute or 0.0) >= (memory or 0.0) else "memory"),
            )
        if level == "deep":
            kernel["diagnostics"] = {
                metric_name: str(values[-1][0])
                for metric_name, values in raw_metrics.items()
                if not metric_name.startswith("gpu__time_duration")
            }
        kernel["metrics"] = {
            metric_name: values[-1][0] for metric_name, values in raw_metrics.items()
        }
        kernels.append(kernel)
    kernels.sort(key=lambda item: float(item.get("duration") or 0.0), reverse=True)
    if top_kernels is not None:
        kernels = kernels[:top_kernels]
    result["kernels"] = kernels
    if not kernels:
        result["note"] = "no kernels captured by ncu"
    return result


def _parse_rocprof_csv(directory: Path, *, level: str, top_kernels: int | None) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except OSError:
            continue
        for row in rows:
            name = next(
                (
                    str(row.get(field) or "").strip()
                    for field in ("Kernel_Name", "Kernel Name", "Name")
                    if str(row.get(field) or "").strip()
                ),
                "",
            )
            if not name:
                continue
            kernel = grouped.setdefault(
                name, {"name": name, "duration": 0.0, "duration_unit": "ns", "metrics": {}}
            )
            duration = _float_value(row.get("Duration"))
            if duration is None:
                start = _float_value(row.get("Start_Timestamp") or row.get("Start_Timestamp_ns"))
                end = _float_value(row.get("End_Timestamp") or row.get("End_Timestamp_ns"))
                duration = end - start if start is not None and end is not None else 0.0
            kernel["duration"] += max(0.0, duration)
            for field, value in row.items():
                number = _float_value(value)
                if number is not None and field not in {
                    "Duration", "Start_Timestamp", "Start_Timestamp_ns",
                    "End_Timestamp", "End_Timestamp_ns",
                }:
                    kernel["metrics"][field] = number
    kernels = sorted(
        grouped.values(), key=lambda item: float(item.get("duration") or 0.0), reverse=True
    )
    if top_kernels is not None:
        kernels = kernels[:top_kernels]
    result: dict[str, Any] = {"profiler": "rocprofv3", "level": level, "kernels": kernels}
    if not kernels:
        result["note"] = "no kernels captured by rocprofv3"
    return result


def _find_profile_tool(name: str, env: dict[str, str]) -> str | None:
    discovered = shutil.which(name, path=env.get("PATH"))
    if discovered:
        return discovered
    configured = env.get("NCU_BIN" if name == "ncu" else "ROCPROFV3_BIN")
    candidates = [Path(configured)] if configured else []
    if name == "ncu":
        candidates.extend((Path("/usr/local/cuda/bin/ncu"),))
        candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/ncu"), reverse=True))
    else:
        candidates.extend((Path("/opt/rocm/bin/rocprofv3"), Path("/usr/bin/rocprofv3")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _auto_profile_tool(env: dict[str, str]) -> str:
    preferred: str | None = None
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print('rocprofv3' if torch.version.hip else 'ncu')",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        value = probe.stdout.strip()
        if value in PROFILE_TOOLS:
            preferred = value
    except (OSError, subprocess.SubprocessError):
        pass
    if preferred is not None:
        if _find_profile_tool(preferred, env):
            return preferred
        raise FileNotFoundError(f"auto-selected profiler {preferred!r} is not available")
    for candidate in ("ncu", "rocprofv3"):
        if _find_profile_tool(candidate, env):
            return candidate
    raise FileNotFoundError("neither ncu nor rocprofv3 is available in PATH")


class LocalScheduler:
    """Persistent FIFO worker pool for commands targeting the local GPU."""

    def __init__(
        self,
        store: JobStore,
        jobs_dir: Path,
        workers: int,
        max_output_bytes: int,
        atrex_bench_root: Path | None = None,
    ) -> None:
        self.store = store
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workers = workers
        self.max_output_bytes = max_output_bytes
        self.atrex_bench_root = atrex_bench_root
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._process_lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def start(self) -> None:
        for number in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"local-gateway-worker-{number}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        self.notify()
        deadline = time.monotonic() + 5
        while any(thread.is_alive() for thread in self._threads) and time.monotonic() < deadline:
            # A worker may have claimed a job just before shutdown. Re-sample the
            # process map so that launch/stop races cannot leave a child behind.
            with self._process_lock:
                processes = list(self._processes.values())
            for process in processes:
                self._terminate(process)
            for thread in self._threads:
                thread.join(timeout=0.1)
        self.store.fail_running("scheduler_stopped", "local scheduler stopped during execution")

    def notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def wait_for_job(self, job_id: str, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                job = self.store.get(job_id)
                if job is None or job["status"] in TERMINAL_STATUSES:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return job
                self._condition.wait(timeout=remaining)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job, was_running = self.store.cancel(job_id)
        if was_running:
            with self._process_lock:
                process = self._processes.get(job_id)
            if process is not None:
                self._terminate(process)
        self.notify()
        return job

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            claimed = self.store.claim_next()
            if claimed is None:
                with self._condition:
                    self._condition.wait(timeout=0.5)
                continue
            job_id, kind, request = claimed
            self.notify()
            if self._stop.is_set():
                self.store.complete(
                    job_id,
                    status="failed",
                    result=None,
                    error=_error("scheduler_stopped", "local scheduler stopped before execution"),
                )
                break
            self._execute(job_id, kind, request)
            self.notify()

    def _execute(self, job_id: str, kind: str, request: dict[str, Any]) -> None:
        workdir = self.jobs_dir / job_id
        clock_reset: list[str] | None = None
        clock_env: dict[str, str] | None = None
        try:
            if self._stop.is_set():
                self.store.complete(
                    job_id,
                    status="failed",
                    result=None,
                    error=_error("scheduler_stopped", "local scheduler stopped before execution"),
                )
                return
            workdir.mkdir(parents=False, exist_ok=False, mode=0o700)
            env = os.environ.copy()
            request_env = request.get("env_vars") or {}
            env.update(request_env)
            if "PATH" not in request_env:
                # Starting the gateway with an explicit environment Python
                # The active interpreter and its sibling tools must make that same
                # toolchain the default for worker commands.  Invoking an
                # interpreter by absolute path does not otherwise update PATH.
                python_bin = str(Path(sys.executable).resolve().parent)
                path_parts = [
                    part for part in env.get("PATH", "").split(os.pathsep)
                    if part and part != python_bin
                ]
                env["PATH"] = os.pathsep.join([python_bin, *path_parts])
            if "TORCH_EXTENSIONS_DIR" not in request_env:
                # Torch's load/load_inline lock file has no owner metadata and
                # survives SIGTERM/SIGKILL.  Reusing the gateway account's
                # global cache can therefore make a later, unrelated job wait
                # forever on a stale lock (or collide with another candidate
                # that chose the same extension name).  A job-local cache
                # matches fresh remote workers and contains all build state
                # inside the job's already-isolated directory.
                torch_extensions = workdir / ".torch_extensions"
                torch_extensions.mkdir(mode=0o700)
                env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions)
            env["ATREX_LOCAL_JOB_ID"] = job_id
            argv, timeout_s = self._prepare_job(kind, request, workdir, env)
            argv = self._install_requirements_argv(argv, request, workdir, env)
            if kind in {"eval", "profile"} and request.get("lock_clocks", False):
                clock_reset = self._lock_clocks(env)
                clock_env = env
            current = self.store.get(job_id)
            if self._stop.is_set():
                self.store.complete(
                    job_id,
                    status="failed",
                    result=None,
                    error=_error("scheduler_stopped", "local scheduler stopped before execution"),
                )
                return
            if current is None or current["status"] != "running":
                return

            stdout_path = workdir / ".stdout"
            stderr_path = workdir / ".stderr"
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                stdout_path.chmod(0o600)
                stderr_path.chmod(0o600)
                process = subprocess.Popen(
                    argv,
                    cwd=workdir,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                with self._process_lock:
                    self._processes[job_id] = process
                self.store.set_pid(job_id, process.pid)
                current = self.store.get(job_id)
                if current is not None and current["status"] == "cancelled":
                    self._terminate(process)
                timed_out = False
                try:
                    process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate(process)

            if clock_reset is not None:
                self._reset_clocks(clock_reset, clock_env)
                clock_reset = None
            current = self.store.get(job_id)
            if current is None or current["status"] == "cancelled":
                return
            stdout, stdout_truncated = self._read_output(stdout_path)
            stderr, stderr_truncated = self._read_output(stderr_path)
            command_result = {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr}
            if timed_out:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=command_result,
                    error=_error(
                        "command_timeout",
                        f"{kind} job exceeded {timeout_s} seconds",
                    ),
                )
            elif stdout_truncated or stderr_truncated:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=command_result,
                    error=_error(
                        "output_too_large",
                        f"command output exceeded the {self.max_output_bytes} byte stream limit",
                    ),
                )
            else:
                self._complete_job(job_id, kind, request, workdir, command_result)
        except Exception as exc:
            self.store.complete(
                job_id,
                status="failed",
                result=None,
                error=_error("execution_error", f"{type(exc).__name__}: {exc}"),
            )
        finally:
            with self._process_lock:
                self._processes.pop(job_id, None)
            if clock_reset is not None:
                try:
                    subprocess.run(
                        clock_reset,
                        env=clock_env,
                        capture_output=True,
                        timeout=20,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass

    def _prepare_job(
        self,
        kind: str,
        request: dict[str, Any],
        workdir: Path,
        env: dict[str, str],
    ) -> tuple[list[str], int]:
        if kind == "dev":
            for name, content in (request.get("files") or {}).items():
                self._write_text(workdir, name, content)
            return ["bash", "-c", request["command"]], int(
                request.get("timeout_s", DEFAULT_JOB_TIMEOUT)
            )

        self._materialize_typed_bundle(request, workdir)
        options = request.get("options") or {}
        timeout_s = int(options.get("timeout_s", DEFAULT_JOB_TIMEOUT))
        if kind == "eval":
            return self._eval_argv(request, workdir, timeout_s), timeout_s
        if kind == "profile":
            return self._profile_argv(request, workdir, env), timeout_s
        raise ValueError(f"unsupported job kind {kind!r}")

    @staticmethod
    def _write_text(workdir: Path, name: str, content: str, *, executable: bool = False) -> Path:
        relative = _safe_destination(name)
        target = workdir.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        target.chmod(0o700 if executable else 0o600)
        return target

    def _materialize_typed_bundle(self, request: dict[str, Any], workdir: Path) -> None:
        reference = request["reference"]
        self._write_text(workdir, "candidate.py", request["candidate"])
        self._write_text(workdir, "reference/reference.py", reference["reference_py"])
        self._write_text(workdir, "reference/input.py", reference["input_py"])
        self._write_text(workdir, "reference/shapes.json", _json_dumps(reference["shapes"]))
        self._write_text(
            workdir, "reference/metadata.json", _json_dumps(reference.get("metadata", {}))
        )
        if "roofline" in reference:
            self._write_text(
                workdir, "reference/roofline.json", _json_dumps(reference["roofline"])
            )

    def _eval_argv(self, request: dict[str, Any], workdir: Path, timeout_s: int) -> list[str]:
        runner = _find_atrex_bench_runner(self.atrex_bench_root)
        options = request.get("options") or {}
        common = [
            "--input", str(workdir / "candidate.py"),
            "--reference-dir", str(workdir / "reference"),
            "--atol", str(options.get("atol", 1e-2)),
            "--rtol", str(options.get("rtol", 0.05)),
            "--num-correctness-cases", str(options.get("num_correctness_cases", 1)),
        ]
        correctness_max_rel_l2 = options.get("correctness_max_rel_l2")
        if correctness_max_rel_l2 is not None:
            common.extend(
                ["--correctness-max-rel-l2", str(correctness_max_rel_l2)]
            )
        if request.get("mode", "full") == "correctness_only":
            wrapper = self._write_text(
                workdir, "correctness_only_runner.py", CORRECTNESS_ONLY_RUNNER, executable=True
            )
            argv = [
                sys.executable,
                str(wrapper),
                "--run-eval", str(runner),
                "--candidate", str(workdir / "candidate.py"),
                "--reference", str(workdir / "reference"),
                "--output", str(workdir / "eval_output"),
                "--atol", str(options.get("atol", 1e-2)),
                "--rtol", str(options.get("rtol", 0.05)),
                "--num-correctness-cases", str(options.get("num_correctness_cases", 1)),
                "--candidate-timeout-s", str(min(60, timeout_s)),
            ]
            if correctness_max_rel_l2 is not None:
                argv.extend(
                    ["--correctness-max-rel-l2", str(correctness_max_rel_l2)]
                )
            if request.get("lock_clocks", False):
                argv.append("--clock-locked")
            return argv

        argv = [
            sys.executable,
            str(runner),
            *common,
            "--output", str(workdir / "eval_output"),
            "--bench-iters", str(options.get("bench_iters", 100)),
        ]
        if request.get("lock_clocks", False):
            argv.append("--clock-locked")
        return argv

    def _profile_argv(
        self, request: dict[str, Any], workdir: Path, env: dict[str, str]
    ) -> list[str]:
        # Validate availability of the evaluator package before starting a profiler whose
        # target driver would otherwise fail with a less useful import error.
        _find_atrex_bench_runner(self.atrex_bench_root)
        driver = self._write_text(workdir, "profile_driver.py", PROFILE_DRIVER, executable=True)
        output_dir = workdir / "profile_output"
        output_dir.mkdir(mode=0o700)
        profiler = request.get("profiler")
        if profiler is None:
            profiler = _auto_profile_tool(env)
        executable = _find_profile_tool(profiler, env)
        if executable is None:
            raise FileNotFoundError(f"requested profiler {profiler!r} is not available in PATH")
        level = request.get("level", "sol")
        counters = request.get("counters") or []
        kernel_regex = request.get("kernel_regex")
        if profiler == "ncu":
            argv = [
                executable,
                "--csv",
                "--log-file", str(output_dir / "ncu.csv"),
                "--profile-from-start", "off",
                "--target-processes", "all",
                "--kernel-name-base", "demangled",
            ]
            if counters:
                argv += ["--metrics", ",".join(counters)]
            elif level == "deep":
                argv += ["--set", "full"]
            else:
                metrics = (NCU_SOL_METRICS[:1] if level == "survey" else NCU_SOL_METRICS)
                argv += ["--metrics", ",".join(metrics)]
            if kernel_regex:
                argv += ["--kernel-name", f"regex:{kernel_regex}"]
            argv += [sys.executable, str(driver)]
            return argv

        argv = [
            executable,
            "--kernel-trace",
            "--output-format", "csv",
            "-d", str(output_dir),
        ]
        if counters:
            argv += ["--pmc", *counters]
        if kernel_regex:
            argv += ["--kernel-include-regex", kernel_regex]
        argv += ["--", sys.executable, str(driver)]
        return argv

    @staticmethod
    def _install_requirements_argv(
        argv: list[str],
        request: dict[str, Any],
        workdir: Path,
        env: dict[str, str],
    ) -> list[str]:
        requirements = request.get("requirements") or []
        if not requirements:
            return argv
        target = workdir / ".requirements"
        target.mkdir(mode=0o700)
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(target) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        install = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(target),
            *requirements,
        ]
        return ["bash", "-c", f"{shlex.join(install)} && exec {shlex.join(argv)}"]

    @staticmethod
    def _lock_clocks(env: dict[str, str]) -> list[str]:
        nvidia_smi = shutil.which("nvidia-smi", path=env.get("PATH"))
        if nvidia_smi:
            query = subprocess.run(
                [
                    nvidia_smi,
                    "--query-supported-clocks=gr",
                    "--format=csv,noheader,nounits",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            clocks = [
                value
                for line in query.stdout.splitlines()
                if (value := _float_value(line)) is not None
            ]
            if not clocks:
                raise RuntimeError("nvidia-smi returned no supported graphics clocks")
            maximum = str(int(max(clocks)))
            subprocess.run(
                [nvidia_smi, "--lock-gpu-clocks", f"{maximum},{maximum}"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            return [nvidia_smi, "--reset-gpu-clocks"]

        rocm_smi = shutil.which("rocm-smi", path=env.get("PATH"))
        if rocm_smi:
            subprocess.run(
                [rocm_smi, "--setperflevel", "high"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            return [rocm_smi, "--setperflevel", "auto"]
        raise FileNotFoundError("lock_clocks requires nvidia-smi or rocm-smi in PATH")

    @staticmethod
    def _reset_clocks(argv: list[str], env: dict[str, str] | None) -> None:
        subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )

    def _complete_job(
        self,
        job_id: str,
        kind: str,
        request: dict[str, Any],
        workdir: Path,
        command_result: dict[str, Any],
    ) -> None:
        if kind == "dev":
            if command_result["exit_code"] == 0:
                self.store.complete(job_id, status="succeeded", result=command_result, error=None)
            else:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=command_result,
                    error=_error(
                        "command_failed",
                        f"command exited with status {command_result['exit_code']}",
                    ),
                )
            return

        if kind == "eval":
            result_paths = sorted((workdir / "eval_output").rglob("eval_result.json"))
            if result_paths:
                try:
                    result = json.loads(result_paths[-1].read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self.store.complete(
                        job_id,
                        status="failed",
                        result=command_result,
                        error=_error("invalid_eval_result", f"cannot read eval_result.json: {exc}"),
                    )
                    return
                if isinstance(result, dict):
                    # Candidate compile/correctness failure is an evaluation outcome, not
                    # gateway infrastructure failure. run_eval intentionally exits nonzero
                    # in that case, while the public job still completes successfully.
                    self.store.complete(job_id, status="succeeded", result=result, error=None)
                    return
            self.store.complete(
                job_id,
                status="failed",
                result=command_result,
                error=_error(
                    "evaluator_failed",
                    f"atrex-bench exited with status {command_result['exit_code']} without eval_result.json",
                ),
            )
            return

        if kind == "profile":
            if command_result["exit_code"] != 0:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=command_result,
                    error=_error(
                        "profiler_failed",
                        f"profiler exited with status {command_result['exit_code']}",
                    ),
                )
                return
            level = request.get("level", "sol")
            top_kernels = request.get("top_kernels")
            profiler = request.get("profiler")
            ncu_csv = workdir / "profile_output" / "ncu.csv"
            if profiler == "ncu" or (profiler is None and ncu_csv.is_file()):
                result = _parse_ncu_csv(ncu_csv, level=level, top_kernels=top_kernels)
            else:
                result = _parse_rocprof_csv(
                    workdir / "profile_output", level=level, top_kernels=top_kernels
                )
            self.store.complete(job_id, status="succeeded", result=result, error=None)
            return
        raise ValueError(f"unsupported job kind {kind!r}")

    def _read_output(self, path: Path) -> tuple[str, bool]:
        size = path.stat().st_size
        with path.open("rb") as stream:
            data = stream.read(self.max_output_bytes)
        return data.decode("utf-8", errors="replace"), size > self.max_output_bytes

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


class LocalGateway:
    def __init__(
        self,
        state_dir: Path,
        *,
        workers: int = 1,
        max_output_bytes: int = DEFAULT_OUTPUT_LIMIT,
        gpu_aliases: tuple[str, ...] = (),
        atrex_bench_root: Path | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._state_lock = (self.state_dir / ".scheduler.lock").open("a+")
        self._state_lock_path = self.state_dir / ".scheduler.lock"
        self._state_lock_path.chmod(0o600)
        try:
            fcntl.flock(self._state_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._state_lock.close()
            raise RuntimeError(f"state directory is already in use: {self.state_dir}") from exc
        self.store = JobStore(self.state_dir / "jobs.db")
        self.gpu_aliases = frozenset(("local", *gpu_aliases))
        self.scheduler = LocalScheduler(
            self.store,
            self.state_dir / "jobs",
            workers=workers,
            max_output_bytes=max_output_bytes,
            atrex_bench_root=atrex_bench_root,
        )
        self._env_lock = threading.Lock()
        self._env_cache: dict[str, Any] | None = None

    def start(self) -> None:
        self.scheduler.start()

    def close(self) -> None:
        self.scheduler.stop()
        self.store.close()
        fcntl.flock(self._state_lock.fileno(), fcntl.LOCK_UN)
        self._state_lock.close()

    def submit(self, kind: str, payload: Any, trace_id: str) -> tuple[dict[str, Any], bool]:
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported job kind {kind!r}")
        request = (
            _validate_dev_request(payload)
            if kind == "dev"
            else _validate_typed_request(payload, kind)
        )
        targets = request["spec"]["target_hardware"]
        if not any(target in self.gpu_aliases for target in targets):
            accepted = ", ".join(sorted(self.gpu_aliases))
            raise ValueError(
                f"localhost scheduler accepts target_hardware aliases: {accepted}"
            )
        job, created = self.store.create(kind, request, trace_id)
        if created:
            self.scheduler.notify()
        accepted = {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "status": job["status"],
            "trace_id": job["trace_id"],
        }
        return accepted, created

    def submit_dev(self, payload: Any, trace_id: str) -> tuple[dict[str, Any], bool]:
        return self.submit("dev", payload, trace_id)

    def environment(self, force: bool = False) -> dict[str, Any]:
        with self._env_lock:
            if self._env_cache is None or force:
                self._env_cache = _probe_environment()
            result = dict(self._env_cache)
            result["aliases"] = sorted(self.gpu_aliases)
            return result


def _probe_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu": "local",
        "vendor": "unknown",
        "gpu_model": None,
        "aliases": ["local"],
        "arch": None,
        "sm_count": None,
        "total_memory_mb": None,
        "driver_version": None,
        "toolchain": {"python": platform.python_version()},
        "source": "declared",
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "probe_error": None,
    }
    errors: list[str] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first = result.stdout.splitlines()[0]
        model, memory, driver = (part.strip() for part in first.split(",", 2))
        info.update(
            vendor="nvidia",
            gpu_model=model,
            total_memory_mb=int(float(memory)),
            driver_version=driver,
            source="probe",
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        errors.append(f"nvidia-smi: {exc}")

    try:
        code = (
            "import json, torch; p=torch.cuda.get_device_properties(0); "
            "hip=getattr(torch.version,'hip',None); "
            "arch=(getattr(p,'gcnArchName','').split(':')[0] if hip else "
            "'sm_%d%d'%torch.cuda.get_device_capability(0)); "
            "print(json.dumps({'arch':arch,'sm_count':p.multi_processor_count,"
            "'torch':torch.__version__,'runtime':hip or torch.version.cuda}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        runtime = json.loads(result.stdout)
        info["arch"] = runtime["arch"]
        info["sm_count"] = runtime["sm_count"]
        info["toolchain"].update(
            torch=runtime["torch"],
            runtime=runtime["runtime"],
        )
        info["source"] = "probe"
        if info["vendor"] == "unknown":
            info["vendor"] = "amd" if str(runtime["arch"]).startswith("gfx") else "nvidia"
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        errors.append(f"torch: {exc}")
    if errors and info["source"] == "declared":
        info["probe_error"] = "; ".join(errors)
    return info


class GatewayRequestHandler(BaseHTTPRequestHandler):
    gateway: LocalGateway
    body_limit = DEFAULT_BODY_LIMIT
    server_version = "atrex-local-gateway/0.2"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        trace_id = self._trace_id()
        if path in ("/healthz", "/status.taobao"):
            self._send_text(HTTPStatus.OK, "ok")
            return
        if path == "/v1/env":
            env = self.gateway.environment(force=self._bool_query(query, "force"))
            self._send_json(HTTPStatus.OK, {"env": [env]})
            return
        if path.startswith("/v1/env/"):
            gpu = unquote(path.removeprefix("/v1/env/"))
            if gpu != "local":
                self._not_found("environment_not_found", f"environment {gpu!r} not found", trace_id)
                return
            self._send_json(
                HTTPStatus.OK,
                self.gateway.environment(force=self._bool_query(query, "force")),
            )
            return
        if path == "/v1/jobs":
            kind = self._first(query, "kind")
            status = self._first(query, "status")
            user = self._first(query, "user")
            if kind is not None and kind not in KNOWN_KINDS:
                self._bad_request("invalid_kind", f"unknown job kind {kind!r}", trace_id)
                return
            if status is not None and status not in VALID_STATUSES:
                self._bad_request("invalid_status", f"unknown job status {status!r}", trace_id)
                return
            try:
                limit = min(200, max(1, int(self._first(query, "limit") or "50")))
            except ValueError:
                self._bad_request("invalid_limit", "limit must be an integer", trace_id)
                return
            descending = (self._first(query, "sort") or "-created_at") != "created_at"
            jobs = self.gateway.store.list(
                kind=kind,
                user=user,
                status=status,
                limit=limit,
                descending=descending,
            )
            self._send_json(HTTPStatus.OK, {"jobs": jobs})
            return
        job_id = self._job_id_from_get_path(path)
        if job_id is not None:
            job = self.gateway.store.get(job_id)
            if job is None:
                self._not_found("job_not_found", f"job {job_id!r} not found", trace_id)
                return
            if self._bool_query(query, "wait") and job["status"] not in TERMINAL_STATUSES:
                try:
                    timeout = min(300.0, max(0.0, float(self._first(query, "timeout") or "30")))
                except ValueError:
                    self._bad_request("invalid_timeout", "timeout must be numeric", trace_id)
                    return
                job = self.gateway.scheduler.wait_for_job(job_id, timeout) or job
            self._send_json(HTTPStatus.OK, job)
            return
        self._not_found("route_not_found", f"route {path!r} not found", trace_id)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        trace_id = self._trace_id()
        cancel_match = re.fullmatch(r"/v1/jobs/([^/]+)/cancel", path)
        if cancel_match:
            job_id = unquote(cancel_match.group(1))
            job = self.gateway.scheduler.cancel(job_id)
            if job is None:
                self._not_found("job_not_found", f"job {job_id!r} not found", trace_id)
            else:
                self._send_json(HTTPStatus.OK, job)
            return
        kind_match = re.fullmatch(r"/v1/jobs/(eval|profile|dev)", path)
        if kind_match or path == "/v1/evals":
            kind = kind_match.group(1) if kind_match else "eval"
            try:
                payload = self._read_json()
                accepted, _ = self.gateway.submit(kind, payload, trace_id)
            except RequestError as exc:
                self._send_json(exc.status, _error(exc.reason, str(exc), trace_id))
                return
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    _error("validation_error", str(exc), trace_id),
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, accepted)
            return
        if path == "/v1/jobs/disassemble":
            self._send_json(
                HTTPStatus.NOT_IMPLEMENTED,
                _error(
                    "kind_not_supported",
                    "community localhost scheduler does not implement disassemble jobs",
                    trace_id,
                ),
            )
            return
        self._not_found("route_not_found", f"route {path!r} not found", trace_id)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[local-gateway] {self.address_string()} {fmt % args}", file=sys.stderr)

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length") from exc
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "empty_body", "request body is required")
        if length > self.body_limit:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request body is too large")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid JSON") from exc

    def _job_id_from_get_path(self, path: str) -> str | None:
        match = re.fullmatch(r"/v1/jobs/([^/]+)", path)
        if match:
            return unquote(match.group(1))
        legacy = re.fullmatch(r"/v1/evals/([^/]+)", path)
        return unquote(legacy.group(1)) if legacy else None

    def _trace_id(self) -> str:
        incoming = self.headers.get("X-Trace-Id")
        return incoming if incoming and len(incoming) <= 128 else f"req-{secrets.token_hex(6)}"

    @staticmethod
    def _first(query: dict[str, list[str]], name: str) -> str | None:
        values = query.get(name)
        return values[0] if values else None

    def _bool_query(self, query: dict[str, list[str]], name: str) -> bool:
        value = (self._first(query, name) or "false").lower()
        return value in {"1", "true", "yes", "on"}

    def _bad_request(self, reason: str, message: str, trace_id: str) -> None:
        self._send_json(HTTPStatus.BAD_REQUEST, _error(reason, message, trace_id))

    def _not_found(self, reason: str, message: str, trace_id: str) -> None:
        self._send_json(HTTPStatus.NOT_FOUND, _error(reason, message, trace_id))

    def _send_text(self, status: HTTPStatus, value: str) -> None:
        data = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: HTTPStatus, value: Any) -> None:
        data = _json_dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason


def create_server(
    gateway: LocalGateway,
    host: str,
    port: int,
    *,
    body_limit: int = DEFAULT_BODY_LIMIT,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundGatewayRequestHandler",
        (GatewayRequestHandler,),
        {"gateway": gateway, "body_limit": body_limit},
    )
    return ThreadingHTTPServer((host, port), handler)


def _default_state_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "atrex-local-gateway"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue agate run/profile/dev jobs on a trusted localhost GPU."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the localhost-compatible HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--state-dir", type=Path, default=_default_state_dir())
    serve.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent local commands (default: 1, FIFO serialization)",
    )
    serve.add_argument(
        "--gpu-alias",
        action="append",
        default=[],
        metavar="NAME",
        help="additional hardware token accepted as the local GPU (repeatable)",
    )
    serve.add_argument("--max-request-mb", type=int, default=32)
    serve.add_argument("--max-output-mb", type=int, default=32)
    serve.add_argument(
        "--atrex-bench-root",
        type=Path,
        default=None,
        help=(
            "atrex-bench checkout containing scripts/run_eval.py; normally auto-detected "
            "from the active Python environment or ATREX_BENCH_ROOT"
        ),
    )
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback bind; unsafe because commands run without isolation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.error("a command is required")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be in 1..64")
    if args.max_request_mb <= 0 or args.max_output_mb <= 0:
        parser.error("request/output limits must be positive")
    if any(not alias or len(alias) > 128 for alias in args.gpu_alias):
        parser.error("--gpu-alias values must be non-empty and at most 128 characters")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        parser.error("non-loopback --host requires --allow-remote")

    try:
        gateway = LocalGateway(
            args.state_dir.resolve(),
            workers=args.workers,
            max_output_bytes=args.max_output_mb * 1024 * 1024,
            gpu_aliases=tuple(args.gpu_alias),
            atrex_bench_root=(args.atrex_bench_root.resolve() if args.atrex_bench_root else None),
        )
    except RuntimeError as exc:
        print(f"local-gateway: {exc}", file=sys.stderr)
        return 2
    server = create_server(
        gateway,
        args.host,
        args.port,
        body_limit=args.max_request_mb * 1024 * 1024,
    )
    gateway.start()
    actual_host, actual_port = server.server_address[:2]
    print(
        f"[local-gateway] listening on http://{actual_host}:{actual_port} "
        f"workers={args.workers} state={args.state_dir.resolve()}",
        flush=True,
    )
    print(
        "[local-gateway] WARNING: commands run as the current user without isolation; trusted inputs only",
        file=sys.stderr,
        flush=True,
    )

    stopping = threading.Event()

    def stop_server(signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print(f"[local-gateway] received signal {signum}; shutting down", file=sys.stderr, flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop_server)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        gateway.close()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
