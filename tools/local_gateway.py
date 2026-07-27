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

"""Small localhost scheduler compatible with the public agate dev API.

The server intentionally implements the subset used by ``tools/sandbox.py``:

* ``GET /healthz``
* ``GET /v1/env`` and ``GET /v1/env/local``
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
import fcntl
import json
import os
import platform
import re
import secrets
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
SUPPORTED_KIND = "dev"
KNOWN_KINDS = frozenset({"eval", "profile", "dev", "disassemble"})
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_BODY_LIMIT = 32 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 32 * 1024 * 1024


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
    return {"dev": "dv", "eval": "ev", "profile": "pr", "disassemble": "ds"}.get(kind, "jb")


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

    def claim_next(self) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT job_id, request_json FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
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
            return row["job_id"], _json_loads(row["request_json"])

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
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "mode": request.get("mode"),
            "lock_clocks": request.get("lock_clocks"),
            "mode_enforced": None,
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


def _validate_dev_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    spec = payload.get("spec")
    targets = spec.get("target_hardware") if isinstance(spec, dict) else None
    if not isinstance(targets, list) or not targets or not all(isinstance(v, str) for v in targets):
        raise ValueError("spec.target_hardware must be a non-empty string array")
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if len(command.encode("utf-8")) > 1024 * 1024:
        raise ValueError("command exceeds the 1 MiB limit")

    timeout_s = payload.get("timeout_s", 600)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or not 1 <= timeout_s <= 600:
        raise ValueError("timeout_s must be an integer in the range 1..600")

    env_vars = payload.get("env_vars") or {}
    if not isinstance(env_vars, dict) or not all(
        isinstance(k, str) and ENV_NAME_RE.fullmatch(k) and isinstance(v, str)
        for k, v in env_vars.items()
    ):
        raise ValueError("env_vars must map valid environment names to strings")

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

    idempotency_key = payload.get("idempotency_key")
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not idempotency_key or len(idempotency_key) > 256
    ):
        raise ValueError("idempotency_key must be a non-empty string of at most 256 characters")
    return payload


class LocalScheduler:
    """Persistent FIFO worker pool for commands targeting the local GPU."""

    def __init__(self, store: JobStore, jobs_dir: Path, workers: int, max_output_bytes: int) -> None:
        self.store = store
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workers = workers
        self.max_output_bytes = max_output_bytes
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
            job_id, request = claimed
            self.notify()
            if self._stop.is_set():
                self.store.complete(
                    job_id,
                    status="failed",
                    result=None,
                    error=_error("scheduler_stopped", "local scheduler stopped before execution"),
                )
                break
            self._execute(job_id, request)
            self.notify()

    def _execute(self, job_id: str, request: dict[str, Any]) -> None:
        workdir = self.jobs_dir / job_id
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
            for name, content in (request.get("files") or {}).items():
                relative = _safe_destination(name)
                target = workdir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                target.chmod(0o600)

            stdout_path = workdir / ".stdout"
            stderr_path = workdir / ".stderr"
            env = os.environ.copy()
            env.update(request.get("env_vars") or {})
            env["ATREX_LOCAL_JOB_ID"] = job_id
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                stdout_path.chmod(0o600)
                stderr_path.chmod(0o600)
                process = subprocess.Popen(
                    ["bash", "-lc", request["command"]],
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
                    process.wait(timeout=int(request.get("timeout_s", 600)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate(process)

            current = self.store.get(job_id)
            if current is None or current["status"] == "cancelled":
                return
            stdout, stdout_truncated = self._read_output(stdout_path)
            stderr, stderr_truncated = self._read_output(stderr_path)
            result = {
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            if timed_out:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=result,
                    error=_error("command_timeout", f"command exceeded {request.get('timeout_s', 600)} seconds"),
                )
            elif stdout_truncated or stderr_truncated:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=result,
                    error=_error(
                        "output_too_large",
                        f"command output exceeded the {self.max_output_bytes} byte stream limit",
                    ),
                )
            elif process.returncode == 0:
                self.store.complete(job_id, status="succeeded", result=result, error=None)
            else:
                self.store.complete(
                    job_id,
                    status="failed",
                    result=result,
                    error=_error("command_failed", f"command exited with status {process.returncode}"),
                )
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

    def submit_dev(self, payload: Any, trace_id: str) -> tuple[dict[str, Any], bool]:
        request = _validate_dev_request(payload)
        targets = request["spec"]["target_hardware"]
        if not any(target in self.gpu_aliases for target in targets):
            accepted = ", ".join(sorted(self.gpu_aliases))
            raise ValueError(
                f"localhost scheduler accepts target_hardware aliases: {accepted}"
            )
        job, created = self.store.create(SUPPORTED_KIND, request, trace_id)
        if created:
            self.scheduler.notify()
        accepted = {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "status": job["status"],
            "trace_id": job["trace_id"],
        }
        return accepted, created

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
    server_version = "atrex-local-gateway/0.1"

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
        if path == "/v1/jobs/dev":
            try:
                payload = self._read_json()
                accepted, _ = self.gateway.submit_dev(payload, trace_id)
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
        kind_match = re.fullmatch(r"/v1/jobs/(eval|profile|disassemble)", path)
        if kind_match or path == "/v1/evals":
            kind = kind_match.group(1) if kind_match else "eval"
            self._send_json(
                HTTPStatus.NOT_IMPLEMENTED,
                _error(
                    "kind_not_supported",
                    f"community localhost scheduler supports dev jobs; {kind} jobs are not implemented",
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
    parser = argparse.ArgumentParser(description="Queue agate dev jobs on a trusted localhost GPU.")
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
