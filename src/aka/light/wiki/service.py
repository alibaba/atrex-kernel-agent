"""Campaign-owned Wiki slots, Episode-scoped single-flight and frozen responses."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from aka.light.execution.process import ExecutionControl
from aka.light.runtime.errors import GatewayExecutionUnavailable, SessionRevokedError
from aka.light.runtime.request_errors import RuntimeStateError
from aka.light.workspace.files import publish
from aka.light.agent.home import open_private_directory
from aka.light.agent.session_tail import read_regular_bytes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class WikiUnavailable(RuntimeStateError):
    def __init__(self, code, message):
        super().__init__(message, code=code, next_action=(
            "Wiki is read-only. Retry the same query unchanged after at least 5 seconds; "
            "the Supervisor will reuse a completed result or wait for the in-flight query. "
            "Do not restart GPU measurements or change provider credentials. If repeated, continue "
            "without Wiki and report the blocker to the operator."))
        self.response.update(retry_after_seconds=5, retryable=True)


class WikiService:
    def __init__(self, root, config, revision):
        self.root, self.config = root, config
        self.revision = digest(revision)
        self.slots = threading.BoundedSemaphore(config.concurrency)
        self._lock_files = threading.Lock()

    @contextmanager
    def _reservation(self, key, deadline, live):
        # Serialize inode creation only, not query execution or waiting. The
        # permanent lock inode is also shared by other Supervisor processes.
        with self._lock_files, open_private_directory(self.root / "locks") as directory:
            try:
                fd = os.open(key + ".lock", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            except FileExistsError:
                fd = os.open(key + ".lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
        try:
            while True:
                if not live():
                    raise SessionRevokedError("Session revoked")
                if time.monotonic() >= deadline:
                    raise WikiUnavailable("wiki_query_in_progress", "An identical Wiki query is still running; no duplicate was started.")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            yield
        finally:
            os.close(fd)

    def execute(self, owner, capability, command, staged, environment, argv):
        started = time.monotonic()
        policy = self.config
        deadline = started + policy.timeout
        request_id = uuid.uuid4().hex
        live = lambda: owner._live(capability)
        tool = Path(command[1]).stem
        normalized = list(command[2:])
        # Request-file identity binds frozen content, never the random staging path.
        for i, word in enumerate(normalized):
            if word == "--file" and i + 1 < len(normalized):
                normalized[i + 1] = read_regular_bytes(Path(normalized[i + 1]), limit=128 * 1024).decode()
            elif word.startswith("--file="):
                normalized[i] = "--file=" + read_regular_bytes(Path(word.split("=", 1)[1]), limit=128 * 1024).decode()
        scope = str(capability.journal.service.path) if capability.journal else str(capability.workspace.resolve())
        provider = {"model": policy.model or environment.get("ANTHROPIC_MODEL", ""),
                    "endpoint": policy.base_url or environment.get("ANTHROPIC_BASE_URL", "")}
        if policy.settings_file:
            try:
                info = Path(policy.settings_file).stat()
                provider["settings_revision"] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            except OSError as error:
                raise RuntimeStateError("Wiki provider settings are unavailable.", code="wiki_configuration_invalid") from error
        key = digest({"schema": 1, "scope": scope, "tool": tool, "argv": normalized,
                      "policy": asdict(policy), "provider": provider, "revision": self.revision})
        record = {"request_id": request_id, "query_key": key, "tool": tool,
                  "cache_hit": False, "status": "waiting", "queue_latency_ms": 0,
                  "bridge_latency_ms": None, "retrieval_latency_ms": None}
        try:
            with self._reservation(key, min(deadline, started + policy.queue_timeout), live):
                record["queue_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                try:
                    cached = json.loads(read_regular_bytes(self.root / "results" / (key + ".json"), limit=2 * 1024 * 1024))
                except FileNotFoundError:
                    cached = None
                except (ValueError, OSError):
                    # Corruption is not evidence; a read-only query may be safely repeated.
                    cached = None
                if (isinstance(cached, dict) and cached.get("key") == key
                        and cached.get("response_digest") == digest(cached.get("response"))):
                    if not live():
                        raise SessionRevokedError("Session revoked")
                    record.update(cache_hit=True, status="reused", query_id=cached.get("query_id"))
                    return cached["response"]

                def acquire():
                    queue_deadline = min(deadline, started + policy.queue_timeout)
                    while True:
                        if not live():
                            raise SessionRevokedError("Session revoked")
                        remaining = queue_deadline - time.monotonic()
                        if remaining <= 0:
                            raise WikiUnavailable("wiki_queue_timeout", "Wiki concurrency limit reached; this query did not start.")
                        if self.slots.acquire(timeout=min(0.1, remaining)):
                            record["queue_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                            return

                environment = dict(environment)
                from .worker import provider_environment
                try:
                    redaction_values = provider_environment(policy, environment)
                except ValueError as error:
                    raise RuntimeStateError("Wiki provider settings are invalid or the credential endpoint does not match.",
                                            code="wiki_configuration_invalid") from error
                environment["ATREX_WIKI_METRICS_LOG"] = str(staged / "native-metrics.jsonl")
                # Native bridge is a standalone no-tools session, not a Campaign child handoff.
                for name in ("ATREX_ENVIRONMENT_RESTART_HANDOFF_ID", "ATREX_ENVIRONMENT_RESTART_LOCK_FD"):
                    environment.pop(name, None)
                request = {"tools": str(Path(command[1]).parent), "tool": tool,
                           "argv": command[2:], "config": asdict(policy)}
                publish(staged, "wiki-request.json", json.dumps(request).encode())
                record["status"] = "running"
                control = ExecutionControl(deadline, request_id, live, acquire, self.slots.release)
                try:
                    process = owner.executor.run_gateway(
                        [sys.executable, "-m", "aka.light.wiki.worker", str(staged / "wiki-request.json")],
                        staged, environment, control)
                except GatewayExecutionUnavailable as error:
                    if not live():
                        raise SessionRevokedError("Session revoked") from error
                    record["status"] = "timed_out" if time.monotonic() >= deadline else "execution_failed"
                    raise WikiUnavailable("wiki_query_timeout" if record["status"] == "timed_out" else "wiki_unavailable",
                                          "Wiki query did not complete within its execution limits.") from error
                finally:
                    self._timing(staged, record)
                if not live():
                    raise SessionRevokedError("Session revoked")
                # Use a request-local identity; never mutate capability.deadline/request_id.
                from dataclasses import replace
                owner.audit_process(replace(capability, request_id=request_id), "wiki", argv, process)
                from aka.light.evaluation.projection import credential_values, project_response
                response = project_response(process, generalized=owner.config.private_reference_dir is not None,
                    private_values=credential_values(dict(environment, **redaction_values)), wiki=True,
                    private_paths=(str(staged), str(owner.root), str(owner.audit_root or ""), policy.settings_file, policy.base_url))
                record["status"] = "failed" if process.returncode else "completed"
                # Only successful, validated native JSON may become reusable evidence.
                if process.returncode == 0:
                    try:
                        if tool == "query_nl":
                            value = json.loads(response["stdout"])
                            if (not isinstance(value, dict) or not isinstance(value.get("query_id"), str)
                                    or not value["query_id"] or not isinstance(value.get("records"), dict)
                                    or not isinstance(value.get("notes"), list)):
                                raise ValueError("invalid Wiki envelope")
                            record["query_id"] = value["query_id"]
                    except (ValueError, TypeError):
                        record["status"] = "invalid_output"
                        raise WikiUnavailable("wiki_invalid_response", "Wiki returned an invalid result; no reusable response was saved.")
                    publish(self.root, "results/" + key + ".json", json.dumps({
                        "key": key, "response_digest": digest(response), "response": response,
                        "query_id": record.get("query_id"),
                    }, ensure_ascii=False).encode())
                return response
        finally:
            if record["status"] in {"waiting", "running"}:
                record["status"] = "revoked" if not live() else "not_completed"
            record["total_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            record["finished_at"] = time.time()
            # One private receipt per caller (including reuse/failure); no prompts or credentials.
            publish(self.root, "requests/" + request_id + ".json", json.dumps(record).encode())

    @staticmethod
    def _timing(staged, record):
        try:
            phase = json.loads(read_regular_bytes(staged / "phase.json", limit=65536))
            record.update({k: phase[k] for k in ("phase", "bridge_attempts", "bridge_latency_ms", "worker_latency_ms") if k in phase})
        except (OSError, ValueError):
            pass
        try:
            lines = read_regular_bytes(staged / "native-metrics.jsonl", limit=65536).splitlines()
            native = json.loads(lines[-1])
            for name in ("bridge_protocol", "bridge_attempts", "bridge_retry_count", "bridge_latency_ms",
                         "retrieval_latency_ms", "record_count", "bridge_input_tokens", "bridge_output_tokens"):
                if name in native:
                    record[name] = native[name]
        except (OSError, ValueError, IndexError, TypeError):
            pass
