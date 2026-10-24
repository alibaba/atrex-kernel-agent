"""Campaign-owned Wiki slots, Episode-scoped single-flight and frozen responses."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
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

from aka.contracts.wiki import WikiCall, WikiProviderError
from aka.light.execution.process import ExecutionControl
from aka.light.runtime.errors import GatewayExecutionUnavailable, SessionRevokedError
from aka.light.runtime.request_errors import AgentRequestError, RuntimeStateError
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


def _provider_failure(error: WikiProviderError) -> Exception:
    """Classify a provider failure without trusting it to choose an HTTP outcome."""
    if error.code == "wiki_revision_changed":
        return RuntimeStateError(error.message, code=error.code, next_action=(
            "The Wiki source changed after this Campaign pinned it. Continue without Wiki and report the "
            "blocker; only the operator can restore the pinned source or start a new Campaign."))
    if error.retryable:
        return WikiUnavailable(error.code, error.message)
    if error.repairable:
        return AgentRequestError(error.message, code=error.code, next_action=error.next_action or (
            "Correct the request using this operation's input schema (`python3 tools/plugin.py list`), then retry."))
    return RuntimeStateError(error.message, code=error.code, next_action=error.next_action or (
        "Continue without this Wiki operation and report the blocker to the operator. "
        "Do not change provider credentials or access the Wiki source directly."))


class WikiService:
    def __init__(self, root, config, revision, provider=None):
        self.root, self.config = root, config
        self.revision = digest(revision)
        self.provider = provider
        self._provider_revision = None
        self._revision_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(config.concurrency)
        self._lock_files = threading.Lock()

    def pinned_revision(self) -> str:
        """Pin one provider revision across every Supervisor using this Campaign root."""
        with self._revision_lock:
            if self._provider_revision is not None:
                return self._provider_revision
            with open_private_directory(self.root) as directory:
                lock = os.open(
                    "revision.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600,
                    dir_fd=directory,
                )
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    path = self.root / "revision.json"
                    try:
                        pinned = json.loads(read_regular_bytes(path, limit=64 * 1024))["revision"]
                        if not isinstance(pinned, str) or not pinned:
                            raise ValueError("invalid pinned revision")
                    except FileNotFoundError:
                        pinned = None
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        raise RuntimeStateError(
                            "The pinned Wiki revision record is unreadable.",
                            code="wiki_configuration_invalid",
                        ) from error
                    try:
                        current = self.provider.revision()
                    except WikiProviderError as error:
                        raise _provider_failure(error) from error
                    if not isinstance(current, str) or not current:
                        raise RuntimeStateError(
                            "The Wiki provider returned no revision.",
                            code="wiki_configuration_invalid",
                        )
                    if pinned is None:
                        publish(
                            self.root, "revision.json",
                            json.dumps({"schema_version": 1, "revision": current}).encode(),
                        )
                    elif current != pinned:
                        raise _provider_failure(WikiProviderError(
                            "wiki_revision_changed",
                            "The Wiki source changed after this Campaign pinned its revision.",
                        ))
                    self._provider_revision = current
                    return current
                finally:
                    os.close(lock)

    def _identity(self):
        return digest([self.revision, self.pinned_revision()]) if self.provider is not None else self.revision

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
                      "policy": asdict(policy), "provider": provider, "revision": self._identity()})
        record = {"request_id": request_id, "query_key": key, "tool": tool,
                  "cache_hit": False, "status": "waiting", "queue_latency_ms": 0,
                  "bridge_latency_ms": None, "retrieval_latency_ms": None}
        try:
            with self._reservation(key, min(deadline, started + policy.queue_timeout), live):
                record["queue_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                cached = self._cached(key)
                if cached is not None:
                    if not live():
                        raise SessionRevokedError("Session revoked")
                    record.update(cache_hit=True, status="reused", query_id=cached.get("query_id"))
                    return cached["response"]

                def acquire():
                    self._acquire_slot(started, deadline, live, record)

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

    def invoke(self, owner, capability, operation, payload, scratch, environment, *, project,
               redact_error=lambda value: value):
        """Run one provider-declared operation under the same slots, single-flight and receipts.

        ``project`` redacts and validates a raw provider result; only its output is cached.
        ``redact_error`` applies the same Runtime floor to provider diagnostics.
        """
        started = time.monotonic()
        policy = self.config
        deadline = started + policy.timeout
        request_id = uuid.uuid4().hex
        live = lambda: owner._live(capability)
        revision = self.pinned_revision()
        scope = str(capability.journal.service.path) if capability.journal else str(capability.workspace.resolve())
        key = digest({"schema": 2, "scope": scope, "operation": operation, "payload": payload,
                      "policy": asdict(policy), "revision": self._identity()})
        record = {"request_id": request_id, "query_key": key, "tool": operation,
                  "cache_hit": False, "status": "waiting", "queue_latency_ms": 0, "revision": revision}
        try:
            with self._reservation(key, min(deadline, started + policy.queue_timeout), live):
                record["queue_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                cached = self._cached(key)
                if cached is not None:
                    if not live():
                        raise SessionRevokedError("Session revoked")
                    response = cached["response"]
                    query_id = cached.get("query_id")
                    if not query_id and isinstance(response, dict):
                        query_id = response.get("query_id")
                    record.update(cache_hit=True, status="reused", query_id=query_id)
                    return response
                self._acquire_slot(started, deadline, live, record)
                record["status"] = "running"
                no_slot = lambda: None
                call = WikiCall(operation=operation, payload=payload, revision=revision, deadline=deadline,
                                is_live=live, scratch=scratch, environment=environment,
                                run=lambda argv, cwd, env: owner.executor.run_gateway(
                                    list(argv), cwd, dict(env),
                                    ExecutionControl(deadline, request_id, live, no_slot, no_slot)))
                try:
                    result = self.provider.invoke(call)
                except WikiProviderError as error:
                    # Provider codes are diagnostics, never receipt lifecycle states.
                    message = str(redact_error(error.message))[:2000]
                    next_action = str(redact_error(error.next_action))[:2000]
                    record.update(
                        status="provider_error", provider_error_code=error.code[:128],
                        provider_error_message=message,
                    )
                    public_error = WikiProviderError(
                        error.code, message, retryable=error.retryable,
                        repairable=error.repairable, next_action=next_action,
                    )
                    raise _provider_failure(public_error) from error
                except GatewayExecutionUnavailable as error:
                    if not live():
                        raise SessionRevokedError("Session revoked") from error
                    record["status"] = "timed_out" if time.monotonic() >= deadline else "execution_failed"
                    raise WikiUnavailable("wiki_query_timeout" if record["status"] == "timed_out" else "wiki_unavailable",
                                          "Wiki query did not complete within its execution limits.") from error
                finally:
                    self.slots.release()
                if not live():
                    raise SessionRevokedError("Session revoked")
                if time.monotonic() >= deadline:
                    record["status"] = "timed_out"
                    raise WikiUnavailable("wiki_query_timeout", "Wiki query did not complete within its execution limits.")
                try:
                    if not isinstance(result, Mapping):
                        raise TypeError("result is not an object")
                    result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
                except (TypeError, ValueError, RecursionError) as error:
                    record["status"] = "invalid_output"
                    raise WikiUnavailable("wiki_invalid_response",
                                          "Wiki returned an invalid result; no reusable response was saved.") from error
                record["status"] = "invalid_output"
                response = project(result)
                record["status"] = "completed"
                query_id = response.get("query_id") if isinstance(response, dict) else None
                if isinstance(query_id, str) and query_id:
                    record["query_id"] = query_id
                owner.audit_process(
                    replace(capability, request_id=request_id), "wiki", [operation],
                    subprocess.CompletedProcess(
                        [operation], 0, json.dumps(response, ensure_ascii=False), "",
                    ),
                )
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
            publish(self.root, "requests/" + request_id + ".json", json.dumps(record).encode())

    def _cached(self, key):
        try:
            cached = json.loads(read_regular_bytes(self.root / "results" / (key + ".json"), limit=2 * 1024 * 1024))
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            # Corruption is not evidence; a read-only query may be safely repeated.
            return None
        if (isinstance(cached, dict) and cached.get("key") == key
                and cached.get("response_digest") == digest(cached.get("response"))):
            return cached
        return None

    def _acquire_slot(self, started, deadline, live, record=None):
        queue_deadline = min(deadline, started + self.config.queue_timeout)
        while True:
            if not live():
                raise SessionRevokedError("Session revoked")
            remaining = queue_deadline - time.monotonic()
            if remaining <= 0:
                raise WikiUnavailable("wiki_queue_timeout", "Wiki concurrency limit reached; this query did not start.")
            if self.slots.acquire(timeout=min(0.1, remaining)):
                if record is not None:
                    record["queue_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                return

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
