"""Local experiment persistence. Disk codecs and paths stay inside Experiment."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
import uuid

from aka.contracts.execution import (
    ExecutionAttempt, ExecutionDiagnostic, ExecutionFacts, JobResult,
    ResourceIdentity, ExecutionFileReceipt,
)
from aka.contracts.experiment import (
    ExperimentArtifact, ExperimentAttempt, ExperimentFailure, ExperimentRecord,
    ExperimentRecordError, ExperimentRef, ExperimentStep, ExperimentEvaluation,
)
from aka.contracts.evaluation import EvaluationFailure


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _diagnostic(value):
    return ExecutionDiagnostic(**value)


def _job(value):
    if value is None:
        return None
    data = dict(value)
    data["file_receipts"] = tuple(ExecutionFileReceipt(**item) for item in data.get("file_receipts", ()))
    data["resource"] = ResourceIdentity(**data["resource"]) if data["resource"] else None
    facts = dict(data["facts"])
    facts["cleanup_diagnostics"] = tuple(map(_diagnostic, facts["cleanup_diagnostics"]))
    data["facts"] = ExecutionFacts(**facts)
    data["diagnostics"] = tuple(map(_diagnostic, data["diagnostics"]))
    attempts = []
    for item in data["attempts"]:
        item = dict(item)
        item["resource"] = ResourceIdentity(**item["resource"]) if item["resource"] else None
        item["diagnostics"] = tuple(map(_diagnostic, item["diagnostics"]))
        attempts.append(ExecutionAttempt(**item))
    data["attempts"] = tuple(attempts)
    return JobResult(**data)


def _record(value):
    data = dict(value)
    data["ref"] = ExperimentRef(**data["ref"])
    steps = []
    for item in data["steps"]:
        item = dict(item)
        item["argv"] = tuple(item["argv"])
        item["environment_keys"] = tuple(item["environment_keys"])
        steps.append(ExperimentStep(**item))
    data["steps"] = tuple(steps)
    data["attempts"] = tuple(ExperimentAttempt(
        item["attempt_id"], item["task_id"], item["disposition"], _job(item["result"]),
        ExperimentFailure(**item["failure"]) if item["failure"] else None,
    ) for item in data["attempts"])
    data["artifacts"] = tuple(ExperimentArtifact(
        ExperimentRef(**item["experiment"]), item["artifact_id"], item["role"],
        item["sha256"], item["size_bytes"], item["visibility"],
    ) for item in data["artifacts"])
    data["failure"] = ExperimentFailure(**data["failure"]) if data["failure"] else None
    if data["evaluation"] is not None:
        evaluation = dict(data["evaluation"])
        if evaluation.get("experiment_ref"):
            evaluation["experiment_ref"] = ExperimentRef(**evaluation["experiment_ref"])
        if evaluation.get("failure"):
            evaluation["failure"] = EvaluationFailure(**evaluation["failure"])
        data["evaluation"] = ExperimentEvaluation(**evaluation)
    return ExperimentRecord(**data)


def _atomic(path, data, *, mode=None):
    """Publish complete bytes and sync both file and containing directory."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("record paths cannot traverse symlinks")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            if mode is not None:
                os.fchmod(output.fileno(), mode)
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _settled(record):
    return all(item.disposition in {"terminal", "not_submitted"} for item in record.attempts)


class FileExperimentRecords:
    """Single-host records. Writers hold a request lock for the whole operation.

    The public reader only consumes committed files. Incomplete operations are
    deliberately not resumed by this implementation: they remain queryable.
    """
    def __init__(self, root: Path, run_id: str):
        if not run_id:
            raise ValueError("experiment run_id is required")
        self.run_id = run_id
        self.root = Path(root).resolve() / _digest(run_id)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._write_lock = RLock()

    def _ref(self, request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("experiment request_id is required")
        return ExperimentRef(self.run_id, _digest([self.run_id, request_id]))

    def _directory(self, ref):
        if ref.run_id != self.run_id or not re.fullmatch(r"[0-9a-f]{64}", ref.experiment_id):
            raise LookupError("unknown experiment reference")
        path = self.root / ref.experiment_id
        if path.is_symlink() or path.resolve() != path:
            raise ExperimentRecordError("record_corrupt", ref)
        return path

    def find(self, request_id):
        return self.get(self._ref(request_id))

    def get(self, ref):
        path = self._directory(ref) / "record.json"
        if path.is_symlink():
            raise ExperimentRecordError("record_corrupt", ref)
        try:
            envelope = json.loads(path.read_bytes())
        except FileNotFoundError:
            raise LookupError("unknown experiment reference") from None
        except (OSError, ValueError) as error:
            raise ExperimentRecordError("record_corrupt", ref) from error
        try:
            if envelope["format"] != "aka-experiment-files/1":
                raise ExperimentRecordError("unsupported_record_format", ref)
            value = envelope["record"]
            if _digest(value) != envelope["sha256"]:
                raise ValueError("record digest mismatch")
            record = _record(value)
            if record.ref != ref or self._ref(record.request_id) != ref:
                raise ValueError("record identity mismatch")
            return record
        except ExperimentRecordError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ExperimentRecordError("record_corrupt", ref) from error

    def unresolved(self):
        records = []
        for directory in self.root.iterdir():
            if re.fullmatch(r"[0-9a-f]{64}", directory.name) and directory.is_dir():
                path = directory / "record.json"
                if path.exists():
                    records.append(self.get(ExperimentRef(self.run_id, directory.name)))
        return tuple(record.ref for record in sorted(records, key=lambda item: item.created_at)
                     if record.lifecycle != "terminal")

    def read_artifact(self, artifact):
        record = self.get(artifact.experiment)
        if artifact not in record.artifacts or not re.fullmatch(r"[0-9a-f]{32}", artifact.artifact_id):
            raise LookupError("unknown experiment artifact")
        folder = self._directory(record.ref) / "artifacts"
        path = folder / artifact.artifact_id
        try:
            if folder.is_symlink() or path.is_symlink() or not path.is_file():
                raise ValueError("invalid artifact location")
            data = path.read_bytes()
            if len(data) != artifact.size_bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
                raise ValueError("artifact content mismatch")
            return data
        except (OSError, ValueError) as error:
            raise ExperimentRecordError("artifact_corrupt", record.ref) from error

    @contextmanager
    def operation(self, request_id):
        ref = self._ref(request_id)
        locks = self.root / "locks"
        locks.mkdir(mode=0o700, exist_ok=True)
        path = locks / ref.experiment_id
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExperimentRecordError("experiment_busy", ref) from None
            yield ref
        finally:
            os.close(fd)

    def _save(self, record):
        value = record.as_dict()
        try:
            _atomic(self._directory(record.ref) / "record.json", _encoded({
                "format": "aka-experiment-files/1", "sha256": _digest(value), "record": value,
            }))
        except (OSError, ValueError, TypeError) as error:
            raise ExperimentRecordError("storage_unavailable", record.ref) from error
        return record

    def register(self, ref, request_id, request, *, kind, purpose):
        with self._write_lock:
            try:
                old = self.get(ref)
            except LookupError:
                old = None
            digest = _digest(request)
            if old is not None:
                if old.request_digest != digest or old.kind != kind or old.purpose != purpose:
                    raise ExperimentRecordError("request_conflict", ref)
                return old, False
            now = _now()
            record = ExperimentRecord(ref, request_id, kind, purpose, 1, "registered", "prepare",
                                      request, digest, created_at=now, updated_at=now)
            return self._save(record), True

    def _update(self, ref, transform):
        with self._write_lock:
            old = self.get(ref)
            if old.lifecycle == "terminal":
                raise ExperimentRecordError("experiment_terminal", ref)
            updated = transform(old)
            return self._save(replace(updated, record_version=old.record_version + 1, updated_at=_now()))

    def prepared(self, ref, commands, inputs=()):
        steps = tuple(ExperimentStep(f"step-{index}", index, tuple(command.command), command.operation,
                                     command.timeout, command.hardware, tuple(sorted(command.environment)))
                      for index, command in enumerate(commands))
        # Environment values affect identity but are not persisted as credentials.
        plan = [{**step.as_dict(), "environment_digest": _digest(dict(command.environment)),
                 "artifacts": list(command.artifacts), "num_gpus": command.num_gpus, "submission_sha256":
                 hashlib.sha256(command.submission).hexdigest() if command.submission is not None else None} for step, command in zip(steps, commands)]
        identity = {"steps": plan, "inputs": [
            {"path": item.path, "sha256": hashlib.sha256(item.content).hexdigest(),
             "mode": item.mode, "content_type": item.content_type} for item in inputs
        ]} if inputs else plan
        return self._update(ref, lambda old: replace(old, steps=steps, plan_digest=_digest(identity),
                                                    lifecycle="prepared", phase="execute"))

    def intent(self, ref, index):
        attempt = ExperimentAttempt(uuid.uuid4().hex, f"step-{index}", "intent")
        self._update(ref, lambda old: replace(old, attempts=(*old.attempts, attempt),
                                             lifecycle="running", phase="execute"))
        return attempt.attempt_id

    def observed(self, ref, attempt_id, result=None, error=None):
        if result is None and error is not None:
            result = getattr(error, "outcome", None) or getattr(error, "execution_outcome", None)
        if not isinstance(result, JobResult):
            result = None
        disposition = result.facts.disposition if result else "unknown"
        if disposition not in {"terminal", "not_submitted"} or (
            result and any(item.disposition == "unknown" for item in result.attempts)
        ):
            disposition = "unknown"
        failure = ExperimentFailure("execute", "execution_error", type(error).__name__) if error else None

        def transform(old):
            if not any(item.attempt_id == attempt_id for item in old.attempts):
                raise ExperimentRecordError("attempt_mismatch", ref)
            return replace(old, attempts=tuple(replace(item, disposition=disposition, result=result, failure=failure)
                                               if item.attempt_id == attempt_id else item for item in old.attempts),
                           phase="record_step")
        return self._update(ref, transform)

    def phase(self, ref, phase):
        return self._update(ref, lambda old: replace(old, phase=phase))

    def publish(self, ref, path, role):
        """Collect an existing command/compatibility artifact."""
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact must be a regular file")
            data = path.read_bytes()
            return self.publish_content(ref, data, role)
        except ExperimentRecordError:
            raise
        except (OSError, ValueError) as error:
            raise ExperimentRecordError("artifact_unavailable", ref) from error

    def publish_content(self, ref, content: bytes, role):
        """Publish returned content before committing its opaque record reference."""
        if not isinstance(content, bytes):
            raise TypeError("experiment artifact content must be bytes")
        artifact = ExperimentArtifact(ref, uuid.uuid4().hex, role,
                                      hashlib.sha256(content).hexdigest(), len(content))
        try:
            destination = self._directory(ref) / "artifacts" / artifact.artifact_id
            _atomic(destination, content)
            self._update(ref, lambda old: replace(old, artifacts=(*old.artifacts, artifact)))
            return artifact
        except ExperimentRecordError:
            raise
        except (OSError, ValueError) as error:
            raise ExperimentRecordError("storage_unavailable", ref) from error

    def completed(self, ref, evaluation=None):
        if evaluation is not None:
            value = dict(evaluation)
            if value.get("failure"):
                value["failure"] = EvaluationFailure(**value["failure"])
            if value.get("experiment_ref"):
                value["experiment_ref"] = ExperimentRef(**value["experiment_ref"])
            evaluation = ExperimentEvaluation(**value)
        def transform(old):
            outcome = "completed"
            if evaluation is not None and (evaluation.gate != "PASS" or evaluation.error):
                outcome = "failed" if evaluation.gate == "FAIL" else "inconclusive"
            if old.failure and old.failure.code == "artifact_unavailable":
                outcome = "inconclusive"
            return replace(old, evaluation=evaluation, outcome=outcome,
                           lifecycle="terminal" if _settled(old) else "result_recorded", phase="complete")
        return self._update(ref, transform)

    def artifact_missing(self, ref):
        return self._update(ref, lambda old: replace(old, failure=ExperimentFailure(
            "artifact", "artifact_unavailable")))

    def failed(self, ref, error):
        if isinstance(error, ExperimentRecordError):
            old = self.get(ref)
            if old.phase != "prepare" or old.attempts:
                return  # Preserve previously committed execution snapshots.
        def transform(old):
            code = error.code if isinstance(error, ExperimentRecordError) else "experiment_error"
            return replace(old, failure=ExperimentFailure(old.phase, code, type(error).__name__),
                           lifecycle="terminal" if _settled(old) else "failed", outcome="inconclusive")
        return self._update(ref, transform)
