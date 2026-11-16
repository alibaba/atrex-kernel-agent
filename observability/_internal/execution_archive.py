"""Read physical Agate attempts from the Campaign's public execution contract."""
import json
from pathlib import Path

from aka.contracts.execution import JobResult
from aka.experiment.file_records import FileExperimentRecords


def _object(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def execution_attempt_rows(record_directory: Path, *, latest=False) -> list[dict]:
    """Resolve Gateway ID → ExperimentRef without scanning other executions.

    Use the validated, read-only Record/Artifact API. Never reconstruct a raw
    Agate response from normalized facts; where only facts survive, exports
    retain those separately and report the raw response as unavailable.
    """
    if record_directory.parent.name != "records":
        return []
    root = record_directory.parent.parent
    if not (root / "executions").is_dir():
        return []
    records = FileExperimentRecords(root / "executions", "campaign", create=False,
                                    namespace_directory=False, artifact_root=root)
    try:
        record = records.find(record_directory.name)
    except LookupError:
        return []
    path = records.root / record.ref.experiment_id / "record.json"
    submissions = []
    receipts = {}
    for artifact in record.artifacts:
        if artifact.role not in {"evaluation_input", "measurement_batch", "measurement_receipt"}:
            continue
        value = _object(records.read_artifact(artifact))
        if not value:
            continue
        if artifact.role == "evaluation_input" and isinstance(value.get("request"), dict):
            submissions.append(value)
        elif value.get("job_id"):
            receipts[value["job_id"]] = value
    # Resuming republishes the same immutable plan. Only map by position when
    # every step has a prepared payload and any later blocks match that plan.
    count = len(record.steps)
    mapped = count and len(submissions) >= count and all(
        value == submissions[index % count] for index, value in enumerate(submissions))
    payloads = {step.task_id: submissions[index] for index, step in enumerate(record.steps)} if mapped else {}
    steps = {step.task_id: step for step in record.steps}
    gateway = _object((record_directory / "request.json").read_text()) if (record_directory / "request.json").is_file() else {}
    gateway_record = _object((record_directory / "record.json").read_text()) if (record_directory / "record.json").is_file() else {}
    request_id = (gateway_record or {}).get("runtime_request_id") or (gateway or {}).get("runtime_request_id")
    rows = []
    submissions_seen = {}
    for attempt in record.attempts:
        result = attempt.result
        if result is None:
            continue  # A local intent alone does not establish an Agate call.
        raw = _object(result.stdout) if result.output_scope == "transport" else None
        if raw and raw.get("job_id"):
            receipts[raw["job_id"]] = raw
        physical = [item for item in result.attempts if item.resource and item.resource.backend == "gateway"]
        if not physical:
            if not (raw and (raw.get("job_id") or raw.get("error"))) and not (
                result.resource and result.resource.backend == "gateway"
            ):
                continue
            # Some terminal adapters persist only the JobResult, without an
            # inner ExecutionAttempt. The outer result is still evidence.
            from aka.contracts.execution import ExecutionAttempt
            physical = [ExecutionAttempt(steps[attempt.task_id].operation, result.facts.disposition,
                result.resource, result.facts.termination, result.exit_code,
                result.diagnostics, causes=result.causes)]
        for index, item in enumerate(physical):
            captured = JobResult(0, attempts=(item,)).as_dict()["attempts"][0]
            identifier = item.resource.identifier if item.resource else None
            if not identifier and len(physical) == 1 and raw:
                identifier = raw.get("job_id")
            raw_job = receipts.get(identifier) if identifier else raw if len(physical) == 1 else None
            # Inner attempts retain failed/retried Job IDs even when their raw
            # response is absent from the final stdout.
            job = dict(raw_job or {})
            if identifier:
                job.setdefault("job_id", identifier)
            observations = [dict(d.details) for d in result.diagnostics
                            if d.code == "gateway-job-wait" and d.details.get("identifier") == identifier]
            if not job.get("status") and observations:
                job["status"] = observations[-1].get("status")
                job["execution_phase"] = observations[-1].get("execution_phase")
            if not job.get("status") and item.disposition == "terminal":
                job["status"] = ("cancelled" if item.termination == "cancelled" else
                                 "succeeded" if item.exit_code == 0 else
                                 "failed" if item.exit_code is not None or item.causes else "unknown")
            if not job.get("error") and item.causes:
                cause = item.causes[0]
                job["error"] = {"error_class": cause.category, "reason": cause.code,
                                "message": cause.message, "details": captured["causes"][0]["details"]}
            phase = ("rejected" if item.disposition == "not_submitted" else
                     "terminal" if item.disposition == "terminal" or job.get("status") in
                     {"succeeded", "failed", "cancelled"} else "accepted" if identifier else "submitting")
            physical_key = identifier or item.attempt_id or (attempt.attempt_id, index)
            serials = submissions_seen.setdefault(attempt.task_id, {})
            serials.setdefault(physical_key, len(serials) + 1)
            submission = serials[physical_key]
            prepared = payloads.get(attempt.task_id, {})
            current = index == len(physical) - 1
            process = {**result.as_dict(), "returncode": result.returncode} if current else None
            rows.append({
                "path": str(path), "checkpoint_path": str(path.relative_to(root)),
                "checkpoint_id": f"{record.ref.experiment_id}/{attempt.task_id}",
                "checkpoint_updated_at": job.get("updated_at") or job.get("finished_at") or record.updated_at,
                "phase": phase, "submission": submission,
                "retries": {"physical_resubmissions": submission - 1},
                "identity": {"kind": prepared.get("kind", steps[attempt.task_id].operation),
                             "payload": prepared.get("request")},
                "runtime_request_ids": [request_id] if request_id else [],
                "phase_observations": observations, "process": process,
                "job": job, "raw_job": raw_job, "execution_attempt": captured,
            })
    # Recovery queries can add observations of the same physical Job. Prefer
    # terminal evidence over a later failed GET; keep no-ID submissions apart.
    unique = {}
    for index, row in enumerate(rows):
        key = row["job"].get("job_id") or (row["checkpoint_id"], row["submission"], index)
        old = unique.get(key)
        if old is None or row["phase"] == "terminal" or old["phase"] != "terminal":
            unique[key] = row
    rows = list(unique.values())
    if latest:
        rows = list({row["checkpoint_id"]: row for row in rows}.values())
    return rows
