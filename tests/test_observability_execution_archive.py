"""Observe real public execution archives, including shared and unfinished calls."""
from dataclasses import replace
import hashlib
import json

import pytest

import test_integrated_campaign as composed
from aka.contracts.execution import CommandSpec, ExecutionFacts, JobResult, ResourceIdentity
from aka.contracts.execution_recovery import JobReference
from aka.execution.gateway.jobs import result_from_job
from aka.experiment.measurement_store import MeasurementStore, private_write
from observability import light_agate_calls, light_gateway_jobs, export_light_gateway_record
from observability._internal.common import job_attempt_rows


def test_command_raw_job_survives_output_extraction_and_export(archive, tmp_path):
    from test_session_and_command_receipts import command_receipt
    campaign, _, _, _, _ = archive
    store = MeasurementStore(campaign / "measurements")
    executions = store.execution_records()
    job = {"job_id": "dv_full_raw", "status": "succeeded", "result": {
        "exit_code": 1, "stdout": "Traceback\nAssertionError: cuda err 201", "stderr": ""}}
    receipt = command_receipt(job, job["result"]["stdout"], "[sandbox] using Agate SDK")
    with store.reserve({"operation": "dev"}, {}, request_id="command-raw") as task:
        with executions.operation(task.record_id) as ref:
            executions.register(ref, task.record_id, {"operation": "dev"}, kind="command", purpose="development")
            executions.prepared(ref, [CommandSpec(tmp_path, (), operation="dev")])
            attempt = executions.intent(ref, 0)
            executions.observed(ref, attempt, result=receipt)
        task.finish({"status": "failed", "result": {"exit_code": 1}}, cacheable=False)
        record_id = task.record_id
    exported = export_light_gateway_record.snapshot(campaign, record_id)
    assert exported["agate_attempts"][0]["agate_response"] == job


def _files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


@pytest.fixture
def archive():
    case = composed.ComposedCampaignTest()
    case.setUp()
    try:
        campaign = case.run_campaign("fast", ())
        store = MeasurementStore(campaign / "measurements")
        executions = store.execution_records()
        jobs = [
            {"job_id": "ev_failed", "status": "failed", "error": {"error_class": "candidate", "reason": "compile_failed"}},
            {"job_id": "ev_passed", "status": "succeeded", "result": {"exit_code": 0}},
            {"job_id": "ev_running", "status": "running", "result": None},
        ]
        jobs[1].update(created_at="2026-11-17T00:00:00Z", started_at="2026-11-17T00:00:03Z",
                       finished_at="2026-11-17T00:00:13Z", updated_at="2026-11-17T00:00:13Z")
        payloads = [{"kind": "run", "request": {"lock_clocks": True,
            "options": {"shape_ids": [str(index)], "shape_shards": 1}}} for index in range(3)]
        with store.reserve({"operation": "evaluate"}, {}, request_id="creator") as task:
            directory = task.directory
            with executions.operation(task.record_id) as ref:
                executions.register(ref, task.record_id, {"operation": "run"}, kind="command", purpose="development")
                commands = [CommandSpec(case.root, (), submission=json.dumps(payload).encode(), hardware="L20N")
                            for payload in payloads]
                executions.prepared(ref, commands)
                for command in commands:
                    executions.publish_content(ref, command.submission, "evaluation_input")
                for index, job in zip((0, 0, 1), jobs):
                    attempt = executions.intent(ref, index)
                    result = result_from_job(job, JobReference("gateway", job["job_id"]))
                    executions.observed(ref, attempt, result=result)
                # Repeated GET observations are not additional GPU Jobs.
                record = executions.get(ref)
                running = record.attempts[-1]
                executions.observed(ref, running.attempt_id, result=replace(running.result,
                    attempts=(*running.result.attempts, *running.result.attempts)))
                rejected = {"status": "failed", "http_status": 422,
                    "error": {"error_class": "validation", "reason": "request_validation_failed",
                              "details": {"validation_errors": [{"msg": "invalid field"}]}}}
                attempt = executions.intent(ref, 2)
                executions.observed(ref, attempt, result=JobResult(2, stdout=json.dumps(rejected),
                    resource=ResourceIdentity("gateway"), transport_status="rejected",
                    facts=ExecutionFacts("not_submitted")))
            record = task.finish({"status": "running", "result": {"correct": None}}, cacheable=False, pending=True)
        record_id = record["gateway_record_id"]
        # T1 consumes T2's central Record: one Agate execution, two Runtime calls.
        for number, request_id in ((1, "consumer"), (2, "creator")):
            lineage = campaign / "lineages" / f"t{number:04d}"
            (lineage / "agent-workspace").mkdir(parents=True, exist_ok=True)
            private_write(lineage / "runtime-audit/requests" / f"request-{request_id}.json", {
                "request_id": request_id, "gateway_record_id": record_id, "operation": "gateway",
                "agent_request": {"argv": ["--kind", "evaluate"]},
                "agent_response": {"status": "running", "gateway_record_id": record_id},
            })
        yield campaign, directory, record_id, jobs, payloads
    finally:
        case.doCleanups()


def test_agate_counts_filters_and_export_read_central_execution_once(archive, tmp_path):
    campaign, directory, record_id, jobs, payloads = archive
    before = _files(campaign)
    status = light_agate_calls.snapshot(campaign, kinds=("evaluate",), last=20)
    calls = status["calls"]
    assert len(calls) == 4, calls
    assert {row["scope"] for row in calls} == {"t0002"}
    assert {row["status"] for row in calls} == {"succeeded", "failed", "pending"}
    passed = next(row for row in calls if row["agate_job_id"] == "ev_passed")
    assert (passed["queue_seconds"], passed["execution_seconds"], passed["total_seconds"]) == (3, 10, 13)
    failures = light_agate_calls.snapshot(campaign, kinds=("evaluate",), statuses=("failed",), last=10)["calls"]
    assert len(failures) == 2
    linked = light_gateway_jobs.snapshot(campaign, record_id)
    assert linked["summary"]["submission_attempt_count"] == 4
    assert linked["summary"]["agate_job_id_count"] == 3
    evidence = export_light_gateway_record.snapshot(campaign, record_id)
    assert len(evidence["agate_attempts"]) == 4
    exported = next(item for item in evidence["agate_attempts"] if item["agate_job_id"] == "ev_passed")
    assert exported["agate_response"] == jobs[1]
    assert exported["agate_sdk_submission_payload"] == payloads[0]["request"]
    rejected = next(item for item in evidence["agate_attempts"] if item["agate_job_id"] is None)
    assert rejected["agate_response"]["http_status"] == 422
    assert rejected["agate_response"]["error"]["details"]["validation_errors"][0]["msg"] == "invalid field"
    assert evidence["agent_response"]["status"] == "running"
    export_light_gateway_record.export_gateway_record(campaign, record_id, tmp_path / "export")
    assert _files(campaign) == before


def test_corrupt_new_archive_is_not_silently_reported_as_zero_jobs(archive):
    campaign, directory, record_id, *_ = archive
    executions = MeasurementStore(campaign / "measurements").execution_records()
    ref = executions.find(record_id).ref
    path = executions.root / ref.experiment_id / "record.json"
    envelope = json.loads(path.read_text())
    envelope["record"]["request_id"] = "tampered"
    path.write_text(json.dumps(envelope))
    from aka.contracts.experiment import ExperimentRecordError
    with pytest.raises(ExperimentRecordError, match="record_corrupt"):
        job_attempt_rows(directory)
