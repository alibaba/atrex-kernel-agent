from pathlib import Path
from types import SimpleNamespace

from observability import (
    common,
    export_light_source,
    light_acceptance,
    light_gateway_jobs,
    light_latency_status,
    light_research_status,
    light_runtime_status,
    submissions,
)
from observability.light_wiki_usage import _journal_adoption
from aka.light.experiment.store import private_write


def test_submission_acceptance_uses_private_journal_and_exact_eval(monkeypatch, tmp_path):
    commit = "a" * 40
    record_id = "gateway-" + "b" * 32
    workspace = tmp_path / "agent"
    audit = tmp_path / "private"
    workspace.mkdir()
    journal = audit / "journals" / "episodes" / "e00000002" / "journal.json"
    private_write(journal, {"episode": 2, "kernel_submissions": [{
        "kernel_commit_id": commit, "accepted": True,
        "evaluation_evidence": {"status": "passed", "gateway_record_id": record_id},
    }]})

    class Store:
        def read(self, value):
            assert value == record_id
            return {"kernels": {"candidate": {"kernel_id": commit}}}

    scope = SimpleNamespace(label="campaign", trajectory=None, workspace=workspace, audit_root=audit)
    monkeypatch.setattr(submissions, "discover_scopes", lambda _: [scope])
    monkeypatch.setattr(submissions, "existing_store", lambda _: Store())
    monkeypatch.setattr(submissions, "eligible_full_eval", lambda _: {
        "all_pass": True, "latency_us_geomean": 4.0, "latency_us_by_shape": {"0": 4.0},
    })
    monkeypatch.setattr(submissions, "full_eval_rows", lambda _: [])
    result = light_acceptance.snapshot(workspace)
    assert result["submissions"][0]["accepted"] is True
    assert result["submissions"][0]["evidence_status"] == "verified"
    assert result["lineage_best"][0]["kernel_commit_id"] == commit


def test_source_export_reads_agent_commit_from_measurement_store(monkeypatch, tmp_path):
    commit = "c" * 40
    source = b"# source snapshot\n"

    class Store:
        def read_kernel(self, value):
            assert value == commit
            return source

    monkeypatch.setattr(
        export_light_source,
        "existing_stores",
        lambda *_: [(tmp_path / "agent", Store())],
    )
    monkeypatch.setattr(export_light_source, "measurement_workspaces", lambda *_: [])
    located = export_light_source.locate_git_commit(tmp_path, tmp_path / "private", commit)
    assert located.snapshot == source
    assert located.identity == {"kernel_commit_id": commit}


def test_research_status_rebuilds_queue_without_writing(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    event = {
        "sequence": 1,
        "type": "direction.observed",
        "origin": {"trajectory": 1, "episode": 2},
        "payload": {
            "direction": {
                "direction_id": "direction_example",
                "name": "Explore vector width families",
                "status": "proposed",
                "access": "local_writable",
            }
        },
    }
    (root / "events.jsonl").write_text(__import__("json").dumps(event) + "\n")
    state = light_research_status._research_state(root)
    assert state["queue"] == ["direction_example"]
    assert state["directions"]["direction_example"]["origin"]["episode"] == 2
    assert not (root / "lock").exists()


def test_wiki_usage_reads_supervisor_private_journal(tmp_path):
    workspace = tmp_path / "agent"
    audit = tmp_path / "private"
    workspace.mkdir()
    journal = audit / "journals" / "episodes" / "e00000003" / "journal.json"
    private_write(
        journal,
        {
            "experiments": [
                {
                    "experiment_id": "experiment_example",
                    "wiki_query_ids": ["query-1"],
                    "wiki_usage": [
                        {"query_id": "query-1", "wiki_id": "wiki-1", "use": "adopted"}
                    ],
                }
            ]
        },
    )
    query_ids, wiki_ids, usages = _journal_adoption(
        SimpleNamespace(workspace=workspace, audit_root=audit, label="campaign")
    )
    assert query_ids == {"query-1"}
    assert wiki_ids == {"wiki-1"}
    assert usages[0]["experiment_id"] == "experiment_example"


def test_acceptance_is_compatibility_alias():
    from observability import light_submission_status
    assert light_acceptance.snapshot is light_submission_status.snapshot


def test_current_episode_zero_session_is_not_startup_framework_stage(tmp_path):
    (tmp_path / ".atrex_long_horizon" / "episodes" / "e0000").mkdir(
        parents=True
    )
    live = tmp_path / "memory" / "live.json"
    live.parent.mkdir()
    live.write_text('{"episode_branch":"framework-baseline"}\n')
    assert common.startup_stage(tmp_path) == "bootstrap"


def test_research_review_summary_reads_audit_outcome_result(tmp_path):
    audit_root = tmp_path / "private"
    outcome = (
        audit_root
        / "journals"
        / "episodes"
        / "e00000002"
        / "claim-audit"
        / "claimaudit_example"
        / "outcome.json"
    )
    private_write(
        outcome,
        {
            "status": "archived",
            "result": {
                "review": {"claim_review_quality": "partially_supported"}
            },
        },
    )
    summary = light_research_status._claim_review_summary(
        SimpleNamespace(audit_root=audit_root)
    )
    assert summary["status_counts"] == {"archived": 1}
    assert summary["quality_counts"] == {"partially_supported": 1}


def test_runtime_status_reads_episode_zero_attempt_without_legacy_export(tmp_path):
    runtime = tmp_path / ".atrex_long_horizon"
    episode = runtime / "episodes" / "e0000"
    episode.mkdir(parents=True)
    (runtime / "public-identity.json").write_text(
        '{"schema":1,"campaign_id":"campaign-example"}\n'
    )
    (runtime / "state.json").write_text(
        '{"episodes":0,"attempts":[{"episode":0,"memory_version":1,'
        '"status":"candidate_ready","accepted":true}]}\n'
    )
    (episode / "attempt.json").write_text(
        '{"episode":0,"memory_version":1,"status":"candidate_ready",'
        '"accepted":true}\n'
    )
    (tmp_path / "trace-retention-manifest.json").write_text(
        '{"status":"completed"}\n'
    )
    result = light_runtime_status.snapshot(tmp_path)
    scope = result["scopes"][0]
    assert result["schema_version"] == 3
    assert scope["campaign_id"] == "campaign-example"
    assert scope["episodes"][0]["episode"] == 0
    assert "accepted" not in scope["episodes"][0]
    assert "version" not in scope["episodes"][0]


def test_job_attempt_rows_retains_failed_retry_and_success(tmp_path):
    import json

    record = tmp_path / "gateway-" / "repetition-0001"
    job = record / "jobs" / "checkpoint"
    job.mkdir(parents=True)

    def write(name, submission, phase, job_id, status, reason=None):
        response = {"job_id": job_id, "status": status}
        if reason:
            response["error"] = {
                "error_class": "infra",
                "reason": reason,
                "message": "retryable failure",
            }
        value = {
            "phase": phase,
            "submission": submission,
            "retries": {"infra": max(0, submission - 1)},
            "identity": {"kind": "eval", "shape_ids": ["0", "1"]},
            "process": {
                "returncode": 0 if status == "succeeded" else 1,
                "stdout": json.dumps(response),
                "stderr": "",
            },
        }
        (job / name).write_text(json.dumps(value))
        return value

    write("submission-0001-accepted.json", 1, "accepted", "ev_failed", "queued")
    write(
        "submission-0001-terminal.json",
        1,
        "terminal",
        "ev_failed",
        "failed",
        "job_deadline_exceeded",
    )
    final = write("submission-0002-terminal.json", 2, "terminal", "ev_ok", "succeeded")
    (job / "state.json").write_text(json.dumps(final))

    rows = common.job_attempt_rows(record)
    assert [(row["submission"], common.job_identifier(row["job"])) for row in rows] == [
        (1, "ev_failed"),
        (2, "ev_ok"),
    ]
    assert [common.normalized_job_status(row) for row in rows] == ["failed", "succeeded"]
    assert (
        common.job_error_details(rows[0]["job"])["reason"]
        == "job_deadline_exceeded"
    )


def test_gateway_jobs_snapshot_reports_status_and_error(monkeypatch, tmp_path):
    import json

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    private = tmp_path / "private"
    record_id = "gateway-" + "a" * 32
    record = private / "measurements" / "records" / record_id
    job = record / "repetition-0001" / "jobs" / "checkpoint"
    job.mkdir(parents=True)
    (record / "request.json").write_text('{"operation":"evaluate"}\n')
    (record / "record.json").write_text(
        '{"operation":"evaluate","cacheable":true,"response":{}}\n'
    )
    state = {
        "phase": "terminal",
        "submission": 1,
        "retries": {},
        "identity": {"kind": "eval", "shape_ids": ["4", "5"]},
        "process": {
            "returncode": 1,
            "stdout": json.dumps(
                {
                    "job_id": "ev_failed",
                    "status": "failed",
                    "error": {
                        "error_class": "timeout",
                        "reason": "job_deadline_exceeded",
                        "message": "job deadline exceeded",
                    },
                }
            ),
            "stderr": "",
        },
    }
    (job / "submission-0001-terminal.json").write_text(json.dumps(state))
    (job / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(common, "supervisor_campaign_root", lambda _: private)

    value = light_gateway_jobs.snapshot(workspace, record_id)
    assert value["summary"] == {
        "record_location_count": 1,
        "submission_attempt_count": 1,
        "physical_job_id_count": 1,
        "reused_attempt_count": 0,
        "status_counts": {"failed": 1},
    }
    attempt = value["attempts"][0]
    assert attempt["job_id"] == "ev_failed"
    assert attempt["status"] == "failed"
    assert attempt["shape_ids"] == ["4", "5"]
    assert attempt["error"]["reason"] == "job_deadline_exceeded"
