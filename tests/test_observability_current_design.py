import json
from pathlib import Path
from types import SimpleNamespace

from observability import (
    export_light_gateway_record,
    light_agate_calls,
    light_gateway_jobs,
    light_kernel_history,
    light_lineage_status,
    light_runtime_calls,
)
from observability._internal import (
    agate_state as light_agate_status,
    common,
    research_state as light_research_status,
    runtime_audit as light_tool_usage,
    runtime_state as light_runtime_status,
    source_export as export_light_source,
    submissions,
)
from aka.light.experiment.store import private_write


def test_kernel_history_nests_submissions_without_repeating_performance(
    monkeypatch, tmp_path
):
    commit = "a" * 40
    record = "gateway-" + "b" * 32
    evaluation = {
        "scope": "t0001",
        "trajectory": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "kernel_commit_id": commit,
        "gateway_record_id": record,
        "latency_us_geomean": 4.0,
        "latency_us_arith_mean": 4.2,
        "shape_count": 2,
        "latency_us_by_shape": {"0": 3.0, "1": 5.0},
    }
    submission = {
        "scope": "t0001",
        "trajectory": 1,
        "episode": 2,
        "submission_index": 1,
        "submission_id": "submission-a",
        "submitted_at": "2026-01-01T00:01:00Z",
        "kernel_commit_id": commit,
        "gateway_record_id": record,
        "accepted": True,
        "evidence_status": "verified",
        "review": {"source_contract": "pass"},
        "implementation_review": {
            "availability": "available",
            "technologies": ["triton"],
            "dsl_policy": "open",
            "configured_dsl": None,
            "required_phase": "open_dsl",
            "evaluator_supported": True,
            "configured_dsl_used": False,
            "required_phase_used": False,
            "unexpected_core_technologies": [],
        },
        "diagnostics": [],
        "token_checkpoint": {"checkpoint_id": "submission-a"},
    }
    monkeypatch.setattr(
        light_kernel_history,
        "campaign_submission_snapshot",
        lambda *_args: {
            "full_evaluations": [evaluation],
            "submissions": [submission],
            "accepted_submissions": [submission],
        },
    )
    monkeypatch.setattr(
        light_kernel_history,
        "token_snapshot",
        lambda *_args: {
            "token_checkpoints": [
                {
                    "scope": "t0001",
                    "event": "passing-full-evaluate",
                    "checkpoint_id": record,
                    "identity": {
                        "gateway_record_id": record,
                        "kernel_commit_id": commit,
                    },
                    "usage": {"total_tokens": 100},
                    "main_agent_usage": {"total_tokens": 90},
                    "review_usage": {"total_tokens": 10},
                },
                {
                    "scope": "t0001",
                    "event": "accepted-submit-kernel",
                    "checkpoint_id": "submission-a",
                    "identity": {
                        "gateway_record_id": record,
                        "kernel_commit_id": commit,
                    },
                    "usage": {"total_tokens": 120},
                    "main_agent_usage": {"total_tokens": 100},
                    "review_usage": {"total_tokens": 20},
                },
            ]
        },
    )

    value = light_kernel_history.snapshot(tmp_path)
    result = value["lineages"][0]["kernel_results"][0]
    nested_submission = result["submissions"][0]

    assert value["schema_version"] == 2
    assert result["performance"]["latency_us_by_shape"] == {"0": 3.0, "1": 5.0}
    assert result["evaluate_token_usage"]["all_paths"]["total_tokens"] == 100
    assert nested_submission["submit_token_usage"]["all_paths"]["total_tokens"] == 120
    assert nested_submission["implementation_review"]["technologies"] == ["triton"]
    assert "latency_us_geomean" not in nested_submission
    assert "Implementation" in light_kernel_history.render(value)
    assert "triton" in light_kernel_history.render(value)


def test_kernel_history_keeps_submission_without_eligible_evaluate(
    monkeypatch, tmp_path
):
    submission = {
        "scope": "t0001",
        "trajectory": 1,
        "episode": 2,
        "submission_index": 1,
        "submission_id": None,
        "submitted_at": "2026-01-01T00:01:00Z",
        "kernel_commit_id": "a" * 40,
        "gateway_record_id": None,
        "accepted": False,
        "evidence_status": "not_accepted",
        "review": {"source_contract": "fail"},
        "diagnostics": ["final source policy rejected"],
    }
    monkeypatch.setattr(
        light_kernel_history,
        "campaign_submission_snapshot",
        lambda *_args: {
            "full_evaluations": [],
            "submissions": [submission],
            "accepted_submissions": [],
        },
    )
    monkeypatch.setattr(
        light_kernel_history,
        "token_snapshot",
        lambda *_args: {"token_checkpoints": []},
    )

    result = light_kernel_history.snapshot(tmp_path)["lineages"][0][
        "kernel_results"
    ][0]

    assert result["performance"] is None
    assert result["submissions"][0]["accepted"] is False
    assert result["submissions"][0]["submit_token_usage_availability"] == "not_recorded"


def test_lineage_status_hides_generation_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        light_lineage_status,
        "runtime_snapshot",
        lambda *_args, **_kwargs: {
            "campaign_workspace": str(tmp_path),
            "scheduler_checkpoint": {
                "epoch": 2,
                "options": {"trajectories": 3},
                "lineages": {},
            },
            "scopes": [
                {
                    "scope": "campaign",
                    "trajectory": None,
                    "workspace": str(tmp_path),
                    "liveness": "not_running",
                }
            ],
        },
    )
    monkeypatch.setattr(
        light_lineage_status,
        "research_snapshot",
        lambda *_args, **_kwargs: {"scopes": []},
    )
    monkeypatch.setattr(
        light_lineage_status,
        "token_snapshot",
        lambda *_args, **_kwargs: {"episodes": [], "sessions": [], "reviews": []},
    )

    value = light_lineage_status.snapshot(tmp_path, research_limit=50)

    assert "scheduler_checkpoint" not in value
    assert "detail_selection" not in value
    assert value["campaign_configuration"]["search"] == {"trajectories": 3}


def test_submission_acceptance_uses_central_store_and_exact_eval(monkeypatch, tmp_path):
    commit = "a" * 40
    record_id = "gateway-" + "b" * 32
    workspace = tmp_path / "agent"
    audit = tmp_path / "private"
    workspace.mkdir()
    from aka.light.workflow.submissions import append_submission
    submission_root = submissions.supervisor_submission_root(workspace)
    append_submission(submission_root, {
        "submission_id": "submission-" + "c" * 32,
        "lineage": "T1",
        "episode": 2,
        "kernel_commit_id": commit, "accepted": True,
        "evaluation_evidence": {"status": "passed", "gateway_record_id": record_id},
        "implementation_review": {
            "availability": "available",
            "technologies": ["cuda"],
        },
        })

    class Store:
        def read(self, value):
            assert value == record_id
            return {"kernels": {"candidate": {"kernel_id": commit}}}

    scope = SimpleNamespace(
        label="campaign",
        trajectory=None,
        workspace=workspace,
        audit_root=audit,
        campaign_root=audit,
    )
    monkeypatch.setattr(submissions, "discover_scopes", lambda _: [scope])
    monkeypatch.setattr(submissions, "existing_store", lambda _: Store())
    monkeypatch.setattr(submissions, "eligible_full_eval", lambda _: {
        "all_pass": True, "latency_us_geomean": 4.0, "latency_us_by_shape": {"0": 4.0},
    })
    monkeypatch.setattr(submissions, "full_eval_rows", lambda _: [])
    result = submissions.campaign_submission_snapshot(workspace)
    assert result["submissions"][0]["accepted"] is True
    assert result["submissions"][0]["evidence_status"] == "verified"
    assert result["submissions"][0]["implementation_review"]["technologies"] == ["cuda"]
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


def test_research_module_configuration_is_not_guessed(tmp_path):
    scope = SimpleNamespace(audit_root=tmp_path / "private")

    assert light_research_status._latest_modules(scope) is None


def test_current_episode_zero_session_is_not_startup_framework_stage(tmp_path):
    (tmp_path / ".atrex_long_horizon" / "episodes" / "e0000").mkdir(
        parents=True
    )
    live = tmp_path / "memory" / "live.json"
    live.parent.mkdir()
    live.write_text('{"episode_branch":"framework-baseline"}\n')
    assert common.startup_stage(tmp_path) == "bootstrap"


def test_research_status_reads_progress_from_persistent_agent_workspace(
    monkeypatch, tmp_path
):
    controller = tmp_path / "controller"
    controller.mkdir()
    private = tmp_path / "private"
    agent = private / "agent-workspace"
    agent.mkdir(parents=True)
    (private / "state.json").write_text(
        '{"workspace_mode":"persistent"}\n'
    )
    (agent / "PROGRESS.md").write_text("# Research Board\n")
    scope = common.Scope(controller, private, "campaign", None)
    monkeypatch.setattr(common, "supervisor_campaign_root", lambda _: private)
    monkeypatch.setattr(light_research_status, "discover_scopes", lambda _: [scope])

    result = light_research_status.snapshot(controller)["scopes"][0]

    assert result["agent_workspace"] == str(agent)
    assert result["progress_projection"] == {
        "path": str(agent / "PROGRESS.md"),
        "exists": True,
        "size_bytes": len("# Research Board\n"),
    }


def test_runtime_status_does_not_treat_scheduler_checkpoint_as_live_process(
    monkeypatch, tmp_path
):
    campaign = tmp_path / "campaign"
    trajectory = tmp_path / "t0001"
    campaign.mkdir()
    session = (
        trajectory
        / ".atrex_long_horizon"
        / "episodes"
        / "e0001"
        / "sessions"
        / "run-example"
    )
    session.mkdir(parents=True)
    (trajectory / ".atrex_long_horizon" / "active_episode.json").write_text(
        '{"episode":1,"phase":"running"}\n'
    )
    (session / "token-usage.json").write_text(
        '{"session_id":"session-example","backend":"claude",'
        '"state":"running","started_at":"2099-01-01T00:00:00Z"}\n'
    )
    (session / "conversation.jsonl").write_text("")
    scopes = [
        common.Scope(campaign, tmp_path / "campaign-private", "campaign", None),
        common.Scope(trajectory, tmp_path / "trajectory-private", "t0001", 1),
    ]
    search = {
        "epoch": 1,
        "completed_epochs": 0,
        "options": {"epochs": 5},
        "active": {
            "phase": "running",
            "jobs": [
                {
                    "epoch": 1,
                    "trajectory": 1,
                    "episode": 1,
                    "step": 1,
                    "status": "running",
                }
            ],
        },
    }
    monkeypatch.setattr(light_runtime_status, "discover_scopes", lambda _: scopes)
    monkeypatch.setattr(light_runtime_status, "search_state", lambda _: search)
    monkeypatch.setattr(light_runtime_status, "writer_active", lambda _: False)

    result = light_runtime_status.snapshot(campaign)
    row = result["scopes"][1]

    assert row["writer_active"] is False
    assert row["liveness"] == "not_running"
    assert row["active_session_count"] == 0
    assert result["scheduler_checkpoint"]["source"] == (
        "persisted_scheduler_checkpoint"
    )
    assert result["scheduler_checkpoint"]["active_phase"] == "running"
    assert result["scheduler_checkpoint"]["jobs"][0]["trajectory"] == 1


def test_runtime_status_projects_asynchronous_jobs():
    state = {
        "epoch": 2,
        "completed_epochs": 1,
        "options": {"epochs": 5},
        "active": None,
        "asynchronous": {
            "jobs": {
                "1": {
                    "epoch": 2,
                    "trajectory": 1,
                    "episode": 2,
                    "status": "running",
                },
                "2": None,
            }
        },
    }

    result = light_runtime_status._search_projection(state)

    assert result["active_phase"] is None
    assert result["jobs"] == [
        {
            "epoch": 2,
            "trajectory": 1,
            "episode": 2,
            "step": None,
            "status": "running",
        }
    ]


def test_runtime_status_applies_campaign_writer_lease_to_lineages(
    monkeypatch, tmp_path
):
    campaign = tmp_path / "campaign"
    lineage = campaign / "lineages" / "t0001"
    agent_workspace = lineage / "agent-workspace"
    session = (
        lineage
        / "segments"
        / "episodes"
        / "e0001"
        / "sessions"
        / "run-example"
    )
    agent_workspace.mkdir(parents=True)
    session.mkdir(parents=True)
    (campaign / "campaign.json").write_text("{}\n")
    (lineage / "state.json").write_text(
        '{"trajectory":1,"segment_status":"active","completed_episodes":0}\n'
    )
    (lineage / "segments" / "active_episode.json").write_text(
        '{"episode":1,"phase":"running"}\n'
    )
    (session / "token-usage.json").write_text(
        '{"session_id":"session-example","backend":"claude",'
        '"state":"running","started_at":"2099-01-01T00:00:00Z"}\n'
    )
    (session / "conversation.jsonl").write_text("")
    scopes = [
        common.Scope(campaign, campaign, "campaign", None),
        common.Scope(agent_workspace, lineage, "t0001", 1),
    ]
    monkeypatch.setattr(light_runtime_status, "discover_scopes", lambda _: scopes)
    monkeypatch.setattr(light_runtime_status, "search_state", lambda _: None)
    monkeypatch.setattr(
        light_runtime_status, "campaign_runtime_configuration", lambda _: None
    )
    observed = []

    def active(path):
        observed.append(Path(path))
        return Path(path) == campaign

    monkeypatch.setattr(light_runtime_status, "writer_active", active)

    result = light_runtime_status.snapshot(campaign)
    row = result["scopes"][1]

    assert observed == [campaign]
    assert row["writer_active"] is True
    assert row["liveness"] == "running_recent_activity"
    assert row["active_session_count"] == 1


def test_runtime_status_does_not_mark_completed_idle_lineage_as_stalled(
    monkeypatch, tmp_path
):
    campaign = tmp_path / "campaign"
    lineage = campaign / "lineages" / "t0001"
    agent_workspace = lineage / "agent-workspace"
    segments = lineage / "segments"
    agent_workspace.mkdir(parents=True)
    segments.mkdir()
    (campaign / "campaign.json").write_text("{}\n")
    (lineage / "state.json").write_text(
        '{"trajectory":1,"segment_status":"idle","completed_episodes":1}\n'
    )
    (segments / "state.json").write_text('{"episodes":1}\n')
    scopes = [
        common.Scope(campaign, campaign, "campaign", None),
        common.Scope(agent_workspace, lineage, "t0001", 1),
    ]
    monkeypatch.setattr(light_runtime_status, "discover_scopes", lambda _: scopes)
    monkeypatch.setattr(light_runtime_status, "search_state", lambda _: None)
    monkeypatch.setattr(
        light_runtime_status, "campaign_runtime_configuration", lambda _: None
    )
    monkeypatch.setattr(light_runtime_status, "writer_active", lambda _: True)

    result = light_runtime_status.snapshot(campaign)
    row = result["scopes"][1]

    assert row["completed_episodes"] == 1
    assert row["writer_active"] is False
    assert row["liveness"] == "not_running"
    assert row["active_session_count"] == 0


def test_tool_usage_normalizes_wiki_question_to_stable_tool_name():
    assert light_tool_usage._tool("wiki", ["a long natural-language question"]) == (
        "wiki-query"
    )


def test_tool_usage_returns_only_authoritative_audit_facts(monkeypatch, tmp_path):
    scope = SimpleNamespace(
        label="campaign", trajectory=None, audit_root=tmp_path / "private"
    )
    scope.audit_root.mkdir()
    private_write(
        scope.audit_root / "request-example.json",
        {"operation": "gateway", "argv": ["--kind", "evaluate"]},
    )
    monkeypatch.setattr(light_tool_usage, "discover_scopes", lambda _: [scope])

    result = light_tool_usage.snapshot(tmp_path)

    assert "session_attribution" not in result
    assert "attribution_note" not in result
    assert result["calls"][0]["success"] is None
    assert result["summary"][0]["failures"] == 0
    assert light_tool_usage._status_success("running") is None
    assert light_tool_usage._status_success("failed") is False


def test_agate_status_deduplicates_checkpoint_references_to_one_job():
    rows = [
        {
            "path": "/checkpoint/reused/state.json",
            "checkpoint_id": "reused",
            "phase": "terminal",
            "submission": 0,
            "checkpoint_updated_at": "2026-01-01T00:00:00Z",
            "job": {"job_id": "ev_example", "status": "succeeded"},
        },
        {
            "path": "/checkpoint/submitted/state.json",
            "checkpoint_id": "submitted",
            "phase": "terminal",
            "submission": 1,
            "checkpoint_updated_at": "2026-01-01T00:00:01Z",
            "job": {"job_id": "ev_example", "status": "succeeded"},
        },
    ]

    unique, references = light_agate_status._deduplicate_states(rows)

    assert references == 2
    assert len(unique) == 1
    assert unique[0]["submission"] == 1
    assert len(unique[0]["checkpoint_references"]) == 2


def test_agate_status_default_projection_returns_only_compact_issues():
    successful_job = {
        "scope": "campaign",
        "trajectory": None,
        "source": "measurement_record",
        "gateway_record_id": "gateway-" + "a" * 32,
        "checkpoint_id": "success-checkpoint",
        "operation": "eval",
        "job_id": "ev_success",
        "checkpoint_phase": "terminal",
        "status": "succeeded",
        "raw_status": "Succeeded",
        "execution_phase": "complete",
        "submission": 1,
        "retry_count": 0,
        "checkpoint_references": [{"path": "/large/private/reference"}],
        "timing": {"total_seconds": 3.0},
        "pending_seconds": None,
        "error": None,
    }
    failed_job = {
        **successful_job,
        "gateway_record_id": "gateway-" + "b" * 32,
        "checkpoint_id": "failed-checkpoint",
        "job_id": "ev_failed",
        "status": "failed",
        "raw_status": "Failed",
        "error": {"category": "candidate", "reason": "compile_failed"},
    }
    value = {
        "summary": {"current_job_status_counts": {"failed": 1, "succeeded": 1}},
        "calls": [
            {
                "gateway_record_id": successful_job["gateway_record_id"],
                "complete": True,
                "outcome": "completed",
                "error": None,
            },
            {
                "gateway_record_id": failed_job["gateway_record_id"],
                "complete": True,
                "outcome": "failed",
                "error": failed_job["error"],
            },
        ],
        "current_jobs": [successful_job, failed_job],
        "submission_attempts": [successful_job, failed_job],
        "pending_jobs": [],
    }

    projected = light_agate_status.project_details(value)

    assert [row["gateway_record_id"] for row in projected["calls"]] == [
        failed_job["gateway_record_id"]
    ]
    assert projected["current_jobs"] == [
        {field: failed_job.get(field) for field in light_agate_status._COMPACT_JOB_FIELDS}
    ]
    assert "checkpoint_references" not in projected["current_jobs"][0]
    assert projected["detail_selection"]["omitted_counts"] == {
        "logical_calls": 1,
        "current_jobs": 1,
        "submission_attempts": 1,
    }
    assert projected["summary"] == value["summary"]


def test_agate_status_expands_all_or_selected_gateway_record_details():
    first = {
        "gateway_record_id": "gateway-" + "a" * 32,
        "status": "succeeded",
        "checkpoint_references": [{"path": "/first"}],
    }
    second = {
        "gateway_record_id": "gateway-" + "b" * 32,
        "status": "succeeded",
        "checkpoint_references": [{"path": "/second"}],
    }
    value = {
        "calls": [first, second],
        "current_jobs": [first, second],
        "submission_attempts": [first, second],
        "pending_jobs": [],
    }

    expanded = light_agate_status.project_details(value, jobs="all")
    selected = light_agate_status.project_details(
        value, gateway_record_ids=(second["gateway_record_id"],)
    )

    assert expanded["current_jobs"] == [first, second]
    assert selected["calls"] == [second]
    assert selected["current_jobs"] == [second]
    assert selected["detail_selection"]["mode"] == "gateway_records"


def test_agate_status_default_projection_renders_pending_job_without_status():
    pending = {
        "scope": "campaign",
        "trajectory": None,
        "source": "measurement_record",
        "gateway_record_id": "gateway-" + "c" * 32,
        "operation": "eval",
        "job_id": "ev_pending",
        "checkpoint_phase": "accepted",
        "status": None,
        "execution_phase": "queued",
        "submission": 1,
        "retry_count": 0,
        "timing": {"total_seconds": None},
        "pending_seconds": 12.0,
        "error": None,
    }
    value = {
        "summary": {
            "operation_counts": {"eval": 1},
            "trusted_operation_counts": {},
            "current_checkpoint_job_count": 1,
            "submission_attempt_count": 1,
            "historical_retry_attempt_count": 0,
            "retry_count": 0,
            "duplicate_cache_hits": 0,
            "same_request_running": 0,
        },
        "calls": [],
        "current_jobs": [pending],
        "submission_attempts": [pending],
        "pending_jobs": [pending],
    }

    projected = light_agate_status.project_details(value)

    assert "ev_pending" in light_agate_status.render(projected)
    assert projected["pending_jobs"][0]["checkpoint_phase"] == "accepted"


def test_agate_status_last_selects_recent_logical_calls_and_associated_jobs():
    def call(index):
        return {
            "gateway_record_id": "gateway-" + f"{index:032x}",
            "created_at": f"2026-01-01T00:00:0{index}+00:00",
            "complete": True,
            "outcome": "completed",
            "error": None,
        }

    calls = [call(1), call(3), call(2)]
    jobs = [
        {
            "gateway_record_id": row["gateway_record_id"],
            "status": "succeeded",
            "error": None,
            "pending_seconds": None,
        }
        for row in calls
    ]
    value = {
        "calls": calls,
        "current_jobs": jobs,
        "submission_attempts": jobs,
        "pending_jobs": [],
    }

    projected = light_agate_status.project_details(value, last=2)

    expected = [call(3)["gateway_record_id"], call(2)["gateway_record_id"]]
    assert [row["gateway_record_id"] for row in projected["calls"]] == expected
    assert [row["gateway_record_id"] for row in projected["current_jobs"]] == expected
    assert projected["detail_selection"]["mode"] == "recent"
    assert projected["detail_selection"]["summary_scope"] == "campaign"


def test_agate_status_filters_failed_retry_even_when_logical_call_succeeded():
    record_id = "gateway-" + "d" * 32
    successful_call = {
        "gateway_record_id": record_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "complete": True,
        "outcome": "completed",
        "error": None,
    }
    current = {
        "gateway_record_id": record_id,
        "status": "succeeded",
        "error": None,
        "pending_seconds": None,
    }
    failed_attempt = {
        "gateway_record_id": record_id,
        "status": "failed",
        "error": {"category": "infrastructure", "reason": "queue_timeout"},
        "pending_seconds": None,
    }
    value = {
        "calls": [successful_call],
        "current_jobs": [current],
        "submission_attempts": [failed_attempt, current],
        "pending_jobs": [],
    }

    failed = light_agate_status.project_details(value, statuses=("failed",))
    succeeded = light_agate_status.project_details(value, statuses=("succeeded",))

    assert failed["calls"] == [successful_call]
    assert failed["current_jobs"] == []
    assert len(failed["submission_attempts"]) == 1
    assert failed["submission_attempts"][0]["status"] == "failed"
    assert succeeded["calls"] == [successful_call]
    assert len(succeeded["current_jobs"]) == 1
    assert succeeded["submission_attempts"][0]["status"] == "succeeded"


def test_research_review_summary_reads_audit_outcome_result(tmp_path):
    audit_root = tmp_path / "private"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from aka.light.runtime.paths import supervisor_research_catalog_root
    outcome = (
        supervisor_research_catalog_root(workspace)
        / "reviews"
        / "T1"
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
        SimpleNamespace(
            audit_root=audit_root,
            workspace=workspace,
            trajectory=None,
        )
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
    assert result["schema_version"] == 4
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
    monkeypatch.setattr(
        common, "supervisor_measurement_root", lambda _: private / "measurements"
    )

    value = light_gateway_jobs.snapshot(workspace, record_id)
    assert value["summary"] == {
        "record_location_count": 1,
        "submission_attempt_count": 1,
        "agate_job_id_count": 1,
        "backend_agate_job_id_count": 0,
        "reused_attempt_count": 0,
        "status_counts": {"failed": 1},
    }
    attempt = value["attempts"][0]
    assert attempt["agate_job_id"] == "ev_failed"
    assert attempt["status"] == "failed"
    assert attempt["shape_ids"] == ["4", "5"]
    assert attempt["error"]["reason"] == "job_deadline_exceeded"


def test_gateway_record_export_preserves_runtime_jobs_and_agent_receipt(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    audit = tmp_path / "private"
    audit.mkdir()
    measurement = tmp_path / "measurements"
    output = tmp_path / "gateway-export"
    record_id = "gateway-" + "c" * 32
    record_directory = measurement / "records" / record_id
    record_directory.mkdir(parents=True)
    (record_directory / "request.json").write_text('{"operation":"evaluate"}\n')
    (record_directory / "record.json").write_text(
        '{"runtime_request_id":"req1","response":'
        '{"exit_code":0,"stdout":"agent receipt"}}\n'
    )
    (audit / "request-req1.json").write_text(
        '{"request_id":"req1","operation":"gateway",'
        '"argv":["--kind","evaluate"],"gateway_record_id":"'
        + record_id
        + '"}\n'
    )
    scope = SimpleNamespace(
        label="t0001",
        trajectory=1,
        workspace=workspace,
        audit_root=audit,
        measurement_root=measurement,
    )
    monkeypatch.setattr(
        export_light_gateway_record, "discover_scopes", lambda *_a: [scope]
    )
    monkeypatch.setattr(
        export_light_gateway_record,
        "job_attempt_rows",
        lambda _path: [
            {
                "checkpoint_path": "jobs/eval",
                "checkpoint_id": "eval",
                "phase": "terminal",
                "submission": 1,
                "runtime_request_ids": ["req1"],
                "retries": {},
                "job": {
                    "job_id": "ev_outer",
                    "status": "succeeded",
                    "result": {"backend_job_ids": ["ray_backend_1"]},
                },
                "process": {"returncode": 0, "stdout": "raw agate"},
                "path": "/private/submission.json",
            }
        ],
    )

    result = export_light_gateway_record.export_gateway_record(
        workspace, record_id, output
    )

    assert result["gateway_record_id"] == record_id
    assert result["runtime_request_ids"] == ["req1"]
    assert result["agate_job_ids"] == ["ev_outer"]
    assert result["backend_agate_job_ids"] == ["ray_backend_1"]
    assert result["source_location_count"] == 1
    assert result["runtime_audit_count"] == 1
    assert "agent receipt" in (output / "agent-response.json").read_text()
    assert "raw agate" in (output / "agate-attempts.json").read_text()


def test_central_gateway_record_is_located_once_and_attributed_by_runtime_audit(
    monkeypatch, tmp_path
):
    measurement = tmp_path / "campaign-private" / "measurements"
    record_id = "gateway-" + "d" * 32
    record_directory = measurement / "records" / record_id
    record_directory.mkdir(parents=True)
    (record_directory / "request.json").write_text(
        '{"operation":"evaluate","runtime_request_id":"req-owner"}\n'
    )
    (record_directory / "record.json").write_text(
        '{"operation":"evaluate","runtime_request_id":"req-owner",'
        '"response":{"exit_code":0}}\n'
    )
    campaign_audit = tmp_path / "campaign-private"
    lineage_audit = campaign_audit / "lineages" / "t0001"
    lineage_audit.mkdir(parents=True)
    (lineage_audit / "request-req-owner.json").write_text(
        '{"request_id":"req-owner","gateway_record_id":"' + record_id + '"}\n'
    )
    campaign_scope = SimpleNamespace(
        label="campaign",
        trajectory=None,
        workspace=tmp_path / "controller",
        audit_root=campaign_audit,
        measurement_root=measurement,
    )
    lineage_scope = SimpleNamespace(
        label="t0001",
        trajectory=1,
        workspace=tmp_path / "trajectory",
        audit_root=lineage_audit,
        measurement_root=measurement,
    )
    monkeypatch.setattr(
        export_light_gateway_record,
        "discover_scopes",
        lambda *_a: [campaign_scope, lineage_scope],
    )
    monkeypatch.setattr(
        export_light_gateway_record, "job_attempt_rows", lambda _path: []
    )

    output = tmp_path / "central-export"
    result = export_light_gateway_record.export_gateway_record(
        tmp_path, record_id, output
    )
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["source_location_count"] == 1
    assert result["runtime_audit_count"] == 1
    assert manifest["source_locations"][0]["scope"] == "t0001"


def test_agate_calls_exposes_only_canonical_job_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        light_agate_calls,
        "agate_snapshot",
        lambda *_a: {
            "campaign_workspace": str(tmp_path),
            "submission_attempts": [
                {
                    "operation": "evaluate",
                    "job_id": "ev_example",
                    "status": "succeeded",
                    "error": None,
                    "pending_seconds": None,
                    "checkpoint_updated_at": "2026-01-01T00:00:00Z",
                    "timing": {},
                }
            ],
        },
    )

    call = light_agate_calls.snapshot(tmp_path, last=1)["calls"][0]

    assert call["agate_job_id"] == "ev_example"
    assert "job_id" not in call


def test_runtime_calls_exposes_only_canonical_request_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        light_runtime_calls,
        "audit_snapshot",
        lambda *_a: {
            "campaign_workspace": str(tmp_path),
            "calls": [
                {
                    "request_id": "request-example",
                    "tool": "evaluate",
                    "status": "succeeded",
                    "success": True,
                    "created_at": "2026-01-01T00:00:00Z",
                    "duration_seconds": 1.0,
                }
            ],
        },
    )

    call = light_runtime_calls.snapshot(tmp_path, last=1)["calls"][0]

    assert call["runtime_request_id"] == "request-example"
    assert "request_id" not in call
