from pathlib import Path
from types import SimpleNamespace

from observability import (
    common,
    export_light_source,
    light_acceptance,
    light_latency_status,
    light_research_status,
    light_runtime_status,
)
from observability.light_wiki_usage import _journal_adoption
from aka.light.experiment.store import private_write


def test_episode_zero_accepted_evidence_uses_agent_commit(monkeypatch, tmp_path):
    commit = "a" * 40
    record_id = "gateway-" + "b" * 32
    measurement_root = tmp_path / "private" / "measurements"
    measurement_root.mkdir(parents=True)
    (measurement_root / "identity.key").write_bytes(b"x" * 32)

    class Store:
        def __init__(self, root):
            assert root == measurement_root

        def read(self, value):
            assert value == record_id
            return {
                "operation": "evaluate",
                "kernels": {"candidate": {"kernel_id": commit}},
            }

    monkeypatch.setattr(light_latency_status, "supervisor_campaign_root", lambda _: tmp_path / "private")
    monkeypatch.setattr(light_latency_status, "MeasurementStore", Store)
    attempt = {
        "episode": 0,
        "accepted": True,
        "candidate_kernel_commit_id": commit,
        "gateway_record_id": record_id,
    }
    result = light_latency_status._accepted_evidence(
        tmp_path,
        {"episode": 0, "kernel_commit_id": commit},
        attempt,
    )
    assert result["verification_evidence"] == "verified"
    assert result["verification_kernel_commit_id"] == commit


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


def test_acceptance_uses_attempts_and_treats_episode_zero_as_baseline(
    monkeypatch, tmp_path
):
    scope = SimpleNamespace(label="campaign", trajectory=None)
    attempts = [
        {
            "episode": 0,
            "memory_version": 1,
            "accepted": True,
            "status": "candidate_ready",
            "candidate_kernel_commit_id": "d" * 40,
            "latency_us_geomean": 8.0,
        },
        {
            "episode": 1,
            "version": 2,
            "accepted": True,
            "status": "candidate_ready",
            "candidate_kernel_commit_id": "e" * 40,
            "verification": {
                "gate": "PASS",
                "candidate_latency_us": 4.0,
                "incumbent_latency_us": 8.0,
                "improvement_pct": 50.0,
                "gateway_record_id": "gateway-" + "f" * 32,
            },
        },
    ]
    monkeypatch.setattr(light_acceptance, "discover_scopes", lambda _: [scope])
    monkeypatch.setattr(light_acceptance, "attempt_documents", lambda _: attempts)
    result = light_acceptance.snapshot(tmp_path)
    assert result["schema_version"] == 2
    assert result["candidates"][0]["promoted"] is False
    assert result["candidates"][0]["decision_reason"].startswith(
        "framework baseline Episode 0"
    )
    assert result["candidates"][1]["promoted"] is True
    assert result["promotions"][0]["kernel_commit_id"] == "e" * 40


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
    assert result["schema_version"] == 2
    assert scope["campaign_id"] == "campaign-example"
    assert scope["episodes"][0]["episode"] == 0
    assert scope["episodes"][0]["accepted"] is True
