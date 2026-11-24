"""Research feedback preserves useful observations without forcing mechanism proofs."""
from copy import deepcopy
import json

import pytest

from aka.experiment.measurement_store import MeasurementStore
from aka.task.source.formats import Artifact
from aka.task.source.formats.source_tree import default_tree
from aka.workflow.light.host.journal import SupervisorJournalService
from aka.workflow.light.host.errors import AgentRequestError
from aka.workflow.light.long_horizon.claim_review import validate_review
from aka.workflow.light.support.paths import initialize_campaign_root


NO_CHECK = "No additional check needed for the next decision."


def review(quality="supported"):
    return {
        "verdict": quality,
        "primary_issue": "The observations justify a scoped follow-up trial.",
        "findings": [
            {"claim": "The measured observations justify trying workload dispatch.",
             "judgment": quality, "decision_critical": True,
             "reason": "The cited trials identify the measured benefit and regression.",
             "evidence": ["submissions/target.json"]},
            {"claim": "The gain is caused by occupancy.",
             "judgment": "partially_supported", "decision_critical": False,
             "reason": "The cause is unknown and does not determine whether to try dispatch.",
             "evidence": []},
        ],
        "scope_safe_interpretation": "Only the measured regimes support another dispatch trial.",
        "next_check": NO_CHECK,
        "confidence": "medium",
    }


def service_at(root, modules, reviewer, *, claim_review_enabled=True):
    policy, _ = default_tree({"kernel.py": b"def kernel(): return None\n"})
    initialize_campaign_root(
        root, name="research-feedback",
        launch_composition={"lock_version": 2, "profile": "test", "variables_digest": "test",
                            "core_version": "test", "core_digest": "test", "tokens": [], "entries": []},
        source={"file_system": "source_tree", "contract": policy},
    )
    lineage = root / "lineages/t0001"
    workspace = lineage / "agent-workspace"
    workspace.mkdir(parents=True)
    return SupervisorJournalService(
        workspace=workspace, campaign_root=lineage,
        evidence_root=lineage / "runtime-audit", research_root=lineage / "research",
        measurement_root=root / "measurements", tool_modules=modules,
        claim_reviewer=reviewer, claim_review_enabled=claim_review_enabled,
    )


def measured_trial(service, number):
    store = MeasurementStore(service.measurement_root)
    source = f"def kernel(): return {number}\n".encode()
    kernel = store.kernel(source, f"{number:040x}")
    with store.reserve({"operation": "dev", "options": {"command": ["python3", "probe.py"]},
                        "inputs": {"kernel.py": Artifact(source).digest},
                        "trial": number}, {"candidate": kernel}, request_id=f"trial-{number}") as task:
        record = task.finish({"exit_code": 0, "stdout": f"trial {number}\n", "stderr": ""},
                             cacheable=False)
    return record["gateway_record_id"]


def start_direction(service):
    proposed = service.execute({"operation": "direction_update", "request": {
        "action": "propose", "name": "Decode parallelism and traffic architecture",
        "change_scope": "major_redesign", "objective": "Improve decode parallelism",
        "applicability": "Short-query workloads", "expected_benefit": "Lower full-workload latency",
        "plan": ["Compare partition families", "Explore workload dispatch"],
    }})
    started = service.execute({"operation": "direction_update", "request": {
        "action": "start", "direction_id": proposed["direction_id"],
    }})
    assert "review" not in started
    return started


@pytest.mark.parametrize("modules", [("experiments",), ("directions", "experiments")])
@pytest.mark.parametrize("quality", ["supported", "partially_supported"])
def test_related_trials_archive_together_without_forcing_revise(tmp_path, modules, quality):
    seen = []

    def reviewer(root, **kwargs):
        target = json.loads((root / "submissions/target.json").read_text())
        seen.append(target)
        if "experiment_id" in target:
            assert len(list((root / "gateway-records").glob("*/record.json"))) == 2
            return review(quality)
        return review()

    service = service_at(tmp_path / "campaign", modules, reviewer)
    started = start_direction(service) if "directions" in modules else None
    records = [measured_trial(service, number) for number in (1, 2)]
    assert not any("experiment_id" in target for target in seen)
    request = {"name": "Partition-family comparison", "intent": "Choose whether to try dispatch",
               "interpretation": "Long shapes improved, short shapes regressed. Cause unknown; try dispatch.",
               "gateway_record_ids": records}
    if started:
        request["direction_id"] = started["direction_id"]
        direction_before = deepcopy(service.research.read()["directions"][started["direction_id"]])
    result = service.execute({"operation": "experiment_record", "request": request})
    receipt = result["review"]
    assert receipt["archive_status"] == "archived"
    assert receipt["claim_review_quality"] == quality
    assert receipt["evidence_eligible"] is (quality == "supported")
    assert receipt["scientific_next_check"] == NO_CHECK
    assert "resubmit a corrected claim" not in receipt["runtime_action"]
    if quality == "partially_supported":
        assert "scope_safe_interpretation" in receipt["runtime_action"]
        assert "only when" in receipt["runtime_action"]
        assert "solely to obtain supported" in receipt["runtime_action"]
    elif started:
        assert receipt["direction_status"] == "in_progress"
        assert "neither a mechanism proof nor an immediate Direction revision" in receipt["runtime_action"]
    projected = json.loads((service.workspace / result["archive_path"]).read_text())
    assert projected["gateway_record_ids"] == records
    assert projected["scientific_next_check"] == NO_CHECK
    assert projected["claim_review_quality"] == quality
    for record_id in records:
        assert MeasurementStore(service.measurement_root).read(record_id)["gateway_record_id"] == record_id
    if started:
        assert service.research.read()["directions"][started["direction_id"]] == direction_before
    # Another trial remains usable without first revising the partial Experiment or Direction.
    next_record = measured_trial(service, 3)
    hint = service.experiment_recording_hint({"gateway_record_id": next_record})
    assert hint["gateway_record_ids"] == [next_record]
    assert "Related candidate trials may continue" in hint["instruction"]
    assert "before launching another candidate" not in hint["instruction"]
    assert sum("experiment_id" in target for target in seen) == 1


def test_no_check_does_not_override_decision_critical_evidence():
    result = review()
    assert validate_review(result)["next_check"] == NO_CHECK
    # The same uncertain mechanism needs support if it is used to rule out a promising path.
    result["findings"][1]["decision_critical"] = True
    with pytest.raises(ValueError, match="every decision-critical finding"):
        validate_review(result)
    result["verdict"] = "partially_supported"
    assert validate_review(result)["verdict"] == "partially_supported"
    result["findings"][0]["judgment"] = "unsupported"
    with pytest.raises(ValueError, match="no decision-critical unsupported"):
        validate_review(result)


@pytest.mark.parametrize("modules", [("directions",), ("directions", "experiments")])
def test_start_analysis_optional_in_schema_validation_and_archived_revision(tmp_path, modules):
    from aka.workflow.light.host.request_feedback import journal_schema
    seen = []
    def reviewer(root, **kwargs):
        seen.append(root)
        return review("partially_supported")
    service = service_at(tmp_path / "campaign", modules, reviewer)
    started = start_direction(service)
    identity = started["direction_id"]
    assert len(seen) == 1
    assert service.research.read()["directions"][identity].get("analysis") is None
    paused = service.execute({"operation": "direction_update", "request": {
        "action": "defer", "direction_id": identity, "analysis": "Prior assessment",
        "reopen_condition": "The next architecture family becomes useful",
    }})
    schema = journal_schema("direction_update", {"action": "start"},
        directions_enabled=True, experiments_enabled="experiments" in modules)
    assert set(schema["required"]) == {"action", "direction_id"}
    assert "analysis" in schema["properties"]
    closure = journal_schema("direction_update", {"action": "complete"},
        directions_enabled=True, experiments_enabled="experiments" in modules)
    assert "analysis" in closure["required"]
    for invalid in (None, "", 42):
        with pytest.raises((ValueError, AgentRequestError)):
            service.execute({"operation": "direction_update", "request": {
                "action": "start", "direction_id": identity, "analysis": invalid,
            }})
    restarted = service.execute({"operation": "direction_update", "request": {
        "action": "start", "direction_id": identity, "analysis": "New assessment",
    }})
    assert "review" not in restarted and len(seen) == 2
    archived = json.loads((service.workspace / restarted["archive_path"]).read_text())
    assert archived["analysis"] == "New assessment"
    assert archived["revision"] == paused["revision"] + 1
    assert archived["status"] == "in_progress"


def test_partial_direction_keeps_lifecycle_and_does_not_require_more_review(tmp_path):
    seen = []

    def reviewer(root, **kwargs):
        seen.append(json.loads((root / "submissions/target.json").read_text()))
        return review("partially_supported")

    service = service_at(tmp_path / "campaign", ("directions",), reviewer)
    started = start_direction(service)
    assert len(seen) == 1
    view = service.research.read()["directions"][started["direction_id"]]
    assert view["status"] == "in_progress"
    assert view["claim_review_quality"] == "partially_supported"
    assert view["scientific_next_check"] == NO_CHECK
    record_id = measured_trial(service, 1)
    assert service.experiment_recording_hint({"gateway_record_id": record_id}) is None


@pytest.mark.parametrize("modules", [("directions",), ("directions", "experiments")])
@pytest.mark.parametrize("action", ["complete", "abandon"])
@pytest.mark.parametrize("quality", ["supported", "partially_supported", "unsupported", "review_unavailable"])
def test_terminal_direction_requires_current_supported_review(tmp_path, modules, action, quality):
    seen = []
    terminal_quality = quality

    def reviewer(root, **kwargs):
        target = json.loads((root / "submissions/target.json").read_text())
        seen.append(target)
        if target["action"] == "propose":
            return review()
        if terminal_quality == "review_unavailable":
            raise RuntimeError("Reviewer transport unavailable")
        return review(terminal_quality)

    service = service_at(tmp_path / "campaign", modules, reviewer)
    started = start_direction(service)
    direction_id = started["direction_id"]
    before = deepcopy(service.research.read()["directions"][direction_id])
    request = {"operation": "direction_update", "request": {
        "action": action, "direction_id": direction_id,
        "analysis": "All worthwhile architecture families have been assessed.",
    }}
    if quality != "supported":
        with pytest.raises(AgentRequestError) as rejected:
            service.execute(request)
        error = rejected.value.response["error"]
        assert error["code"] == (
            "research_claim_unsupported" if quality == "unsupported" else "direction_review_insufficient"
        )
        assert error["review"]["claim_review_quality"] == quality
        assert error["review"]["archive_status"] == "not_archived"
        assert "archive_path" not in error
        if quality != "unsupported":
            assert error["direction_status"] == "in_progress"
            assert error["required_claim_review_quality"] == "supported"
            assert "remains in_progress" in error["review"]["runtime_action"]
        assert service.research.read()["directions"][direction_id] == before
        assert len(list((service.workspace / "research-library/directions" / direction_id).glob("v*"))) == 2
        audits = list((service.measurement_root / "reviews" / direction_id).glob("attempts/*/result.json"))
        assert len(audits) == 1
        audit = json.loads(audits[0].read_text())["review"]
        assert audit["claim_review_quality"] == quality
        assert audit["archive_status"] == "not_archived"
        terminal_quality = "supported"

    result = service.execute(request)
    assert result["review"]["archive_status"] == "archived"
    assert result["review"]["claim_review_quality"] == "supported"
    assert result["revision"] == 3
    view = service.research.read()["directions"][direction_id]
    assert view["status"] == {"complete": "completed", "abandon": "abandoned"}[action]
    assert view["claim_review_quality"] == "supported"
    assert [item["action"] for item in seen] == (
        ["propose", action] if quality == "supported" else ["propose", action, action]
    )


@pytest.mark.parametrize("modules", [("directions",), ("directions", "experiments")])
@pytest.mark.parametrize("action", ["defer", "block"])
def test_partial_pause_and_restart_require_no_experiment(tmp_path, modules, action):
    seen = []

    def reviewer(root, **kwargs):
        seen.append(json.loads((root / "submissions/target.json").read_text())["action"])
        return review("partially_supported")

    service = service_at(tmp_path / "campaign", modules, reviewer)
    started = start_direction(service)
    direction_id = started["direction_id"]
    paused = service.execute({"operation": "direction_update", "request": {
        "action": action, "direction_id": direction_id,
        "analysis": "Other work currently has greater expected value.",
        "reopen_condition": "The next measurement resolves the prioritization uncertainty.",
    }})
    assert paused["review"]["claim_review_quality"] == "partially_supported"
    assert paused["review"]["archive_status"] == "archived"
    assert service.research.read()["directions"][direction_id]["status"] == {
        "defer": "deferred", "block": "blocked"
    }[action]
    restarted = service.execute({"operation": "direction_update", "request": {
        "action": "start", "direction_id": direction_id,
    }})
    assert "review" not in restarted
    assert service.research.read()["directions"][direction_id]["analysis"] == "Other work currently has greater expected value."
    assert seen == ["propose", action]
    view = service.research.read()["directions"][direction_id]
    assert view["status"] == "in_progress"
    assert view["claim_review_quality"] == "partially_supported"


@pytest.mark.parametrize("modules", [("directions",), ("experiments",), ("directions", "experiments")])
def test_disabled_claim_review_archives_supported_without_reviewer(tmp_path, modules):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled Claim Review must not call the reviewer")

    service = service_at(tmp_path / "campaign", modules, forbidden, claim_review_enabled=False)
    results = []
    direction_id = None
    if "directions" in modules:
        direction_id = start_direction(service)["direction_id"]
        results.append(service.execute({"operation": "direction_update", "request": {
            "action": "revise", "direction_id": direction_id, "analysis": "Updated assessment",
            "name": "Decode parallelism and traffic architecture", "change_scope": "major_redesign",
            "objective": "Improve decode parallelism", "applicability": "Short-query workloads",
            "expected_benefit": "Lower full-workload latency",
            "plan": ["Compare partition families", "Explore workload dispatch"],
        }}))
    if "experiments" in modules:
        evidence = service.workspace / "scratch/probe.txt"
        evidence.parent.mkdir()
        evidence.write_text("frozen evidence")
        request = {"name": "Probe", "intent": "Choose the next trial", "interpretation": "Try a new tile",
                   "references": ["scratch/probe.txt"]}
        if direction_id:
            request["direction_id"] = direction_id
        result = service.execute({"operation": "experiment_record", "request": request})
        results.append(result)
        results.append(service.execute({"operation": "experiment_record", "request": {
            **request, "experiment_id": result["experiment_id"], "interpretation": "Try another tile",
        }}))
        with pytest.raises((ValueError, AgentRequestError)):
            service.execute({"operation": "experiment_record", "request": {
                **request, "references": ["scratch/missing.txt"],
            }})
    if direction_id:
        results.append(service.execute({"operation": "direction_update", "request": {
            "action": "complete", "direction_id": direction_id, "analysis": "The program is exhausted.",
        }}))
    for result in results:
        receipt = result["review"]
        assert receipt["claim_review_quality"] == "review_not_requested"
        assert receipt["evidence_eligible"] is False
        assert receipt["review_skipped"] is True
        assert "skipped" in receipt["runtime_action"]
        projected = json.loads((service.workspace / result["archive_path"]).read_text())
        assert projected["review_skipped"] is True
        assert projected["claim_review_quality"] == "review_not_requested"
        assert "scope_safe_interpretation" not in projected
        assert receipt["archive_status"] == "archived"
        assert "/v0/" not in result["archive_path"]
    archives = list(service.measurement_root.rglob("result.json"))
    assert any(json.loads(path.read_text()).get("review", {}).get("review_skipped") for path in archives)


def test_claim_review_option_default_and_validation():
    from aka.workflow.light.provider import WorkflowOptions
    from aka.workflow.light.plugin import validate_config
    assert WorkflowOptions().claim_review is False
    assert WorkflowOptions(claim_review=True).claim_review is True
    assert WorkflowOptions(claim_review=False).claim_review is False
    validate_config({"tool_modules": ["directions", "experiments"], "claim_review": False})
    with pytest.raises(ValueError, match="claim_review must be boolean"):
        WorkflowOptions(claim_review="false")
