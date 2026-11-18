"""Research feedback preserves useful observations without forcing mechanism proofs."""
from copy import deepcopy
import json

import pytest

from aka.experiment.measurement_store import MeasurementStore
from aka.task.source.formats import Artifact
from aka.task.source.formats.source_tree import default_tree
from aka.workflow.light.host.journal import SupervisorJournalService
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


def service_at(root, modules, reviewer):
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
        claim_reviewer=reviewer,
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
        "analysis": "Begin related candidate trials.",
    }})
    assert "review" not in started
    return started


@pytest.mark.parametrize("modules", [("experiments",), ("directions", "experiments")])
@pytest.mark.parametrize("quality", ["supported", "partially_supported"])
def test_related_trials_archive_together_without_forcing_revise(tmp_path, modules, quality):
    seen = []

    def reviewer(root):
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


def test_partial_direction_keeps_lifecycle_and_does_not_require_more_review(tmp_path):
    seen = []

    def reviewer(root):
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
