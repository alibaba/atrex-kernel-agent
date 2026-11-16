"""Read real composed Light Campaigns; only Agent/GPU responses are offline."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

import test_integrated_campaign as composed
from aka.workflow.light.long_horizon.lease import campaign_writer_lease, writer_active
from observability import (
    export_light_gateway_record, export_light_kernel, light_dashboard,
    light_kernel_best, light_kernel_history, light_lineage_status,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[None, "asynchronous"])
def campaign(request):
    case = composed.ComposedCampaignTest()
    case.setUp()
    search = None if request.param is None else {
        "synchronization": request.param, "epochs": 2, "trajectories": 2,
        "max_parallel_episodes": 2, "episodes_per_trajectory": 1,
    }
    try:
        yield case.run_campaign("fast", ("directions", "experiments"), search, evaluate=True)
    finally:
        case.doCleanups()


def _files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


def test_observe_and_export_current_campaign(campaign, tmp_path):
    before = _files(campaign)
    status = light_lineage_status.snapshot(campaign)
    count = len(list((campaign / "lineages").glob("t*")))
    assert len(status["lineages"]) == count
    for row in status["lineages"]:
        assert row["runtime"]["completed_episodes"] == 2
        assert row["runtime"]["writer_active"] is False
    history = light_kernel_history.snapshot(campaign)
    assert len(history["lineages"]) == count
    results = [result for lineage in history["lineages"] for result in lineage["kernel_results"]]
    assert all(result["performance"] is not None for result in results), results
    assert results and all(result["performance"]["latency_us_geomean"] == pytest.approx(5) for result in results)
    assert any(s["accepted"] for result in results for s in result["submissions"])
    winner = light_kernel_best.snapshot(campaign)["geomean_best"]
    assert winner["latency_us_geomean"] == pytest.approx(5)

    record_id = results[0]["gateway_record_id"]
    evidence = export_light_gateway_record.snapshot(campaign, record_id)
    assert evidence["runtime_audits"]
    assert evidence["agent_response"]["result"]["correct"] is True
    audit_ids = {p.stem.removeprefix("request-")
                 for p in campaign.glob("lineages/t*/runtime-audit/requests/request-*.json")}
    from aka.experiment.measurement_store import MeasurementStore
    store = MeasurementStore(campaign / "measurements")
    for gateway_id in store.record_ids():
        record = store.read(gateway_id)
        assert record["runtime_request_id"] in audit_ids
        exported_record = export_light_gateway_record.snapshot(campaign, gateway_id)
        creation = next(audit for audit in exported_record["runtime_audits"]
                        if audit["request_id"] == record["runtime_request_id"])
        assert creation["agent_request"]
        assert exported_record["agent_response"] == creation["agent_response"]
        assert exported_record["agent_response"]["record_path"] == f"records/{gateway_id}.json"
    export_light_gateway_record.export_gateway_record(campaign, record_id, tmp_path / "gateway")
    exported = export_light_kernel.export_kernel(campaign, winner["kernel_commit_id"], tmp_path / "kernel")
    assert exported["measurement_count"] >= 1
    assert (tmp_path / "kernel/kernel/kernel.py").is_file()

    capture = light_dashboard.write_snapshot(campaign, tmp_path / "dashboard.json")
    dashboard = json.loads(capture.read_text())
    assert len(dashboard["lineages"]["lineages"]) == count
    assert dashboard["kernels"]["lineages"]
    assert dashboard["gateway_records"][record_id]["agent_response"]["result"]["correct"]
    assert _files(campaign) == before, "Observation must not rewrite Campaign evidence"


def test_writer_observation_reads_os_lease_without_creating_state(tmp_path):
    campaign = tmp_path / "not-created"
    assert writer_active(campaign) is False
    assert not list(tmp_path.iterdir())
    with campaign_writer_lease(campaign):
        assert writer_active(campaign) is True
        assert not campaign.exists()
    assert writer_active(campaign) is False


def test_concurrent_catalog_first_readers(tmp_path):
    from aka.workflow.light.research.store import ResearchStore

    root = tmp_path / "catalog/events"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: ResearchStore(root).events(), range(100))) == [[]] * 100


def test_all_observability_cli_entrypoints():
    for script in (ROOT / "observability").glob("*.py"):
        if script.name == "__init__.py":
            continue
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, (script.name, result.stderr)


@pytest.mark.parametrize("rejected", [False, True])
def test_claim_cleanup_preserves_concurrent_request_attachments(tmp_path, rejected):
    from aka.workflow.light.host.journal import SupervisorJournalService

    service = SupervisorJournalService.__new__(SupervisorJournalService)
    service.research_library_root = tmp_path / "research-library"
    requests = tmp_path / ".review-staging/claim-requests"
    service._claim_audits = {}
    for name in ("first", "second"):
        references = requests / name / "references"
        references.mkdir(parents=True)
        (references / "probe.py").write_text(name)
        service._claim_audits[name] = {"reference_snapshot_root": str(references)}
    service._finish_claim_audit(
        "first", status="rejected" if rejected else "accepted",
        error=ValueError("invalid claim") if rejected else None,
    )
    assert not (requests / "first").exists()
    assert (requests / "second/references/probe.py").read_text() == "second"
    if rejected:
        assert (service.research_library_root / "experiments/unbound-attempts/first/references/probe.py").read_text() == "first"
