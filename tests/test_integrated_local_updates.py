"""Regression coverage for local Runtime/audit updates across public owners."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

import pytest
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.orchestration import numerical_policy as numerical_review
from aka.workflow.light.orchestration.supervisor_runtime import SupervisorRuntime, agent_runtime_response
from aka.core.service_audit import capture_service_exchanges, audit_service_exchange
from aka.evaluation.tool_results import compact_receipt
from aka.execution.gateway import sdk


def test_full_receipt_preserves_large_results_and_error_details():
    value = {
        "operation": "command", "status": "failed", "gateway_record_id": "gateway-test",
        "result": {
            "stdout": "probe output\n" * 10000,
            "stderr": "compiler trace\n" * 10000 + "actionable compiler error",
            "error": {"http_status": 422, "details": {"validation_errors": [
                {"loc": ["shapes", i], "msg": f"invalid shape {i}"} for i in range(30)
            ]}},
        },
    }
    receipt = compact_receipt(value)
    assert receipt["result"] == value["result"]
    assert receipt["record_path"] == "records/gateway-test.json"
    receipt["result"]["error"]["http_status"] = 500
    assert value["result"]["error"]["http_status"] == 422


@pytest.mark.parametrize("route,body,operation", [
    ("/v1/journal/execute", {"operation": "experiment_record"}, "experiment_record"),
    ("/v1/wiki/query", {"tool": "query_nl"}, "query_nl"),
    ("/v1/gateway/execute", {"argv": ["--kind", "env"]}, "env"),
    ("/v1/gateway/execute", {"argv": ["--kind=evaluate", "--baseline-commit", "a" * 40]}, "same_allocation_abba"),
])
def test_runtime_non_gateway_receipt_has_same_envelope(route, body, operation):
    details = {"id": "example", "full_result": list(range(100))}
    response, code = agent_runtime_response(route, body, {
        "exit_code": 0, "stdout": json.dumps(details), "stderr": "",
    })
    assert code == 0
    assert response == {"operation": operation, "status": "completed", "result": details}


def test_sdk_env_audit_preserves_provider_response_and_rejection(monkeypatch):
    from atrex_gateway_client.errors import GatewayError
    from aka.execution.gateway.sandbox import GatewayHTTPError
    api = Mock()
    raw = {"gpu": "L20N", "ncu_counters": [f"counter_{i}" for i in range(2000)]}
    api.get_env.return_value = raw
    monkeypatch.setattr(sdk, "client", lambda **_: api)
    rows = []
    with capture_service_exchanges(lambda **row: rows.append(row)):
        assert sdk.request("endpoint", "GET", "/v1/env/L20N?force=true", None, 30) == raw
    assert rows[0]["request"] == {"method": "get_env", "gpu": "L20N", "force": True}
    assert rows[1]["response"] == raw
    assert rows[0]["exchange_id"] == rows[1]["exchange_id"]

    failure = {"reason": "request_validation_failed", "details": {"gpu": "unsupported"}}
    api.get_env.side_effect = GatewayError(422, failure)
    with capture_service_exchanges(lambda **row: rows.append(row)):
        with pytest.raises(GatewayHTTPError):
            sdk.request("endpoint", "GET", "/v1/env/L20N", None, 30)
    assert rows[-1]["response"]["http_status"] == 422
    assert rows[-1]["response"]["payload"] == failure
    assert api.get_env.call_count == 2


def test_environment_snapshot_retains_provider_fields_without_mutability():
    from aka.execution.environment_query import _snapshot
    from aka.contracts.execution_environment import EnvironmentQuery
    detail = {"gpu": "L20N", "ncu_counters": [f"counter_{i}" for i in range(2000)]}
    capabilities = {"custom_feature": {"architectures": ["sm_89"]}}
    snapshot = _snapshot("gateway", "https://example.test", EnvironmentQuery("L20N"),
                         detail, capabilities, [], {"run", "env"})
    assert snapshot.as_dict()["raw"] == {"environment": detail, "capabilities": capabilities}
    detail["ncu_counters"].clear()
    assert len(snapshot.raw["environment"]["ncu_counters"]) == 2000


def test_concurrent_service_audit_keeps_request_identity_and_redacts_secrets(tmp_path):
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.root = tmp_path / "runtime"
    runtime.audit_root = tmp_path / "campaign"
    runtime.environment = {"AGATE_AK": "private-access-key", "AGATE_SK": "private-access-secret"}
    runtime.config = SimpleNamespace(private_reference_dir=None, url="https://private-endpoint")
    runtime.lock = threading.RLock()
    barrier = threading.Barrier(4)

    def worker(index):
        request_id = f"request-{index % 2}"
        with capture_service_exchanges(partial(runtime._record_service_exchange, request_id)):
            audit_service_exchange(service="agate", exchange_id=str(index), request={
                "index": index, "credential": "private-access-secret",
            })
            barrier.wait(timeout=10)
            audit_service_exchange(service="agate", exchange_id=str(index), response={"index": index})
        # Context must not leak into later unrelated work on the same thread.
        audit_service_exchange(service="must-not-persist", response={"index": index})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    for request in range(2):
        path = runtime._request_audit_path(f"request-{request}")
        value = json.loads(path.read_text())
        rows = value["service_exchanges"]
        assert len(rows) == 2
        assert {row["request"]["index"] for row in rows} == {request, request + 2}
        assert all(row["request"]["index"] == row["response"]["index"] for row in rows)
        assert "private-access-secret" not in path.read_text()


@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
def test_claim_review_respects_sandbox_and_archives_service_exchange(monkeypatch, tmp_path, sandbox):
    from test_integrated_campaign import REVIEW
    evidence = tmp_path / "evidence"
    (evidence / "submissions").mkdir(parents=True)
    manifest = {"claim_type": "experiment", "gateway_record_count": 1}
    target = {"name": "A measured candidate", "interpretation": "The receipt contains the output."}
    (evidence / "evidence-manifest.json").write_text(json.dumps(manifest))
    (evidence / "submissions/target.json").write_text(json.dumps(target))
    (tmp_path / "kernel.py").write_text("def custom_kernel(x): return x\n")
    campaign = Campaign(name="operator", kernel_demo=str(tmp_path / "kernel.py"),
                        platform="cuda", framework="cuda", work_dir=str(tmp_path.parent))
    rows = []
    campaign._supervisor_runtime = SimpleNamespace(root=tmp_path / "runtime",
        audit_service_exchange=lambda **row: rows.append(row))
    monkeypatch.setattr(campaign, "agent_boundary_environment", lambda _: {"ATREX_AGENT_SANDBOX": sandbox})

    def run_review(root, prompt, **kwargs):
        assert kwargs["extra_environment"]["ATREX_AGENT_SANDBOX"] == sandbox
        assert kwargs["timeout"] is None
        (root / "review.json").write_text(json.dumps(REVIEW))
        capture = Path(kwargs["extra_environment"]["ATREX_SESSION_CAPTURE_DIR"]) / "run-test"
        capture.mkdir(parents=True)
        (capture / "conversation.jsonl").write_text('{"type":"session_end"}\n')
        (capture / "token-usage.json").write_text(json.dumps({
            "state": "completed", "finished_at": "2026-11-16T00:00:00+00:00"}))
        return SimpleNamespace(exit_status=0, timed_out=False, session_id="claim-review",
                               stdout_tail="", stderr_tail="")

    monkeypatch.setattr(campaign, "_run_session", run_review, raising=False)
    result = campaign.review_research_claim(evidence, archive_root=tmp_path / "archive")
    assert result["verdict"] == "supported"
    assert (tmp_path / "archive/conversation.jsonl").is_file()
    assert (tmp_path / "archive/token-usage.json").is_file()
    assert len(rows) == 2
    assert rows[0]["request"] == {"target": target, "evidence_manifest": manifest}
    assert rows[1]["response"]["verdict"] == "supported"
    assert rows[0]["attempt"] == rows[1]["attempt"] == 1


def test_wiki_audit_distinguishes_provider_result_from_cache_reuse(tmp_path):
    from aka.workflow.light.host.wiki import WikiService
    from aka.workflow.light.orchestration.supervisor_runtime import Capability
    from aka.resources.wiki.config import WikiConfig
    raw = {"query_id": "query-test", "records": {"spec": {"text": "GPU facts"}}, "notes": []}
    provider = SimpleNamespace(revision=lambda: "wiki-v1", invoke=Mock(return_value=raw))
    service = WikiService(tmp_path / "wiki", WikiConfig(), "wiki-v1", provider)
    exchanges = []
    owner = SimpleNamespace(_live=lambda _: True, audit_process=lambda *args: None,
        audit_service_exchange=lambda **row: exchanges.append(row))
    capability = Capability(workspace=tmp_path / "agent", context={})
    for _ in range(2):
        assert service.invoke(owner, capability, "query", {"text": "GPU facts"},
                              tmp_path, {}, project=lambda value: value) == raw
    provider.invoke.assert_called_once()
    assert exchanges[0]["service"] == "wiki"
    assert exchanges[0]["request"]["payload"] == {"text": "GPU facts"}
    assert exchanges[1]["response"] == raw
    assert exchanges[2]["service"] == "wiki_cache"
    assert exchanges[2]["response"] == {"status": "reused", "query_id": "query-test"}


def test_research_refresh_uses_one_snapshot_when_peer_publishes(monkeypatch, tmp_path):
    from aka.workflow.light.host import journal
    from aka.workflow.light.research.store import ResearchStore
    service = journal.SupervisorJournalService.__new__(journal.SupervisorJournalService)
    service.campaign_root = tmp_path
    service.origin = {"trajectory": 1, "episode": 1}
    service.research = ResearchStore(tmp_path / "events")
    service._research_state = {"episode": 1, "direction_events": [], "experiments": [],
                              "central_research": {"directions": {}, "experiments": {}}}
    old_id, new_id = "direction_" + "1" * 32, "direction_" + "2" * 32
    shared = {"directions": {old_id: {"direction_id": old_id, "status": "proposed", "owner_lineage": "T2"}},
              "experiments_by_id": {}, "research_queue": [old_id]}
    reads = []

    def peer_view(_root):
        from copy import deepcopy
        value = deepcopy(shared)
        reads.append(value)
        # Advance the peer's publication as soon as the first read returns.
        shared["directions"][new_id] = {"direction_id": new_id, "status": "proposed", "owner_lineage": "T2"}
        shared["research_queue"] = [new_id, old_id]
        return value

    monkeypatch.setattr(journal, "_shared_research", peer_view)
    service._sync_research()
    assert len(reads) == 1
    state = service.research.read()
    assert state["shared_queue"] == [old_id]
    assert new_id not in state["directions"]
    service._sync_research()
    assert service.research.read()["shared_queue"] == [new_id, old_id]


def _upstream_source_review(
    *,
    framework_decision="allow",
    framework_category="framework_compliant",
    compute_decision="allow",
    compute_category="self_authored_compute",
):
    evidence = ["candidate/kernel.py:1"]
    rejected = framework_decision == "reject" or compute_decision == "reject"
    return {
        "schema_version": 2,
        "verdict": "reject" if rejected else "allow",
        "checks": [
            {
                "id": "framework_compliance",
                "decision": framework_decision,
                "category": framework_category,
                "reason": "the implementation framework is visible",
                "evidence": evidence,
            },
            {
                "id": "compute_provenance",
                "decision": compute_decision,
                "category": compute_category,
                "reason": "the target computation source is visible",
                "evidence": evidence,
            },
            {
                "id": "dependency_inventory",
                "decision": "allow",
                "category": "inventory_complete",
                "reason": "all dependencies are inventoried",
                "evidence": evidence,
            },
            {
                "id": "external_code_loading",
                "decision": "allow",
                "category": "no_external_code",
                "reason": "no external implementation is loaded",
                "evidence": evidence,
            },
            {
                "id": "solution_manifest",
                "decision": "allow",
                "category": "not_applicable",
                "reason": "no manifest is required",
                "evidence": evidence,
            },
        ],
        "dependencies": [
            {
                "name": "torch",
                "decision": "allow",
                "category": "framework_runtime",
                "reason": "Torch supplies tensor and launch plumbing only",
                "evidence": evidence,
            }
        ],
        "summary": "complete upstream source-policy review",
    }

def test_final_source_reviewer_audits_exact_request_and_response(
    monkeypatch, tmp_path
):
    (tmp_path / "kernel.py").write_text(
        "def custom_kernel(x):\n    return x\n", encoding="utf-8"
    )
    campaign = Campaign(
        name="operator",
        kernel_demo=str(tmp_path / "kernel.py"),
        platform="cuda",
        framework="cuda",
        work_dir=str(tmp_path.parent),
        optimization_mode="performance",
    )
    exchanges = []
    campaign._supervisor_runtime = SimpleNamespace(
        root=tmp_path / "runtime",
        audit_service_exchange=lambda **value: exchanges.append(value),
    )
    review = _upstream_source_review()

    def run_review(root, _prompt, **kwargs):
        (root / "dependency_review.json").write_text(
            json.dumps(review), encoding="utf-8"
        )
        capture = Path(kwargs["extra_environment"]["ATREX_SESSION_CAPTURE_DIR"])
        complete = capture / "run-test"
        complete.mkdir(parents=True, exist_ok=True)
        (complete / "conversation.jsonl").write_text(
            '{"type":"session_end"}\n', encoding="utf-8"
        )
        (complete / "token-usage.json").write_text(
            json.dumps({
                "state": "completed",
                "finished_at": "2026-11-16T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(
            exit_status=0,
            timed_out=False,
            session_id="final-source-review",
            stdout_tail="",
            stderr_tail="",
        )

    monkeypatch.setattr(campaign, "_run_session", run_review, raising=False)
    monkeypatch.setattr(
        campaign, "agent_boundary_environment", lambda _role: {}, raising=False
    )
    campaign._review_production_candidate_once(
        tmp_path,
        "cuda",
        False,
        candidate_digest="a" * 64,
    )

    assert len(exchanges) == 2
    assert {row["service"] for row in exchanges} == {
        "independent_final_source_reviewer"
    }
    assert exchanges[0]["exchange_id"] == exchanges[1]["exchange_id"]
    assert exchanges[0]["request"]["review_request"]["dsl_policy"] == "guided"
    assert exchanges[0]["request"]["candidate_manifest"]["files"][0][
        "path"
    ] == "kernel.py"
    assert exchanges[1]["response"] == review

def test_supplemental_numerical_reviewer_audits_exact_request_and_response(
    tmp_path,
):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("immutable evidence\n", encoding="utf-8")
    files = {"trusted/evidence.txt": evidence}
    digest = numerical_review._digest(files)
    exchanges = []

    def run_review(root, _prompt, **kwargs):
        response = {
            "schema_version": 1,
            "evidence_digest": digest,
            "action": "complete",
            "summary": "No supplemental probe is required.",
            "suite": None,
        }
        (root / "numerical_review.json").write_text(
            json.dumps(response), encoding="utf-8"
        )
        capture = Path(kwargs["extra_environment"]["ATREX_SESSION_CAPTURE_DIR"])
        complete = capture / "run-test"
        complete.mkdir(parents=True, exist_ok=True)
        (complete / "conversation.jsonl").write_text(
            '{"type":"session_end"}\n', encoding="utf-8"
        )
        (complete / "token-usage.json").write_text(
            json.dumps({
                "state": "completed",
                "finished_at": "2026-11-16T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(
            exit_status=0,
            timed_out=False,
            session_id="supplemental-review",
            stdout_tail="",
            stderr_tail="",
        )

    campaign = SimpleNamespace(
        workspace=tmp_path,
        agent_cli="claude",
        _run_session=run_review,
        agent_boundary_environment=lambda _role: {},
        _supervisor_runtime=SimpleNamespace(
            audit_service_exchange=lambda **value: exchanges.append(value),
        ),
    )
    result = numerical_review._request_review(
        campaign,
        tmp_path,
        files,
        digest,
        state_root=tmp_path / "review-state",
        cancel=SimpleNamespace(
            check=lambda: None,
            is_set=lambda: False,
            wait=lambda _seconds: False,
        ),
    )

    assert len(exchanges) == 2
    assert {row["service"] for row in exchanges} == {
        "supplemental_numerical_reviewer"
    }
    assert exchanges[0]["exchange_id"] == exchanges[1]["exchange_id"]
    assert exchanges[0]["request"]["review_request"]["evidence_digest"] == digest
    assert exchanges[1]["response"] == result
