from __future__ import annotations

import hashlib
import json
import subprocess
from io import BytesIO
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from aka.core.errors import CompositionError
from aka.contracts.agent import AgentRuntimeCapabilities, NativeInvocation, TokenUsage
from aka.contracts.content import SourceContent, WorkspaceFile
from aka.contracts.problem import InitialSource
from aka.contracts.workflow import CampaignRequest, ProblemSource, WorkflowSettings
from aka.contracts.workspace import CandidateHandle
from aka.light.agent.home import SESSION_HOME_ROOT_ENV, prepare_agent_environment
from aka.light.agent.runtime.model import AgentRunRequest as RuntimeAgentRunRequest
from aka.light.agent.runtime.runtime import build_agent_runtime
from aka.light.agent.sandbox import sandbox_executable
from aka.light.experiment.frontier import (
    kernel_full_evaluation_evidence,
    kernel_pareto_frontier,
)
from aka.light.experiment import records as experiment_records
from aka.light.experiment.store import DuplicateTask, MeasurementStore
from aka.light.research import ResearchStore
from aka.light.runtime.server import _Handler, _journal_evidence_hint
from aka.light.runtime.paths import (
    campaign_runtime_configuration,
    initialize_campaign_root,
    legacy_supervisor_campaign_root,
    record_campaign_v0,
    supervisor_campaign_base_root,
    supervisor_campaign_root,
    supervisor_measurement_root,
    supervisor_research_library_root,
    supervisor_submission_root,
)
from aka.light.workflow.submissions import (
    accepted_submission_kernel_ids,
    append_submission,
    migrate_legacy_journals,
    submissions,
)
from aka.light.workflow.store import CampaignStore
from aka.light.runtime.request_errors import AgentRequestError
from aka.light.runtime.request_feedback import augment_error, journal_schema
from aka.light.runtime.server import RuntimeConfig, SupervisorRuntime
from aka.light.workflow.journal_service import (
    SupervisorJournalService,
    _progress_markdown,
    _shared_research,
    initialize_journal,
    load_journal,
    private_write,
    publish_catalog_shared_research,
    project_campaign_research_library,
    research_projection,
    shared_research_path,
    update_direction,
)
from aka.light.workflow.journal import JournalFactory
from aka.light.workflow.models import SupervisorState
from aka.light.workflow.bootstrap import Campaign
from aka.light.workflow.campaign import ManagedCampaign
from aka.light.workflow.claim_review import canonical_review_quality, validate_review
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workflow.search import (
    COORDINATOR_STATE,
    PooledSearch,
    SearchOptions,
    lineage_state_path,
    merge_lineage_state,
    publication_import_path,
    publication_path,
    publication_source_root,
    read_state,
)
from aka.light.workflow.provider import WorkflowOptions
from aka.light.workflow.search_driver import (
    EpisodeSearchDriver,
    _aggregate_direction_queue,
)
from aka.light.workflow.session import LongSessionRunner
from aka.light.workspace.agent_git import (
    AgentGit,
    DirtyKernelWorktree,
)
from aka.light.workspace.auxiliary import AuxiliaryWorkspace
from aka.light.workspace.episode import EpisodeWorkspace
from aka.light.workspace.files import publish_if_changed
from aka.light.workspace.provider import ManagedCandidateWorkspace
from aka.light.workspace.shared_history import (
    export_publication,
    import_exports_into_repository,
)
from aka.light.workspace.sources import default_tree, from_git, install, pack, snapshot
from aka.light.workspace.sources.private_state import (
    bind as bind_source_state,
    read as read_source_state,
)


def _claim_review(verdict: str, findings: list[dict[str, object]]) -> dict[str, object]:
    return {
        "verdict": verdict,
        "primary_issue": "The decision is usable within the measured scope.",
        "findings": findings,
        "scope_safe_interpretation": (
            "Use the measured decision without the secondary mechanism claim."
        ),
        "next_check": "Measure the secondary mechanism only if it affects a later decision.",
        "confidence": "medium",
    }


def _claim_finding(judgment: str, *, critical: bool) -> dict[str, object]:
    return {
        "claim": "A bounded decision or a secondary explanation.",
        "judgment": judgment,
        "decision_critical": critical,
        "reason": "The supplied evidence has the stated level of support.",
        "evidence": ["gateway-records/gateway-example/record.json"],
    }


def _test_launch_composition(**overrides) -> dict[str, object]:
    value = {
        "lock_version": 2,
        "profile": "light",
        "variables_digest": "variables",
        "core_version": "test",
        "core_digest": "core",
        "tokens": [],
        "entries": [],
    }
    value.update(overrides)
    return value


def _test_source_binding(
    source: bytes = b"def run():\n    return None\n",
) -> dict[str, object]:
    contract, _ = default_tree({"kernel.py": source})
    return {"file_system": "source_tree", "contract": contract}


def test_campaign_manifest_atomically_owns_launch_composition(tmp_path):
    composition = _test_launch_composition(profile="light-sandbox")
    root = initialize_campaign_root(
        tmp_path / "campaign",
        name="operator-a",
        launch_composition=composition,
        source=_test_source_binding(),
    )

    assert json.loads((root / "campaign.json").read_text()) == {
        "schema_version": 3,
        "layout": "atrex.light.campaign.v3",
        "name": "operator-a",
        "launch_composition": composition,
        "source": _test_source_binding(),
        "v0": None,
        "runtime_configuration": None,
    }
    assert (root / "campaign.lock").is_file()
    assert not (root / ".atrex_plugins").exists()
    assert initialize_campaign_root(
        root,
        name="operator-a",
        launch_composition=composition,
        source=_test_source_binding(),
    ) == root

    with pytest.raises(CompositionError, match="core_digest: changed"):
        initialize_campaign_root(
            root,
            name="operator-a",
            launch_composition=_test_launch_composition(
                profile="light-sandbox", core_digest="different"
            ),
            source=_test_source_binding(),
        )
    with pytest.raises(ValueError, match="source contract differs"):
        initialize_campaign_root(
            root,
            name="operator-a",
            launch_composition=composition,
            source=_test_source_binding(b"def run():\n    return 3\n"),
        )


def test_campaign_manifest_owns_one_source_binding_for_every_lineage(tmp_path):
    binding = _test_source_binding()
    root = initialize_campaign_root(
        tmp_path / "campaign",
        name="operator-a",
        launch_composition=_test_launch_composition(),
        source=binding,
    )
    first = root / "lineages" / "t0001" / "agent-workspace"
    second = root / "lineages" / "t0002" / "agent-workspace"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    for workspace in (first, second):
        bind_source_state(
            workspace,
            file_system=binding["file_system"],
            contract=binding["contract"],
        )
        assert read_source_state(workspace) == (
            binding["file_system"],
            binding["contract"],
        )

    assert not (root / "source").exists()
    assert not (root / "lineages" / "t0001" / "source").exists()
    assert not (root / "lineages" / "t0002" / "source").exists()

    different = _test_source_binding(b"def run():\n    return 2\n")
    with pytest.raises(ValueError, match="source contract cannot change"):
        bind_source_state(
            first,
            file_system=different["file_system"],
            contract=different["contract"],
        )


def test_v2_campaign_migrates_matching_detached_source_into_manifest(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    composition = _test_launch_composition()
    binding = _test_source_binding()
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "layout": "atrex.light.campaign.v2",
                "name": "operator-a",
                "launch_composition": composition,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detached = root / "lineages" / "t0001" / "source"
    detached.mkdir(parents=True)
    (detached / "file-system.json").write_text(
        json.dumps({"schema_version": 1, "name": binding["file_system"]}) + "\n"
    )
    (detached / "source-contract.json").write_text(
        json.dumps({"schema_version": 1, "contract": binding["contract"]}) + "\n"
    )

    initialize_campaign_root(
        root,
        name="operator-a",
        launch_composition=composition,
        source=binding,
    )

    manifest = json.loads((root / "campaign.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["source"] == binding
    assert not detached.exists()


def test_managed_campaign_freezes_source_before_writing_manifest(monkeypatch, tmp_path):
    composition = _test_launch_composition()
    monkeypatch.setattr(
        "aka.bootstrap.continuation.read",
        lambda _environment: json.dumps({"identity": composition}),
    )
    request = CampaignRequest(
        problem=ProblemSource(tmp_path / "problem"),
        workspace=tmp_path / "campaign",
        name="vector-add",
        settings=WorkflowSettings(
            platform="L20N",
            framework="cuda",
            optimization_mode="ultra",
        ),
    )
    problem = SimpleNamespace(
        initial_source=InitialSource(
            kind="native",
            entry=tmp_path / "input.py",
            files=(),
        ),
        evaluation=SimpleNamespace(
            runtime_root=tmp_path / "atrex-bench",
            private_reference_dir=None,
        ),
    )

    class SourceProvider:
        def prepare(self, _request):
            assert not (request.workspace / "campaign.json").exists()
            return SourceContent(
                (WorkspaceFile("kernel.py", text="def run():\n    return None\n"),)
            )

    workflow = SimpleNamespace(
        candidate=ManagedCandidateWorkspace(),
        source=SourceProvider(),
        options=WorkflowOptions(search=None),
        agent=SimpleNamespace(
            options=SimpleNamespace(
                sandbox="none", bwrap_executable="bwrap", read_only_paths=()
            )
        ),
        problem=object(),
        tool_runtime=object(),
    )
    evaluation = SimpleNamespace(
        hardware="L20N",
        profile="",
        url="",
        timeout=1200,
        ssh="",
        ssh_init="",
        ssh_gpu=None,
        health_command="",
        verify_repeats=4,
        evaluation_step_timeout=300,
        shape_batch_size=10,
        gpu_clock_mhz=None,
        min_improvement_pct=0.0,
    )

    campaign = ManagedCampaign.create(request, problem, workflow, evaluation)

    manifest = json.loads((request.workspace / "campaign.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["source"]["file_system"] == "source_tree"
    assert manifest["source"]["contract"]["editable_roots"] == [
        "kernel.py",
        "impl",
    ]
    assert manifest["v0"] is None
    assert (request.workspace / "campaign.lock").is_file()
    assert not (request.workspace / "initial-source").exists()
    assert not (request.workspace / "source").exists()
    assert campaign.workspace == (
        request.workspace / "lineages" / "t0001" / "agent-workspace"
    )


def test_campaign_v0_identity_points_to_the_measurement_store_archive(tmp_path):
    root = initialize_campaign_root(
        tmp_path / "campaign",
        name="operator-a",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(),
    )
    workspace = root / "lineages" / "t0001" / "agent-workspace"
    workspace.mkdir(parents=True)
    commit = "a" * 40
    source = pack(
        {"kernel.py": b"def run():\n    return None\n"},
        _test_source_binding()["contract"],
    )
    identity = MeasurementStore(supervisor_measurement_root(workspace)).kernel(
        source, commit
    )

    record_campaign_v0(
        workspace,
        kernel_commit_id=commit,
        kernel_artifact_digest=identity["kernel_artifact_digest"],
    )
    # Recording the same immutable identity is idempotent.
    record_campaign_v0(
        workspace,
        kernel_commit_id=commit,
        kernel_artifact_digest=identity["kernel_artifact_digest"],
    )

    manifest = json.loads((root / "campaign.json").read_text())
    assert manifest["v0"] == {
        "kernel_commit_id": commit,
        "kernel_artifact_digest": identity["kernel_artifact_digest"],
    }
    assert (
        root / "measurements" / "kernels" / commit / "kernel.py"
    ).read_bytes() == source
    assert not (root / "initial-source").exists()

    other = MeasurementStore(supervisor_measurement_root(workspace)).kernel(
        source, "b" * 40
    )
    with pytest.raises(ValueError, match="V0 identity cannot change"):
        record_campaign_v0(
            workspace,
            kernel_commit_id="b" * 40,
            kernel_artifact_digest=other["kernel_artifact_digest"],
        )


def test_detached_runtime_configuration_is_read_only_compatibility(tmp_path):
    workspace = tmp_path / "legacy-workspace"
    workspace.mkdir()
    legacy_root = legacy_supervisor_campaign_root(workspace)
    legacy_root.mkdir(parents=True)
    detached = legacy_root / "runtime-config.json"
    detached.write_text(
        json.dumps({"hardware": "L20D", "optimization_mode": "ultra"}) + "\n",
        encoding="utf-8",
    )
    before = detached.read_bytes()

    assert campaign_runtime_configuration(workspace) == {
        "hardware": "L20D",
        "optimization_mode": "ultra",
    }
    assert detached.read_bytes() == before


def test_campaign_manifest_without_runtime_snapshot_is_accepted(tmp_path):
    root = tmp_path / "older-campaign"
    root.mkdir()
    composition = _test_launch_composition()
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "layout": "atrex.light.campaign.v2",
                "name": "operator-a",
                "launch_composition": composition,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert initialize_campaign_root(
        root,
        name="operator-a",
        launch_composition=composition,
        source=_test_source_binding(),
    ) == root
    assert campaign_runtime_configuration(root) is None
    migrated = json.loads((root / "campaign.json").read_text())
    assert migrated["schema_version"] == 3
    assert migrated["source"] == _test_source_binding()


def test_new_campaign_rejects_detached_composition_directory(tmp_path):
    root = tmp_path / "campaign"
    (root / ".atrex_plugins").mkdir(parents=True)

    with pytest.raises(ValueError, match="must be empty"):
        initialize_campaign_root(
            root,
            name="operator-a",
            launch_composition=_test_launch_composition(),
            source=_test_source_binding(),
        )


def test_existing_campaign_rejects_detached_composition_authority(tmp_path):
    root = initialize_campaign_root(
        tmp_path / "campaign",
        name="operator-a",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(),
    )
    detached = root / ".atrex_plugins" / "composition.json"
    detached.parent.mkdir()
    detached.write_text("{}\n")

    with pytest.raises(ValueError, match="must live only in campaign.json"):
        initialize_campaign_root(
            root,
            name="operator-a",
            launch_composition=_test_launch_composition(),
            source=_test_source_binding(),
        )


def test_legacy_review_quality_names_are_projected_with_unambiguous_meaning():
    assert canonical_review_quality("not_reviewed") == "review_not_requested"
    assert canonical_review_quality("unreviewed") == "review_unavailable"
    assert canonical_review_quality("supported") == "supported"


def test_completed_exact_gateway_task_reuses_normal_record_response(
    monkeypatch, tmp_path
):
    record_id = "gateway-" + "a" * 32
    cached = {
        "exit_code": 0,
        "stdout": "completed\n",
        "stderr": "",
        "agent_result": {
            "operation": "dev",
            "status": "completed",
            "gateway_record_id": record_id,
            "result": {"exit_code": 0},
        },
    }

    class Store:
        identity_key = b"test-identity"

        @staticmethod
        def reserve(*_args, **_kwargs):
            raise DuplicateTask(record_id)

        @staticmethod
        def read(requested_record_id):
            assert requested_record_id == record_id
            return {"response": cached}

    evaluation = SimpleNamespace(
        prepare=lambda *_args, **_kwargs: SimpleNamespace(
            request={"operation": "dev"},
            inputs={},
            repetitions=1,
            runtime_pins={},
        )
    )
    effects = SimpleNamespace(
        mode="fast",
        repetitions=1,
        kernel_ids={},
        deadline=None,
        request_id="",
    )
    monkeypatch.setattr(
        "aka.light.workspace.sources.measurement_artifacts",
        lambda _request, _inputs: {},
    )

    result = experiment_records.execute(
        Store(),
        evaluation,
        effects,
        tmp_path,
        SimpleNamespace(),
        [],
        {},
        [],
    )

    assert result == {**cached, "reused": True}
    assert "duplicate_gateway_task" not in json.dumps(result)
    assert "previous_gateway_record_id" not in json.dumps(result)


def test_v0_is_recorded_once_in_campaign_measurements_without_baseline_store(
    monkeypatch, tmp_path
):
    campaign_root = initialize_campaign_root(
        tmp_path / "campaign",
        name="v0-measurement",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(),
    )
    workspace = campaign_root / "lineages" / "t0001" / "agent-workspace"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("class Model:\n    pass\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "V0 source"], cwd=workspace, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "all_pass": True,
        "latency_us_geomean": 1.0,
        "latency_us_arith_mean": 1.1,
        "latency_us_by_shape": {"0": 1.0},
        "performance_objective": "latency_us_geomean",
    }
    calls = []

    def execute_v0(*_args, **kwargs):
        calls.append(kwargs)
        state = kwargs["job_root"] / "jobs" / "ev_v0" / "state.json"
        state.parent.mkdir(parents=True)
        state.write_text(
            json.dumps({"process": {"stdout": json.dumps({"job_id": "ev_v0"})}})
        )
        return subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            0,
            "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "",
        )

    monkeypatch.setattr("aka.light.workflow.bootstrap._sandbox_command", execute_v0)
    campaign = SimpleNamespace(
        workspace=workspace,
        sandbox_hardware="L20N",
        sandbox_profile="",
        sandbox_url="",
        sandbox_timeout=900,
        sandbox_ssh="",
        sandbox_ssh_init="",
        sandbox_health_command="",
        private_reference_dir=None,
        evaluation_step_timeout=300,
        gpu_clock_mhz=None,
        _print_v0_evaluator_output=lambda _process: None,
        _generalized_memory_coverage_problem=lambda _memory: None,
    )

    measured = Campaign._run_v0_evaluator(campaign, commit)
    Campaign._finalize_v0_measurement(campaign, measured, commit)
    assert measured == result
    assert len(calls) == 1
    assert calls[0]["terminal_validation"] is True
    assert calls[0]["job_root"].is_relative_to(
        campaign_root / "measurements" / "records"
    )
    assert not (campaign_root / "baseline").exists()
    assert not (workspace / "memory" / "v0.json").exists()
    assert not (workspace / ".atrex_versions" / "v0.json").exists()

    from aka.light.workspace.baseline import read_baseline_record

    summary = read_baseline_record(workspace)
    assert summary["git_commit_hash"] == commit
    assert summary["gateway_record_id"].startswith("gateway-")
    assert summary["backend_job_ids"] == ["ev_v0"]
    record = MeasurementStore(supervisor_measurement_root(workspace)).read(
        summary["gateway_record_id"]
    )
    assert record["request"]["visibility"] == "supervisor_private_v0"
    assert record["response"]["agent_result"]["result"]["correct"] is True
    archived = record["kernels"]["candidate"]
    manifest = json.loads((campaign_root / "campaign.json").read_text())
    assert manifest["v0"] == {
        "kernel_commit_id": commit,
        "kernel_artifact_digest": archived["kernel_artifact_digest"],
    }
    assert not (campaign_root / "initial-source").exists()

    # Restart/recovery reuses the same canonical Record and submits no GPU job.
    measured_again = Campaign._run_v0_evaluator(campaign, commit)
    assert measured_again["all_pass"] is True
    assert len(calls) == 1


def test_rejected_v0_remains_a_failed_measurement_not_a_baseline_file(
    monkeypatch, tmp_path
):
    campaign_root = initialize_campaign_root(
        tmp_path / "campaign",
        name="rejected-v0",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(),
    )
    workspace = campaign_root / "lineages" / "t0001" / "agent-workspace"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("class Model:\n    pass\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "V0 source"], cwd=workspace, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    failed = {"all_pass": False, "failures": ["shape 0 mismatch"]}
    monkeypatch.setattr(
        "aka.light.workflow.bootstrap._sandbox_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["python", "test_kernel.py"],
            1,
            "[test_kernel] RESULT_JSON=" + json.dumps(failed) + "\n",
            "",
        ),
    )
    campaign = SimpleNamespace(
        workspace=workspace,
        sandbox_hardware="L20N",
        sandbox_profile="",
        sandbox_url="",
        sandbox_timeout=900,
        sandbox_ssh="",
        sandbox_ssh_init="",
        sandbox_health_command="",
        private_reference_dir=None,
        evaluation_step_timeout=300,
        gpu_clock_mhz=None,
        _print_v0_evaluator_output=lambda _process: None,
    )

    with pytest.raises(RuntimeError, match="V0 Evaluate failed"):
        Campaign._run_v0_evaluator(campaign, commit)

    from aka.light.workspace.baseline import read_baseline_record

    store = MeasurementStore(supervisor_measurement_root(workspace))
    record_ids = store.record_ids()
    assert len(record_ids) == 1
    record = store.read(record_ids[0])
    # A deterministic correctness rejection is terminal reusable evidence; it
    # is still excluded from the passing V0 reference lookup below.
    assert record["cacheable"] is True
    assert record["response"]["agent_result"]["result"]["correct"] is False
    assert read_baseline_record(workspace) is None
    assert not (campaign_root / "baseline").exists()


def test_campaign_research_catalog_sync_watermark_and_async_head(tmp_path):
    catalog_root = tmp_path / "research-catalog"
    catalog = ResearchStore(catalog_root / "events")
    direction_id = "direction_" + "a" * 32

    def observe(revision: int, objective: str):
        return catalog.observe_direction(
            {
                "direction_id": direction_id,
                "revision": revision,
                "name": "persistent redesign",
                "change_scope": "major_redesign",
                "objective": objective,
                "applicability": "all measured regimes",
                "expected_benefit": "material full-workload improvement",
                "plan": ["implement", "measure"],
                "stop_conditions": [],
                "status": "in_progress",
                "owner_lineage": "T1",
                "access": "local_writable",
            },
            origin={"epoch": 1, "trajectory": 1, "episode": 1},
        )

    first = observe(1, "revision one")
    watermark = first["sequence"]
    observe(2, "revision two")

    synchronous = tmp_path / "synchronous"
    synchronous.mkdir()
    project_campaign_research_library(
        catalog_root=catalog_root,
        workspace=synchronous,
        trajectory=2,
        synchronization="synchronous",
        peer_through_sequence=watermark,
    )
    sync_latest = json.loads(
        (
            synchronous / "research-library" / "directions" / direction_id
            / "v1" / "direction.json"
        ).read_text()
    )
    assert sync_latest["revision"] == 1
    assert sync_latest["owner_lineage"] == "T1"
    assert sync_latest["access"] == "peer_read_only"
    assert not (synchronous / "research-library" / "manifest.json").exists()
    assert not (
        synchronous / "research-library" / "directions" / direction_id / "latest.json"
    ).exists()

    observe(3, "revision three")
    project_campaign_research_library(
        catalog_root=catalog_root,
        workspace=synchronous,
        trajectory=2,
        synchronization="synchronous",
        peer_through_sequence=watermark,
    )
    assert json.loads(
        (
            synchronous / "research-library" / "directions" / direction_id
            / "v1" / "direction.json"
        ).read_text()
    )["revision"] == 1
    assert not (
        synchronous / "research-library" / "directions" / direction_id / "v2"
    ).exists()
    canonical_latest = json.loads(
        (
            tmp_path / "research-library" / "directions" / direction_id
            / "v3" / "direction.json"
        ).read_text()
    )
    assert canonical_latest["revision"] == 3
    assert canonical_latest["owner_lineage"] == "T1"

    # Agent copies are disposable.  A native (non-bwrap) Agent can alter one,
    # but the next Runtime projection repairs it from Campaign authority.
    projected = (
        synchronous / "research-library" / "directions" / direction_id
        / "v1" / "direction.json"
    )
    projected.write_text('{"tampered": true}\n')
    project_campaign_research_library(
        catalog_root=catalog_root,
        workspace=synchronous,
        trajectory=2,
        synchronization="synchronous",
        peer_through_sequence=watermark,
    )
    assert json.loads(projected.read_text())["revision"] == 1

    asynchronous = tmp_path / "asynchronous"
    asynchronous.mkdir()
    project_campaign_research_library(
        catalog_root=catalog_root,
        workspace=asynchronous,
        trajectory=2,
        synchronization="asynchronous",
    )
    async_latest = json.loads(
        (
            asynchronous / "research-library" / "directions" / direction_id
            / "v3" / "direction.json"
        ).read_text()
    )
    assert async_latest["revision"] == 3
    assert async_latest["owner_lineage"] == "T1"


def test_claim_review_verdict_ignores_noncritical_limitations():
    result = validate_review(_claim_review("supported", [
        _claim_finding("supported", critical=True),
        _claim_finding("partially_supported", critical=False),
        _claim_finding("unsupported", critical=False),
    ]))
    assert result["verdict"] == "supported"


def test_claim_review_verdict_tracks_decision_critical_findings():
    with pytest.raises(ValueError, match="decision-critical finding"):
        validate_review(_claim_review("supported", [
            _claim_finding("partially_supported", critical=True),
        ]))

    assert validate_review(_claim_review("partially_supported", [
        _claim_finding("partially_supported", critical=True),
        _claim_finding("unsupported", critical=False),
    ]))["verdict"] == "partially_supported"

    with pytest.raises(ValueError, match="no decision-critical unsupported"):
        validate_review(_claim_review("partially_supported", [
            _claim_finding("partially_supported", critical=True),
            _claim_finding("unsupported", critical=True),
        ]))


def test_claim_review_requires_a_decision_critical_finding():
    with pytest.raises(ValueError, match="requires a decision-critical finding"):
        validate_review(_claim_review("supported", [
            _claim_finding("supported", critical=False),
        ]))


def _measurement(
    store: MeasurementStore,
    source: bytes,
    marker: str,
    *,
    kernel_id: str | None = None,
) -> tuple[str, str]:
    kernel_id = kernel_id or hashlib.sha1((marker + "\0").encode() + source).hexdigest()
    identity = store.kernel(source, kernel_id)
    request = {
        "operation": "evaluate",
        "options": {},
        "inputs": {"kernel.py": hashlib.sha256(source).hexdigest()},
        "kernel_ids": {"candidate": kernel_id},
        "marker": marker,
    }
    response = {"exit_code": 0, "stdout": "", "stderr": ""}
    with store.reserve(request, {"candidate": identity}) as task:
        record = task.finish(response, cacheable=True)
    return record["gateway_record_id"], identity["kernel_id"]


def _passing_full_measurement(
    store: MeasurementStore,
    source: bytes,
    marker: str,
    latencies: dict[str, float],
    *,
    kernel_commit_id: str | None = None,
) -> tuple[str, str]:
    kernel_commit_id = kernel_commit_id or hashlib.sha1(
        (marker + "\0").encode() + source
    ).hexdigest()
    identity = store.kernel(source, kernel_commit_id)
    request = {
        "operation": "evaluate",
        "options": {
            "command": ["python3", "test_kernel.py"],
            "evaluation_mode": "full",
            "evaluation_policy": {"mode": "full"},
        },
        "inputs": {"kernel.py": hashlib.sha256(source).hexdigest()},
        "kernel_ids": {"candidate": kernel_commit_id},
        "marker": marker,
    }
    sample = {
        "all_pass": True,
        "correct": True,
        "mode": "full",
        "latency_us_by_shape": latencies,
    }
    response = {
        "exit_code": 0,
        "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(sample) + "\n",
        "stderr": "",
    }
    with store.reserve(request, {"candidate": identity}) as task:
        record = task.finish(response, cacheable=True)
    return record["gateway_record_id"], identity["kernel_id"]


def test_kernel_full_evaluation_evidence_matches_exact_commit(tmp_path):
    store = MeasurementStore(tmp_path / "measurements")
    source = b"def kernel():\n    return 1\n"
    passing_record, passing_commit = _passing_full_measurement(
        store,
        source,
        "passing",
        {"shape-0": 4.0, "shape-1": 9.0},
        kernel_commit_id="1" * 40,
    )

    assert kernel_full_evaluation_evidence(store, passing_commit) == {
        "status": "passed",
        "gateway_record_id": passing_record,
        "latency_us_geomean": 6.0,
        "measured_shape_count": 2,
    }
    assert kernel_full_evaluation_evidence(store, "2" * 40) == {
        "status": "missing"
    }


def test_measurement_kernel_archive_has_no_supervisor_git_mapping(tmp_path):
    store = MeasurementStore(tmp_path / "measurements")
    kernel_commit_id = "a" * 40

    identity = store.kernel(b"def kernel():\n    return 1\n", kernel_commit_id)

    assert identity["kernel_id"] == kernel_commit_id
    kernel_directory = store.root / "kernels" / kernel_commit_id
    assert (kernel_directory / "kernel.py").is_file()
    assert not (kernel_directory / "supervisor-git.json").exists()


def test_kernel_full_evaluation_evidence_reports_failed_result(tmp_path):
    store = MeasurementStore(tmp_path / "measurements")
    source = b"def kernel():\n    return 1\n"
    kernel_commit_id = "3" * 40
    identity = store.kernel(source, kernel_commit_id)
    request = {
        "operation": "evaluate",
        "options": {
            "command": ["python3", "test_kernel.py"],
            "evaluation_mode": "full",
            "evaluation_policy": {"mode": "full"},
        },
        "inputs": {"kernel.py": hashlib.sha256(source).hexdigest()},
        "kernel_ids": {"candidate": kernel_commit_id},
    }
    sample = {
        "all_pass": False,
        "correct": False,
        "mode": "full",
        "error": "shape-1 correctness failed",
    }
    response = {
        "exit_code": 1,
        "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(sample) + "\n",
        "stderr": "",
    }
    with store.reserve(request, {"candidate": identity}) as task:
        record = task.finish(response, cacheable=True)

    assert kernel_full_evaluation_evidence(store, kernel_commit_id) == {
        "status": "failed",
        "gateway_record_id": record["gateway_record_id"],
        "reason_code": "correctness_failed",
        "reason": "shape-1 correctness failed",
    }


def test_kernel_submission_rejects_incomplete_performance_fields(tmp_path):
    store = MeasurementStore(tmp_path / "measurements")
    source = b"def kernel():\n    return 1\n"
    kernel_commit_id = "4" * 40
    identity = store.kernel(source, kernel_commit_id)
    request = {
        "operation": "evaluate",
        "options": {
            "command": ["python3", "test_kernel.py"],
            "evaluation_mode": "full",
            "evaluation_policy": {"mode": "full"},
        },
        "inputs": {"kernel.py": hashlib.sha256(source).hexdigest()},
        "kernel_ids": {"candidate": kernel_commit_id},
    }
    sample = {"all_pass": True, "correct": True, "mode": "full"}
    response = {
        "exit_code": 0,
        "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(sample) + "\n",
        "stderr": "",
    }
    with store.reserve(request, {"candidate": identity}) as task:
        record = task.finish(response, cacheable=True)

    assert kernel_full_evaluation_evidence(store, kernel_commit_id) == {
        "status": "failed",
        "gateway_record_id": record["gateway_record_id"],
        "reason_code": "performance_result_incomplete",
        "reason": (
            "full Evaluate correctness passed, but latency_us_by_shape is missing, "
            "empty, or contains non-finite/non-positive values"
        ),
    }
    assert kernel_pareto_frontier(store)["best_full_evaluation"] is None


def test_unresolved_direction_closure_may_omit_supporting_experiments(tmp_path):
    journals = tmp_path / "journals"
    journal = journals / "episodes" / "e00000001" / "journal.json"
    initialize_journal(
        journal,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions", "experiments"),
    )
    direction_id = update_direction(
        journal,
        journals,
        {
            "action": "propose",
            "name": "Capability probe",
            "change_scope": "major_redesign",
            "objective": "Determine whether the target instruction family unlocks a faster architecture.",
            "applicability": "The configured target architecture.",
            "expected_benefit": "Unlock a higher-throughput implementation path.",
            "plan": ["Compile one bounded capability probe."],
        },
    )["direction_id"]
    update_direction(
        journal,
        journals,
        {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Run the bounded probe.",
        },
    )
    update_direction(
        journal,
        journals,
        {
            "action": "block",
            "direction_id": direction_id,
            "analysis": "The external compiler service is unavailable.",
            "reopen_condition": "Resume after the compiler service recovers.",
        },
    )
    closure = load_journal(journal)["direction_events"][-1]
    assert closure["supporting_experiment_ids"] == []
    schema = journal_schema(
        "direction_update",
        {"action": "block"},
        directions_enabled=True,
        experiments_enabled=True,
    )
    assert "supporting_experiment_ids" in schema["properties"]
    assert "supporting_experiment_ids" not in schema["required"]


def _supported_claim_review(evidence):
    return {
        "verdict": "supported",
        "primary_issue": "No unsupported claim remains.",
        "findings": [{
            "claim": "The submitted bounded claim.",
            "judgment": "supported",
            "decision_critical": True,
            "reason": "The cited Gateway Record is present in the evidence bundle.",
            "evidence": list(evidence),
        }],
        "scope_safe_interpretation": "Use the claim only within its stated scope.",
        "next_check": "Continue with the next bounded candidate.",
        "confidence": "high",
    }


def test_direction_only_review_receives_direct_gateway_records(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )
    record_id, _ = _passing_full_measurement(
        MeasurementStore(tmp_path / "scope" / "measurements"),
        b"def kernel():\n    return 1\n",
        "direction-only-review",
        {"shape_a": 1.0},
    )

    def reviewer(root):
        manifest = json.loads((root / "evidence-manifest.json").read_text())
        assert manifest["gateway_record_count"] == 1
        assert manifest["missing_gateway_records"] == []
        record_path = root / "gateway-records" / record_id / "record.json"
        assert record_path.is_file()
        record = json.loads(record_path.read_text())
        assert record["gateway_record_id"] == record_id
        return _supported_claim_review(
            [f"gateway-records/{record_id}/record.json"]
        )

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions",),
        claim_reviewer=reviewer,
    )
    result = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Vectorized load",
            "change_scope": "incremental",
            "objective": "Reduce instruction overhead with aligned wider loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Reduce latency on aligned shapes.",
            "plan": ["Compare scalar and vectorized candidates."],
            "gateway_record_ids": [record_id],
        },
    })
    revised_draft = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": result["direction_id"],
            "name": "Aligned vectorized load",
            "change_scope": "incremental",
            "objective": "Reduce instruction overhead with aligned wider loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Reduce latency on aligned shapes.",
            "plan": ["Compare scalar and aligned vectorized candidates."],
        },
    })
    started = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": result["direction_id"],
            "analysis": "Run the bounded aligned-load comparison.",
        },
    })
    revised_active = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": result["direction_id"],
            "name": "Aligned vectorized load",
            "change_scope": "incremental",
            "objective": "Reduce instruction overhead with aligned wider loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Reduce latency on aligned shapes.",
            "plan": ["Compare scalar and aligned vectorized candidates."],
        },
    })

    assert "review" not in result
    assert "review" not in revised_draft
    assert started["review"]["claim_review_quality"] == "supported"
    assert revised_active["review"]["claim_review_quality"] == "supported"
    direction = json.loads(
        (workspace / revised_active["archive_path"]).read_text()
    )
    assert direction["gateway_record_ids"] == [record_id]
    schema = journal_schema(
        "direction_update",
        {"action": "propose"},
        directions_enabled=True,
        experiments_enabled=False,
    )
    assert "gateway_record_ids" in schema["properties"]


def test_derived_direction_review_resolves_experiment_gateway_records(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )
    record_id, _ = _passing_full_measurement(
        MeasurementStore(tmp_path / "scope" / "measurements"),
        b"def kernel():\n    return 2\n",
        "derived-direction-review",
        {"shape_a": 0.8},
    )
    reviewed_derived = []

    def reviewer(root):
        target = json.loads((root / "submissions" / "target.json").read_text())
        proposal = json.loads(
            (root / "submissions" / "direction-proposal.json").read_text()
        )
        if proposal.get("derived_from_experiment_ids"):
            manifest = json.loads((root / "evidence-manifest.json").read_text())
            assert manifest["gateway_record_count"] == 1
            assert manifest["missing_gateway_records"] == []
            assert (root / "gateway-records" / record_id / "record.json").is_file()
            derived_experiment = proposal["derived_from_experiment_ids"][0]
            assert (
                root / "submissions" / "supporting-experiments"
                / f"{derived_experiment}.json"
            ).is_file()
            reviewed_derived.append((target["direction_id"], target["action"]))
        return _supported_claim_review(["submissions/target.json"])

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions", "experiments"),
        claim_reviewer=reviewer,
    )
    source = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Scalar baseline",
            "change_scope": "incremental",
            "objective": "Remove launch-bound overhead from the scalar implementation.",
            "applicability": "Small vector-add shapes.",
            "expected_benefit": "Identify the next scheduling change.",
            "plan": ["Measure the scalar implementation."],
        },
    })
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": source["direction_id"],
            "analysis": "Measure before deriving the next path.",
        },
    })
    experiment = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Scalar measurement",
            "intent": "Measure the launch-bound baseline.",
            "gateway_record_ids": [record_id],
            "interpretation": "The measured result motivates a vectorized alternative.",
        },
    })
    derived = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Vectorized alternative",
            "change_scope": "incremental",
            "objective": "Reduce the measured overhead with vectorized loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Improve the measured launch-bound regime.",
            "plan": ["Measure one vectorized candidate."],
            "relationship": "refinement",
            "derived_from_direction_ids": [source["direction_id"]],
            "derived_from_experiment_ids": [experiment["experiment_id"]],
        },
    })

    revised_draft = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": derived["direction_id"],
            "name": "Vectorized alternative",
            "change_scope": "incremental",
            "objective": "Reduce the measured overhead with aligned vectorized loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Improve the measured launch-bound regime.",
            "plan": ["Measure one aligned vectorized candidate."],
        },
    })
    started = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": derived["direction_id"],
            "analysis": "Run the bounded vectorized-load comparison.",
        },
    })
    revised_active = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": derived["direction_id"],
            "name": "Vectorized alternative",
            "change_scope": "incremental",
            "objective": "Reduce the measured overhead with aligned vectorized loads.",
            "applicability": "Aligned vector-add shapes.",
            "expected_benefit": "Improve the measured launch-bound regime.",
            "plan": ["Measure one aligned vectorized candidate."],
        },
    })

    assert reviewed_derived == [
        (derived["direction_id"], "start"),
        (derived["direction_id"], "revise"),
    ]
    assert "review" not in derived
    assert "review" not in revised_draft
    assert started["review"]["claim_review_quality"] == "supported"
    assert revised_active["review"]["claim_review_quality"] == "supported"


def test_research_records_are_projected_as_normal_workspace_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions", "experiments"),
    )

    proposed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split KV",
            "change_scope": "major_redesign",
            "objective": "Expose sufficient KV parallelism for long-context regimes.",
            "applicability": "Long-context attention shapes on the target GPU.",
            "expected_benefit": "Reduce full-workload geometric-mean latency.",
            "plan": ["Compile a probe", "Evaluate the complete workload"],
        },
    })
    direction_id = proposed["direction_id"]
    assert proposed["archive_path"] == (
        f"research-library/directions/{direction_id}/v0/direction.json"
    )
    started = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Highest expected value.",
        },
    })
    assert started["archive_path"] == (
        f"research-library/directions/{direction_id}/v1/direction.json"
    )
    (workspace / "scratch" / "probe.md").parent.mkdir(parents=True)
    (workspace / "scratch" / "probe.md").write_text("probe evidence\n", encoding="utf-8")
    (workspace / "scratch" / "analysis" / "empty").mkdir(parents=True)
    (workspace / "scratch" / "analysis" / "result.json").write_text(
        '{"winner":"ksplit2"}\n', encoding="utf-8"
    )

    store = MeasurementStore(tmp_path / "scope" / "measurements")
    first_source = b"def kernel():\n    return 1\n"
    (workspace / "kernel.py").write_bytes(first_source)
    agent_git = AgentGit(workspace, tmp_path / "agent.bundle")
    agent_git.prepare()
    record_id, kernel_id = _measurement(
        store, first_source, "probe", kernel_id=agent_git.head()
    )
    second_source = b"def kernel():\n    return 2\n"
    (workspace / "kernel.py").write_bytes(second_source)
    second_commit = agent_git.checkpoint("second measured Kernel")
    pareto_record_id, pareto_kernel_commit_id = _passing_full_measurement(
        store,
        second_source,
        "full",
        {"shape_a": 12.5, "shape_b": 8.25},
        kernel_commit_id=second_commit,
    )
    append_submission(tmp_path / "scope" / "submissions", {
        "submission_id": "submission-" + "1" * 32,
        "lineage": "T1",
        "episode": 1,
        "accepted": True,
        "kernel_commit_id": pareto_kernel_commit_id,
    })
    unsubmitted_source = b"def kernel():\n    return 3\n"
    (workspace / "kernel.py").write_bytes(unsubmitted_source)
    unsubmitted_commit = agent_git.checkpoint("faster but unsubmitted Kernel")
    unsubmitted_record_id, _ = _passing_full_measurement(
        store,
        unsubmitted_source,
        "unsubmitted-full",
        {"shape_a": 1.0, "shape_b": 1.0},
        kernel_commit_id=unsubmitted_commit,
    )
    recorded = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "KSPLIT probe",
            "intent": "Check whether the mechanism compiles and runs.",
            "gateway_record_ids": [record_id],
            "interpretation": "The mechanism is executable; full performance remains open.",
            "references": ["scratch/probe.md", "scratch/analysis"],
        },
    })
    experiment_id = recorded["experiment_id"]
    assert recorded["archive_path"] == (
        f"research-library/experiments/{experiment_id}/v1/experiment.json"
    )
    stored_experiment = ResearchStore(tmp_path / "research").read()["experiments"][
        experiment_id
    ]
    assert set(stored_experiment) == {
        "experiment_id", "sequence", "name", "intent", "gateway_record_ids",
        "interpretation", "direction_id", "references", "revision",
        "reference_revision", "owner_lineage", "access", "origin",
    }

    direction = json.loads(
        (workspace / started["archive_path"]).read_text()
    )
    experiment = json.loads(
        (workspace / recorded["archive_path"]).read_text()
    )
    gateway = json.loads(
        (workspace / "records" / f"{record_id}.json").read_text()
    )
    assert direction["status"] == "in_progress"
    assert direction["owner_lineage"] == "T1"
    assert direction["access"] == "local_writable"
    assert direction["applicability"] == (
        "Long-context attention shapes on the target GPU."
    )
    assert direction["expected_benefit"] == (
        "Reduce full-workload geometric-mean latency."
    )
    assert "completed_plan_steps" not in direction
    assert "next_step" not in direction
    assert "associated_experiment_ids" not in direction
    assert experiment["direction_id"] == direction_id
    assert experiment["owner_lineage"] == "T1"
    assert experiment["access"] == "local_writable"
    assert set(experiment) == {
        "experiment_id", "name", "intent", "gateway_record_ids", "interpretation",
        "direction_id", "references", "revision", "owner_lineage", "access",
    }
    archived_probe = (
        workspace / "research-library" / "experiments" / experiment_id
        / "v1" / "references" / "scratch" / "probe.md"
    )
    archived_result = (
        workspace / "research-library" / "experiments" / experiment_id / "v1" / "references"
        / "scratch" / "analysis" / "result.json"
    )
    archived_empty = (
        workspace / "research-library" / "experiments" / experiment_id / "v1" / "references"
        / "scratch" / "analysis" / "empty"
    )
    assert experiment["references"] == [
        f"research-library/experiments/{experiment_id}/v1/references/scratch/probe.md",
        f"research-library/experiments/{experiment_id}/v1/references/scratch/analysis",
    ]
    assert archived_probe.read_text() == "probe evidence\n"
    assert json.loads(archived_result.read_text()) == {"winner": "ksplit2"}
    assert archived_empty.is_dir()
    assert (
        service.research_library_root / "experiments"
        / experiment_id / "v1" / "references" / "scratch" / "probe.md"
    ).read_text() == "probe evidence\n"
    assert (
        service.research_library_root / "experiments"
        / experiment_id / "v1" / ".references.json"
    ).is_file()
    assert not (
        workspace / "research-library" / "experiments" / experiment_id
        / "v1" / ".references.json"
    ).exists()
    assert not (service.research_catalog_root / "archives").exists()
    assert gateway["gateway_record_id"] == record_id
    assert gateway["kernel_commit_id"] == kernel_id
    assert "kernel_id" not in gateway
    assert "request" not in gateway and "inputs" not in gateway
    assert not (workspace / "research-library" / "experiments" / experiment_id / "records").exists()

    service.publish_research_files()
    progress = (workspace / "PROGRESS.md").read_text()
    assert "proposed by `T1`" in progress
    assert "Applicability: Long-context attention shapes on the target GPU." in progress


    assert "Expected benefit: Reduce full-workload geometric-mean latency." in progress
    assert "atrex.research_board.v2" not in progress
    assert '"schema_version"' not in progress
    assert '"revision"' not in progress
    assert '"recent_experiments"' not in progress
    assert "# Current state" in progress
    assert "## Handoff readiness" not in progress
    assert "## Lineages" in progress
    assert "## Directions" in progress
    assert "### Local work queue" in progress
    assert "local rank #1" in progress
    assert "### Shared direction pool\n\n_None._" in progress
    assert "### Terminal directions" in progress
    assert "### Closed" not in progress
    assert direction_id in progress

    assert "**Best measured passing Kernel commit ID:**" not in progress
    assert "**Global best accepted submission" not in progress
    assert f"**Best accepted submission for `T1`:** `{pareto_kernel_commit_id}`" in progress
    assert "promoted incumbent" not in progress
    assert "**Promoted incumbent commit ID:**" not in progress
    assert "**Last measured working Kernel commit ID:**" in progress
    assert "**Latest visible Gateway result and scope:** evaluate (completed; PASS; GeoMean" in progress
    assert "scope: full, contract, 2 measured Shapes" in progress
    assert "**Suggested next work/probe:**" in progress
    assert "**Current real blocker:** none recorded." in progress
    assert pareto_kernel_commit_id in progress
    assert pareto_record_id in progress
    assert (
        "Eligible full evaluations for accepted local and imported peer submissions: "
        "**1**; covered Shapes: **2**."
    ) in progress
    assert (workspace / "records" / f"{pareto_record_id}.json").is_file()
    assert (workspace / "records" / f"{unsubmitted_record_id}.json").is_file()
    assert "| `shape_a` | 12.5 |" in progress
    assert "| `shape_b` | 8.25 |" in progress
    frontier = json.loads((workspace / "frontier.json").read_text())
    assert frontier["winner_kernel_commit_ids"] == [pareto_kernel_commit_id]
    assert frontier["best_full_evaluation"]["kernel_commit_id"] == pareto_kernel_commit_id

    # Recovery rebuilds deleted convenience files from authoritative stores.
    (workspace / started["archive_path"]).unlink()
    (workspace / "records" / f"{record_id}.json").unlink()
    (workspace / "scratch" / "probe.md").write_text("mutated later\n", encoding="utf-8")
    archived_probe.unlink()
    service.publish_research_files()
    assert (workspace / started["archive_path"]).is_file()
    assert (workspace / "records" / f"{record_id}.json").is_file()
    assert archived_probe.read_text() == "probe evidence\n"


def test_local_experiment_can_be_revised_without_losing_audit_or_references(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("experiments",),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("experiments",),
    )
    (workspace / "scratch").mkdir()
    (workspace / "scratch" / "note.txt").write_text("revision one\n")
    created = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Split-K probe",
            "intent": "Test whether the decomposition is viable.",
            "interpretation": "The mechanism compiles; performance is unresolved.",
            "references": ["scratch/note.txt"],
        },
    })
    experiment_id = created["experiment_id"]
    assert created["status"] == "recorded"
    assert created["revision"] == 1

    revised = service.execute({
        "operation": "experiment_record",
        "request": {
            "experiment_id": experiment_id,
            "name": "Split-K full measurement",
            "intent": "Determine whether the decomposition improves the workload.",
            "interpretation": "The measured implementation is useful only for long-K shapes.",
        },
    })
    assert revised == {
        "status": "updated",
        "experiment_id": experiment_id,
        "revision": 2,
        "archive_path": (
            f"research-library/experiments/{experiment_id}/v2/experiment.json"
        ),
    }
    research = ResearchStore(tmp_path / "research")
    stored = research.read()["experiments"][experiment_id]
    assert stored["revision"] == 2
    assert stored["sequence"] == 1
    assert stored["references"] == [
        f"research-library/experiments/{experiment_id}/v1/references/scratch/note.txt"
    ]
    assert stored["reference_revision"] == 1
    projected = json.loads(
        (workspace / revised["archive_path"]).read_text()
    )
    assert projected["revision"] == 2
    assert projected["interpretation"].startswith("The measured implementation")
    assert "reference_revision" not in projected
    assert (
        workspace / "research-library" / "experiments" / experiment_id
        / "v2" / "references" / "scratch" / "note.txt"
    ).read_text() == "revision one\n"
    assert not (evidence / "claim-audit").exists()

    (workspace / "scratch" / "note.txt").write_text("revision three\n")
    replaced = service.execute({
        "operation": "experiment_record",
        "request": {
            "experiment_id": experiment_id,
            "name": "Split-K full measurement",
            "intent": "Determine whether the decomposition improves the workload.",
            "interpretation": "The updated evidence confirms the scoped result.",
            "references": ["scratch/note.txt"],
        },
    })
    assert replaced["revision"] == 3
    assert research.read()["experiments"][experiment_id]["reference_revision"] == 3
    assert (
        workspace / "research-library" / "experiments" / experiment_id
        / "v3" / "references" / "scratch" / "note.txt"
    ).read_text() == "revision three\n"

    next_evidence = journals / "episodes" / "e00000002"
    next_evidence.mkdir()
    initialize_journal(
        next_evidence / "journal.json",
        episode=2,
        base_commit="b" * 40,
        branch="work-2",
        tool_modules=("experiments",),
    )
    next_service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=next_evidence,
        research_root=tmp_path / "research",
    )
    cross_episode = next_service.execute({
        "operation": "experiment_record",
        "request": {
            "experiment_id": experiment_id,
            "name": "Split-K full measurement",
            "intent": "Determine whether the decomposition improves the workload.",
            "interpretation": "A later Episode narrows the result to long-K shapes.",
        },
    })
    assert cross_episode["revision"] == 4
    assert research.read()["experiments"][experiment_id]["revision"] == 4
    assert json.loads(
        (workspace / cross_episode["archive_path"]).read_text()
    )["revision"] == 4


def test_open_local_direction_can_be_revised_with_stable_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions",),
    )
    proposed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split K",
            "change_scope": "major_redesign",
            "objective": "Improve long-context shapes through K-parallel architecture families.",
            "applicability": "Long-context shapes.",
            "expected_benefit": "Reduce long-context latency.",
            "plan": ["Compile the bounded probe", "Evaluate the workload"],
        },
    })
    direction_id = proposed["direction_id"]
    assert proposed["revision"] == 0
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Highest expected benefit.",
        },
    })
    revised = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": direction_id,
            "name": "Split K crossover",
            "change_scope": "major_redesign",
            "objective": "Determine and optimize the K-parallel crossover across workload regimes.",
            "applicability": "Shapes beyond the measured K crossover.",
            "expected_benefit": "Reduce long-context latency without short-shape regression.",
            "plan": [
                "Measure the crossover",
                "Compile the bounded probe",
                "Evaluate the workload",
            ],
            "stop_conditions": ["Reduction overhead dominates beyond the crossover"],
        },
    })
    assert revised == {
        "status": "updated",
        "direction_id": direction_id,
        "revision": 2,
        "archive_path": f"research-library/directions/{direction_id}/v2/direction.json",
    }
    events = [
        event
        for event in ResearchStore(tmp_path / "research").events()
        if event["type"] == "direction.observed"
    ]
    assert [event["payload"]["direction"]["revision"] for event in events] == [0, 1, 2]
    projected = json.loads(
        (workspace / revised["archive_path"]).read_text()
    )
    assert projected["revision"] == 2
    assert projected["objective"].endswith("workload regimes.")
    assert "hypothesis" not in projected
    assert "hypothesis_status" not in projected
    assert projected["status"] == "in_progress"
    assert "completed_plan_steps" not in projected
    assert "next_step" not in projected
    assert not (evidence / "claim-audit").exists()

    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "defer",
            "direction_id": direction_id,
            "analysis": "Pause until the profiler is restored.",
            "reopen_condition": "Profiler is available.",
        },
    })
    with pytest.raises((AgentRequestError, ValueError)) as error:
        service.execute({
            "operation": "direction_update",
            "request": {
                "action": "revise",
                "direction_id": direction_id,
                "name": "Split K crossover",
                "change_scope": "major_redesign",
                "objective": "A revised but closed optimization objective.",
                "applicability": "Long-context shapes.",
                "expected_benefit": "Reduce latency.",
                "plan": ["Measure"],
            },
        })
    assert "start it before revising" in str(error.value)


def test_claim_revisions_resume_the_prior_review_session(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )
    review_calls = []

    def reviewer(root, *, resume_session_id=None):
        target = json.loads((root / "submissions" / "target.json").read_text())
        session_id = resume_session_id or f"review-session-{len(review_calls) + 1}"
        review_calls.append((target["action"] if "action" in target else "experiment",
                             resume_session_id, session_id))
        return {
            "verdict": "supported",
            "primary_issue": "No unsupported decision-critical claim.",
            "findings": [{
                "claim": "The revised claim is bounded.",
                "judgment": "supported",
                "decision_critical": True,
                "reason": "The supplied evidence is sufficient for this scope.",
                "evidence": ["submissions/target.json"],
            }],
            "scope_safe_interpretation": "Use only within the stated scope.",
            "next_check": "Run the planned bounded measurement.",
            "confidence": "high",
            "_review_session_audit": {
                "schema": "atrex.research-claim-review-session.v1",
                "attempts": [{
                    "attempt": 1,
                    "session_id": session_id,
                    "status": "accepted",
                    "exit_status": 0,
                    "timed_out": False,
                    "resumed": bool(resume_session_id),
                    "conversation_path": f"claim-review-sessions/{session_id}/conversation.jsonl",
                }],
            },
        }

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        claim_reviewer=reviewer,
    )
    proposed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split K",
            "change_scope": "major_redesign",
            "objective": "Improve long-context shapes through split-K architecture families.",
            "applicability": "Long-context shapes.",
            "expected_benefit": "Reduce latency.",
            "plan": ["Compile a bounded probe"],
        },
    })
    direction_id = proposed["direction_id"]
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": direction_id,
            "name": "Split K crossover",
            "change_scope": "major_redesign",
            "objective": "Determine the useful split-K crossover and implementation family.",
            "applicability": "Shapes above the crossover.",
            "expected_benefit": "Reduce long-context latency.",
            "plan": ["Measure the crossover"],
        },
    })
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Proceed with the bounded probe.",
        },
    })
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": direction_id,
            "name": "Split K crossover",
            "change_scope": "major_redesign",
            "objective": "Determine the measured split-K crossover and implementation family.",
            "applicability": "Shapes above the crossover.",
            "expected_benefit": "Reduce long-context latency.",
            "plan": ["Measure the crossover under the same allocation policy"],
        },
    })
    experiment = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Crossover note",
            "intent": "Record the bounded interpretation.",
            "interpretation": "The crossover remains unresolved.",
        },
    })
    service.execute({
        "operation": "experiment_record",
        "request": {
            "experiment_id": experiment["experiment_id"],
            "name": "Crossover note",
            "intent": "Record the corrected bounded interpretation.",
            "interpretation": "The evidence supports only long-context shapes.",
        },
    })

    assert review_calls[0] == ("start", None, "review-session-1")
    assert review_calls[1] == ("revise", "review-session-1", "review-session-1")
    assert review_calls[2][0:2] == ("experiment", None)
    assert review_calls[3] == (
        "experiment", review_calls[2][2], review_calls[2][2]
    )


def test_missing_claim_revision_ids_return_repairable_guidance(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
    )

    with pytest.raises(AgentRequestError) as missing_direction:
        service.execute({
            "operation": "direction_update",
            "request": {
                "action": "revise",
                "direction_id": "direction_" + "1" * 32,
                "name": "Unknown Direction",
                "change_scope": "local",
                "objective": "Resolve a nonexistent optimization objective.",
                "applicability": "No visible scope.",
                "expected_benefit": "No established benefit.",
                "plan": ["Inspect the current projection"],
            },
        })
    direction_error = missing_direction.value.response["error"]
    assert direction_error["code"] == "direction_not_found"
    assert "research-library/directions/" in direction_error["next_action"]
    assert "action=propose" in direction_error["next_action"]

    with pytest.raises(AgentRequestError) as missing_experiment:
        service.execute({
            "operation": "experiment_record",
            "request": {
                "experiment_id": "experiment_" + "2" * 32,
                "name": "Unknown Experiment",
                "intent": "Revise a record that does not exist.",
                "interpretation": "No claim can be revised under this ID.",
            },
        })
    experiment_error = missing_experiment.value.response["error"]
    assert experiment_error["code"] == "experiment_not_found"
    assert "research-library/experiments/" in experiment_error["next_action"]
    assert "omit experiment_id" in experiment_error["next_action"]


def test_research_mutation_receipts_include_archive_and_inline_qwen_review(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )
    reviewed_targets = []

    def reviewer(root):
        target = json.loads((root / "submissions" / "target.json").read_text())
        reviewed_targets.append(target)
        review_number = len(reviewed_targets)
        return {
            "verdict": "supported",
            "primary_issue": "No material unsupported claim was found.",
            "findings": [{
                "claim": "The submitted research statement.",
                "judgment": "supported",
                "decision_critical": True,
                "reason": "The bounded submission is internally consistent.",
                "evidence": ["submissions/target.json"],
            }],
            "scope_safe_interpretation": "Use the claim only within its stated scope.",
            "next_check": "Measure the next candidate against the same baseline.",
            "confidence": "high",
            "_review_session_audit": {
                "schema": "atrex.research-claim-review-session.v1",
                "attempts": [{
                    "attempt": 1,
                    "session_id": f"review-session-{review_number}",
                    "status": "accepted",
                    "exit_status": 0,
                    "timed_out": False,
                    "conversation_path": (
                        f"claim-review-sessions/review-{review_number}/"
                        "run-1/conversation.jsonl"
                    ),
                }],
            },
        }

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        claim_reviewer=reviewer,
    )
    proposed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split KV",
            "change_scope": "major_redesign",
            "objective": "Expose sufficient KV parallelism for long-context workloads.",
            "applicability": "Long-context attention shapes.",
            "expected_benefit": "Reduce end-to-end latency.",
            "plan": ["Compile a bounded probe"],
        },
    })
    direction_id = proposed["direction_id"]
    assert proposed["archive_path"] == (
        f"research-library/directions/{direction_id}/v0/direction.json"
    )
    assert "review" not in proposed
    projected_draft = json.loads(
        (workspace / proposed["archive_path"]).read_text()
    )
    assert projected_draft["claim_review_quality"] == "review_not_requested"
    assert "review_path" not in proposed
    assert not list((workspace / "research-library").rglob("*.review.json"))

    revised_draft = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "revise",
            "direction_id": direction_id,
            "name": "Split KV architecture family",
            "change_scope": "major_redesign",
            "objective": "Explore a bounded split-KV architecture family.",
            "applicability": "Long-context attention shapes.",
            "expected_benefit": "Reduce end-to-end latency.",
            "plan": ["Compile a bounded probe", "Compare viable split factors"],
        },
    })
    assert revised_draft["archive_path"] == proposed["archive_path"]
    assert revised_draft["revision"] == 0
    assert "review" not in revised_draft
    projected_draft = json.loads(
        (workspace / revised_draft["archive_path"]).read_text()
    )
    assert projected_draft["name"] == "Split KV architecture family"
    assert projected_draft["claim_review_quality"] == "review_not_requested"

    started = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "This is the highest-value unresolved path.",
        },
    })
    assert started["archive_path"] == (
        f"research-library/directions/{direction_id}/v1/direction.json"
    )
    assert started["review"]["archive_status"] == "archived"
    assert started["review"]["evidence_eligible"] is True
    assert started["review"]["claim_review_quality"] == "supported"
    assert "evidence_manifest" not in started["review"]
    assert "_review_session_audit" not in started["review"]
    assert "session_audit" not in started["review"]
    reviewed_direction = json.loads(
        (workspace / started["archive_path"]).read_text()
    )
    assert reviewed_direction["status"] == "in_progress"
    assert reviewed_direction["claim_review_quality"] == "supported"
    assert projected_draft["status"] == "proposed"
    assert projected_draft["claim_review_quality"] == "review_not_requested"
    recorded = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Design note",
            "intent": "Specify the first bounded probe.",
            "interpretation": "No measurement claim is made yet.",
        },
    })
    experiment_id = recorded["experiment_id"]
    assert recorded["archive_path"] == (
        f"research-library/experiments/{experiment_id}/v1/experiment.json"
    )
    assert "review_path" not in recorded
    assert "evidence_manifest" not in recorded["review"]
    assert "reviewer_model" not in recorded["review"]
    assert "_review_session_audit" not in recorded["review"]
    assert "session_audit" not in recorded["review"]
    assert recorded["review"]["direction_status"] == "in_progress"
    assert recorded["review"]["runtime_action"] == (
        "Keep the Direction active unless its objective, planned variants, and stop "
        "conditions are genuinely exhausted."
    )
    assert not (
        workspace / "research-library" / "experiments" / experiment_id / "v1" / "review.json"
    ).exists()
    assert len(reviewed_targets) == 2
    library = supervisor_research_library_root(workspace)
    private_reviews = [
        json.loads(path.read_text())
        for path in (
            library / "directions" / direction_id / "v1" / "review"
        ).glob("result.json")
    ] + [
        json.loads(path.read_text())
        for path in (
            library / "experiments" / experiment_id / "v1" / "review"
        ).glob("result.json")
    ]
    assert len(private_reviews) == 2
    assert all(
        row["review"]["claim_review_quality"] == "supported"
        for row in private_reviews
    )
    assert {
        row["review"]["session_audit"]["attempts"][0]["session_id"]
        for row in private_reviews
    } == {"review-session-1", "review-session-2"}
    assert not (evidence / "claim-reviews").exists()
    assert not (evidence / "claim-review-index").exists()


def test_research_claim_reviewer_returns_durable_conversation_links(tmp_path):
    evidence = tmp_path / "episode"
    bundle = evidence / "claim-review-test"
    bundle.mkdir(parents=True)
    attempts = []
    resume_requests = []

    valid_review = {
        "verdict": "supported",
        "primary_issue": "No material unsupported claim was found.",
        "findings": [{
            "claim": "The submitted research statement.",
            "judgment": "supported",
            "decision_critical": True,
            "reason": "The supplied evidence supports the bounded statement.",
            "evidence": ["submissions/target.json"],
        }],
        "scope_safe_interpretation": "Use the claim only within its stated scope.",
        "next_check": "Measure the next candidate against the same baseline.",
        "confidence": "high",
    }


    def run_session(workspace, _prompt, timeout=None, **kwargs):
        assert timeout is None
        assert kwargs["extra_environment"]["ATREX_AGENT_SANDBOX"] == "bwrap"
        attempt = len(attempts) + 1
        attempts.append(attempt)
        resume_requests.append(
            (kwargs.get("session_id"), kwargs.get("resume_session"))
        )
        capture_root = Path(kwargs["extra_environment"]["ATREX_SESSION_CAPTURE_DIR"])
        conversation = capture_root / f"run-{attempt}" / "conversation.jsonl"
        conversation.parent.mkdir(parents=True)
        conversation.write_text(
            json.dumps({"type": "session_end", "attempt": attempt}) + "\n",
            encoding="utf-8",
        )
        (conversation.parent / "token-usage.json").write_text(
            json.dumps({
                "state": "completed",
                "finished_at": "2026-11-12T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        (workspace / "review.json").write_text(
            json.dumps({} if attempt == 1 else valid_review),
            encoding="utf-8",
        )
        # Runtime-private/transient files are not reviewer evidence.  Input
        # integrity is enforced by the auxiliary bwrap view's read-only input
        # mounts, so unrelated files must not invalidate a valid review.
        (workspace / "scratch").mkdir(exist_ok=True)
        (workspace / "scratch" / "runtime-state.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return SimpleNamespace(
            exit_status=0,
            timed_out=False,
            session_id=kwargs.get("session_id") or f"provider-session-{attempt}",
            stdout_tail="",
            stderr_tail="",
        )

    campaign = SimpleNamespace(
        agent_boundary_environment=lambda _role: {
            "ATREX_AGENT_SANDBOX": "none"
        },
        _run_session=run_session,
        _account=lambda _result, _label: None,
    )
    archive_root = evidence / "review"
    review = Campaign.review_research_claim(
        campaign,
        bundle,
        archive_root=archive_root,
    )

    audit = review.pop("_review_session_audit")
    assert review == valid_review
    assert [row["status"] for row in audit["attempts"]] == ["invalid", "accepted"]
    assert [row["session_id"] for row in audit["attempts"]] == [
        "provider-session-1", "provider-session-2"
    ]
    for row in audit["attempts"]:
        conversation = archive_root / row["conversation_path"]
        token_usage = archive_root / row["token_usage_path"]
        assert conversation.is_file()
        assert conversation.name == "conversation.jsonl"
        assert token_usage.is_file()
        assert token_usage.name == "token-usage.json"

    resumed_review = Campaign.review_research_claim(
        campaign,
        bundle,
        resume_session_id="provider-session-2",
        archive_root=archive_root,
    )
    resumed_audit = resumed_review.pop("_review_session_audit")
    assert resumed_review == valid_review
    assert resume_requests[-1] == ("provider-session-2", True)
    assert resumed_audit["attempts"][0]["resumed"] is True
    assert resumed_audit["attempts"][0]["session_id"] == "provider-session-2"


def test_experiment_only_review_does_not_invent_direction_status(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("experiments",),
    )

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("experiments",),
        claim_reviewer=lambda _root: _claim_review(
            "supported",
            [_claim_finding("supported", critical=True)],
        ),
    )
    recorded = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Bounded observation",
            "intent": "Record one measured fact.",
            "interpretation": "The observation is valid within the measured scope.",
        },
    })

    assert recorded["review"]["claim_review_quality"] == "supported"
    assert "direction_status" not in recorded["review"]
    assert recorded["review"]["runtime_action"] == (
        "Use the archived record as reviewed research evidence."
    )


def test_research_claim_reviewer_launch_failures_are_privately_auditable(tmp_path):
    evidence = tmp_path / "episode"
    bundle = evidence / "claim-review-test"
    bundle.mkdir(parents=True)

    def run_session(*_args, **_kwargs):
        raise ValueError("Unknown Agent workspace role: research-claim-review")

    campaign = SimpleNamespace(
        agent_boundary_environment=lambda _role: {
            "ATREX_AGENT_SANDBOX": "none"
        },
        _run_session=run_session,
        _account=lambda _result, _label: None,
    )
    with pytest.raises(RuntimeError) as failure:
        Campaign.review_research_claim(campaign, bundle)

    audit = failure.value.review_session_audit
    assert [row["status"] for row in audit["attempts"]] == [
        "capture_missing",
        "capture_missing",
    ]
    assert all(row["session_id"] == "" for row in audit["attempts"])
    assert all(
        row["error"]
        == {
            "type": "ValueError",
            "message": "Unknown Agent workspace role: research-claim-review",
        }
        for row in audit["attempts"]
    )


def test_research_claim_review_bwrap_unavailable_fails_closed(monkeypatch):
    with pytest.raises(
        RuntimeError,
        match="Research claim review requires Bubblewrap.*native execution is not permitted",
    ):
        sandbox_executable({
            "ATREX_AGENT_SANDBOX": "none",
            "ATREX_AGENT_WORKSPACE_ROLE": "research-claim-review",
        })

    environment = {
        "ATREX_AGENT_SANDBOX": "bwrap",
        "ATREX_AGENT_WORKSPACE_ROLE": "research-claim-review",
        "ATREX_BWRAP_EXECUTABLE": "missing-bwrap",
        "PATH": "",
    }
    monkeypatch.setattr("aka.light.agent.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "aka.light.agent.sandbox.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Research claim review requires Bubblewrap.*"
            "native execution is not permitted"
        ),
    ):
        sandbox_executable(environment)


def test_research_claim_review_auxiliary_workspace_publishes_only_review(tmp_path):
    workspace = tmp_path / "claim-review"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    (workspace / "submissions").mkdir()
    (workspace / "submissions" / "target.json").write_text(
        '{"action":"propose"}\n', encoding="utf-8"
    )
    (workspace / "evidence-manifest.json").write_text(
        '{"claim_type":"direction"}\n', encoding="utf-8"
    )
    (workspace / "private-supervisor-note.json").write_text(
        "must stay private\n", encoding="utf-8"
    )

    view = AuxiliaryWorkspace(workspace, home, "research-claim-review")
    try:
        assert (view.root / "submissions" / "target.json").is_file()
        assert (view.root / "evidence-manifest.json").is_file()
        assert not (view.root / "private-supervisor-note.json").exists()
        (view.root / "review.json").write_text(
            '{"verdict":"supported"}\n', encoding="utf-8"
        )
        view.publish()
    finally:
        view.close()

    assert json.loads((workspace / "review.json").read_text()) == {
        "verdict": "supported"
    }
    assert (workspace / "private-supervisor-note.json").read_text() == (
        "must stay private\n"
    )


def test_light_agent_runtime_uses_native_resume_command(tmp_path):
    commands = []

    def process_runner(command, *, cwd, timeout, env):
        commands.append((list(command), cwd, timeout, dict(env)))
        return "", "", 0, False

    runtime = build_agent_runtime("claude", process_runner=process_runner)
    result = runtime.run(RuntimeAgentRunRequest(
        workspace=tmp_path,
        prompt="Review the revised claim.",
        timeout_s=30,
        reasoning_effort="high",
        session_id="11111111-1111-4111-8111-111111111111",
        resume_session=True,
    ))
    command = commands[0][0]
    assert "--resume" in command
    assert command[command.index("--resume") + 1] == result.session_id
    assert "--session-id" not in command


def test_unsupported_research_claims_are_not_archived(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )

    def reviewer(_root):
        return {
            "verdict": "unsupported",
            "primary_issue": "The decision-critical mechanism has no evidence.",
            "findings": [{
                "claim": "The proposed mechanism will reduce latency.",
                "judgment": "unsupported",
                "decision_critical": True,
                "reason": "No isolating measurement or source evidence is cited.",
                "evidence": ["submissions/target.json"],
            }],
            "scope_safe_interpretation": "The mechanism remains an untested idea.",
            "next_check": "Run one bounded mechanism-isolating probe.",
            "confidence": "high",
            "_review_session_audit": {
                "schema": "atrex.research-claim-review-session.v1",
                "attempts": [{
                    "attempt": 1,
                    "session_id": "rejected-review-session",
                    "status": "accepted",
                    "exit_status": 0,
                    "timed_out": False,
                    "conversation_path": (
                        "claim-review-sessions/rejected/"
                        "run-1/conversation.jsonl"
                    ),
                }],
            },
        }

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        claim_reviewer=reviewer,
    )
    proposed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Unsupported split",
            "change_scope": "major_redesign",
            "objective": "Reduce latency through an uncited architecture path.",
            "applicability": "All workload shapes.",
            "expected_benefit": "Large speedup.",
            "plan": ["Implement the entire redesign"],
        },
    })
    assert "review" not in proposed
    with pytest.raises(AgentRequestError) as direction_rejected:
        service.execute({
            "operation": "direction_update",
            "request": {
                "action": "start",
                "direction_id": proposed["direction_id"],
                "analysis": "Commit resources to the proposed redesign.",
            },
        })
    response = direction_rejected.value.response
    assert response["error"]["code"] == "research_claim_unsupported"
    assert response["error"]["review"]["claim_review_quality"] == "unsupported"
    assert "evidence_manifest" not in response["error"]["review"]
    assert "will not be archived" in response["error"]["review"]["runtime_action"]
    direction_events = [
        event
        for event in ResearchStore(tmp_path / "research").events()
        if event["type"] == "direction.observed"
    ]
    assert [event["payload"]["direction"]["revision"] for event in direction_events] == [0]
    assert (workspace / proposed["archive_path"]).is_file()

    experiment_evidence = journals / "episodes" / "e00000002"
    experiment_evidence.mkdir(parents=True)
    initialize_journal(
        experiment_evidence / "journal.json",
        episode=2,
        base_commit="b" * 40,
        branch="experiment-work",
        tool_modules=("experiments",),
    )
    experiment_service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=experiment_evidence,
        research_root=tmp_path / "experiment-research",
        tool_modules=("experiments",),
        claim_reviewer=reviewer,
    )
    (workspace / "scratch").mkdir()
    (workspace / "scratch" / "unsupported-probe.py").write_text(
        "print('audit me')\n", encoding="utf-8"
    )
    with pytest.raises(AgentRequestError) as experiment_rejected:
        experiment_service.execute({
            "operation": "experiment_record",
            "request": {
                "name": "Unsupported interpretation",
                "intent": "Record an unmeasured causal claim.",
                "interpretation": "The mechanism is proven without measurement.",
                "references": ["scratch/unsupported-probe.py"],
            },
        })
    response = experiment_rejected.value.response
    assert response["error"]["code"] == "research_claim_unsupported"
    assert response["error"]["review"]["claim_review_quality"] == "unsupported"
    assert load_journal(experiment_evidence / "journal.json")["experiments"] == []
    assert not (workspace / "experiments").exists()
    library = supervisor_research_library_root(workspace)
    unsupported_audits = [
        *library.glob("directions/*/attempts/*/review/result.json"),
        *library.glob("experiments/*/attempts/*/review/result.json"),
    ]
    assert len(unsupported_audits) == 2
    for path in unsupported_audits:
        outcome = json.loads(path.read_text())
        assert outcome["status"] == "unsupported"
        assert outcome["error"]["error"]["review"]["claim_review_quality"] == "unsupported"
        assert outcome["review"]["evidence_manifest"]["claim_type"] in {
            "direction", "experiment"
        }
        assert outcome["review"]["session_audit"]["attempts"][0][
            "conversation_path"
        ].endswith("/conversation.jsonl")
        assert "_review_session_audit" not in outcome["review"]
        assert outcome["claim_type"] in {"direction", "experiment"}
    experiment_audit_root = next(
        path.parents[1]
        for path in unsupported_audits
        if json.loads(path.read_text())["claim_type"] == "experiment"
    )
    reference_manifest = json.loads(
        (experiment_audit_root / "references" / "manifest.json").read_text()
    )
    assert reference_manifest["capture_status"] == "captured"
    assert reference_manifest["files"][0]["path"] == "scratch/unsupported-probe.py"
    assert (
        experiment_audit_root / "references" / "files"
        / "scratch" / "unsupported-probe.py"
    ).read_text() == "print('audit me')\n"


def test_partial_experiment_is_projected_but_cannot_support_terminal_verdict(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
    )

    def reviewer(root):
        target = json.loads((root / "submissions" / "target.json").read_text())
        partial = "experiment_id" in target
        verdict = "partially_supported" if partial else "supported"
        return {
            "verdict": verdict,
            "primary_issue": (
                "The mechanism is not isolated." if partial else "The proposal is bounded."
            ),
            "findings": [{
                "claim": "The submitted research statement.",
                "judgment": verdict,
                "decision_critical": True,
                "reason": "The measurement supports only a narrower interpretation.",
                "evidence": ["submissions/target.json"],
            }],
            "scope_safe_interpretation": "Treat the result as a bounded observation only.",
            "next_check": "Run one mechanism-isolating comparison.",
            "confidence": "high",
        }

    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions", "experiments"),
        claim_reviewer=reviewer,
    )
    direction_id = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split KV",
            "change_scope": "major_redesign",
            "objective": "Reduce long-context latency with KV-parallel architecture families.",
            "applicability": "Long-context shapes.",
            "expected_benefit": "Reduce latency.",
            "plan": ["Measure one bounded candidate."],
        },
    })["direction_id"]
    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Run the bounded candidate.",
        },
    })
    record_id, _ = _passing_full_measurement(
        MeasurementStore(tmp_path / "scope" / "measurements"),
        b"def kernel():\n    return 1\n",
        "partial-review",
        {"shape_a": 1.0},
    )
    recorded = service.execute({
        "operation": "experiment_record",
        "request": {
            "name": "Split KV observation",
            "intent": "Measure whether the candidate runs.",
            "gateway_record_ids": [record_id],
            "interpretation": "The candidate ran, but the mechanism remains unisolated.",
        },
    })
    experiment_id = recorded["experiment_id"]
    projected = json.loads((workspace / recorded["archive_path"]).read_text())
    assert projected["claim_review_quality"] == "partially_supported"
    assert projected["scope_safe_interpretation"] == (
        "Treat the result as a bounded observation only."
    )
    assert projected["scientific_next_check"] == "Run one mechanism-isolating comparison."
    assert "review_safe_interpretation" not in projected
    assert "review_scientific_next_check" not in projected

    closed = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "complete",
            "direction_id": direction_id,
            "analysis": "The bounded Direction is complete; retain the narrower observation.",
            "supporting_experiment_ids": [experiment_id],
        },
    })
    assert closed["status"] == "recorded"


def test_missing_experiment_reference_is_repairable_and_records_nothing(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("experiments",),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("experiments",),
    )
    record_id, _ = _measurement(
        MeasurementStore(tmp_path / "scope" / "measurements"),
        b"def kernel():\n    return 1\n",
        "missing-reference",
    )

    with pytest.raises(AgentRequestError) as rejected:
        service.execute({
            "operation": "experiment_record",
            "request": {
                "name": "missing material",
                "intent": "Verify missing references are rejected before mutation.",
                "gateway_record_ids": [record_id],
                "interpretation": "No Experiment should be written.",
                "references": ["scratch/does-not-exist.txt"],
            },
        })

    assert rejected.value.response["error"]["code"] == "reference_not_found"
    assert rejected.value.response["issues"][0]["path"] == "references.0"
    assert ResearchStore(tmp_path / "research").read()["experiments"] == {}
    assert not (evidence / "claim-audit").exists()


    (workspace / "scratch").mkdir()
    (workspace / "scratch" / "target.txt").write_text("not through a link\n")
    (workspace / "scratch" / "linked.txt").symlink_to("target.txt")
    with pytest.raises(AgentRequestError) as linked:
        service.execute({
            "operation": "experiment_record",
            "request": {
                "name": "linked material",
                "intent": "Verify links are rejected before mutation.",
                "gateway_record_ids": [record_id],
                "interpretation": "No Experiment should be written.",
                "references": ["scratch/linked.txt"],
            },
        })
    assert linked.value.response["error"]["code"] == "reference_symlink_forbidden"
    assert ResearchStore(tmp_path / "research").read()["experiments"] == {}


def test_direction_draft_projection_tracks_lifecycle_without_status_directories(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions",),
    )

    proposed_result = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Split KV",
            "change_scope": "major_redesign",
            "objective": "Reduce long-context latency with KV-parallel architecture families.",
            "applicability": "Long-context attention shapes.",
            "expected_benefit": "Reduce latency by exposing CTA parallelism.",
            "plan": ["Compile a bounded probe"],
        },
    })
    direction_id = proposed_result["direction_id"]
    proposed = workspace / proposed_result["archive_path"]
    assert proposed.is_file()
    assert json.loads(proposed.read_text())["status"] == "proposed"
    assert not (workspace / "directions").exists()

    started = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "This is the highest-value bounded probe.",
        },
    })
    assert started["archive_path"] == (
        f"research-library/directions/{direction_id}/v1/direction.json"
    )
    assert json.loads(proposed.read_text())["status"] == "proposed"
    assert json.loads((workspace / started["archive_path"]).read_text())[
        "status"
    ] == "in_progress"

    progress = (workspace / "PROGRESS.md").read_text()
    assert "### Local work queue" in progress
    assert "status `in_progress`" in progress
    assert "### Shared direction pool\n\n_None._" in progress
    assert direction_id in progress

    service.execute({
        "operation": "direction_update",
        "request": {
            "action": "block",
            "direction_id": direction_id,
            "analysis": "The required compiler feature is unavailable.",
            "reopen_condition": "The target compiler exposes the required feature.",
        },
    })
    progress = (workspace / "PROGRESS.md").read_text()
    assert (
        "**Current real blocker:** Split KV — The required compiler feature is unavailable; "
        "reopen when: The target compiler exposes the required feature."
    ) in progress


def test_submit_kernel_records_review_without_finalizing_episode(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    journal_path = evidence / "journal.json"
    initialize_journal(
        journal_path,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=campaign,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        session_capture_root=tmp_path / "sessions/e0001/sessions",
    )
    usage = tmp_path / "sessions/e0001/sessions/run-main/token-usage.json"
    usage.parent.mkdir(parents=True)
    usage.write_text(json.dumps({
        "session_id": "main",
        "started_at": "2026-11-12T00:00:00+00:00",
        "state": "running",
        "accounting_basis": "provider_result_usage",
        "total": {
            "input_tokens": 90,
            "output_tokens": 10,
            "reasoning_tokens": 5,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 100,
            "measurement": "exact",
        },
    }), encoding="utf-8")
    refreshes = []
    monkeypatch.setattr(
        service,
        "publish_research_files",
        lambda **kwargs: refreshes.append(kwargs),
    )
    commit_id = "b" * 40
    result = service.execute(
        {
            "operation": "kernel_submit",
            "request": {"kernel_commit_id": commit_id},
        },
        review_kernel=lambda submitted: {
            "accepted": submitted == commit_id,
            "review": {
                "source_contract": "pass",
                "dsl_policy": "not_applicable",
                "production_policy": "not_applicable",
            },
            "implementation_review": {
                "availability": "available",
                "technologies": ["cuda"],
                "dsl_policy": "pinned",
            },
            "diagnostics": [],
            "evaluation_evidence": {
                "status": "passed",
                "gateway_record_id": "gateway-" + "c" * 32,
                "latency_us_geomean": 4.25,
                "measured_shape_count": 3,
            },
        },
    )
    assert result == {
        "status": "reviewed",
        "kernel_commit_id": commit_id,
        "accepted": True,
        "review": {
            "source_contract": "pass",
            "dsl_policy": "not_applicable",
            "production_policy": "not_applicable",
        },
        "diagnostics": [],
        "evaluation_evidence": {
            "status": "passed",
            "gateway_record_id": "gateway-" + "c" * 32,
            "latency_us_geomean": 4.25,
            "measured_shape_count": 3,
        },
        "runtime_action": (
            "The exact commit has a passing full Evaluate and passed final-source "
            "review. Continue development or end the Session."
        ),
    }
    submission_rows = submissions(tmp_path / "scope" / "submissions")
    assert len(submission_rows) == 1
    submission = submission_rows[0]
    assert submission["kernel_commit_id"] == commit_id
    assert submission["accepted"] is True
    assert submission["implementation_review"] == {
        "availability": "available",
        "technologies": ["cuda"],
        "dsl_policy": "pinned",
    }
    assert submission["token_checkpoint"]["total_tokens"] == 100
    assert (tmp_path / "scope" / submission["token_checkpoint"]["path"]).is_file()
    assert refreshes == [{
        "synchronize": False,
        "direction_ids": frozenset(),
        "experiment_ids": frozenset(),
        "record_ids": frozenset(),
    }]


def test_submit_kernel_requires_saved_full_evaluation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=campaign,
        evidence_root=evidence,
        research_root=tmp_path / "research",
    )
    result = service.execute(
        {
            "operation": "kernel_submit",
            "request": {"kernel_commit_id": "b" * 40},
        },
        review_kernel=lambda _submitted: {
            "accepted": False,
            "review": {
                "source_contract": "pass",
                "dsl_policy": "not_applicable",
                "production_policy": "not_applicable",
            },
            "diagnostics": [],
            "evaluation_evidence": {"status": "missing"},
        },
    )
    assert result["status"] == "needs_evaluation"
    assert result["accepted"] is False
    assert result["evaluation_evidence"] == {"status": "missing"}
    assert "Run a full Evaluate for this exact commit" in result["runtime_action"]

    incomplete = service.execute(
        {
            "operation": "kernel_submit",
            "request": {"kernel_commit_id": "b" * 40},
        },
        review_kernel=lambda _submitted: {
            "accepted": False,
            "review": {
                "source_contract": "pass",
                "dsl_policy": "not_applicable",
                "production_policy": "not_applicable",
            },
            "diagnostics": [],
            "evaluation_evidence": {
                "status": "failed",
                "gateway_record_id": "gateway-" + "d" * 32,
                "reason_code": "performance_result_incomplete",
                "reason": "latency_us_by_shape is missing",
            },
        },
    )
    assert incomplete["status"] == "rejected"
    assert incomplete["accepted"] is False
    assert "performance result is incomplete or unparseable" in incomplete["runtime_action"]
    assert "same commit" in incomplete["runtime_action"]






def test_submit_kernel_schema_and_prompt_have_no_handoff_contract():
    schema = journal_schema(
        "kernel_submit", {"kernel_commit_id": "a" * 40}, research_records=False
    )
    assert schema["required"] == ["kernel_commit_id"]
    assert "handoff_reason" not in schema["properties"]
    prompt = (
        Path(__file__).resolve().parents[1]
        / "src/aka/light/workflow/prompts/episode.md"
    ).read_text(encoding="utf-8")
    assert "Repeat `probe → candidate → measurement → revert/keep` cycles" in prompt
    assert "Use `submit-kernel` when you judge an exact commit ready" in prompt
    assert "greatest plausible full-workload impact" in prompt
    assert "instead of avoiding high-upside work" in prompt
    assert "Manage development with Agent Git" in prompt


def test_episode_report_is_not_a_runtime_operation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=campaign,
        evidence_root=evidence,
        research_root=tmp_path / "research",
    )
    with pytest.raises(ValueError, match="unsupported Runtime research/report operation"):
        service.execute({"operation": "episode_report", "request": {}})




def test_direction_proposal_schema_requires_applicability_and_expected_benefit():
    schema = journal_schema(
        "direction_update", {"action": "propose"}, research_records=True
    )
    assert "applicability" in schema["required"]
    assert "expected_benefit" in schema["required"]
    assert "objective" in schema["required"]
    assert "hypothesis" not in schema["properties"]
    assert schema["properties"]["applicability"]["type"] == "string"
    assert schema["properties"]["expected_benefit"]["type"] == "string"
    assert "architecture goal" in schema["properties"]["objective"]["description"]
    assert "implementation decision" in schema["properties"]["expected_benefit"]["description"]
    assert "probe-led Direction" in schema["properties"]["change_scope"]["description"]


def test_direction_schema_explains_field_types_and_mode_specific_closure_evidence():
    proposal = journal_schema(
        "direction_update",
        {"action": "propose"},
        research_records=True,
        directions_enabled=True,
        experiments_enabled=False,
    )
    assert proposal["properties"]["change_scope"]["enum"] == [
        "incremental",
        "major_redesign",
    ]
    assert proposal["properties"]["stop_conditions"]["type"] == "array"
    assert "Optional array" in proposal["properties"]["stop_conditions"]["description"]

    direction_only = journal_schema(
        "direction_update",
        {"action": "abandon"},
        research_records=True,
        directions_enabled=True,
        experiments_enabled=False,
    )
    assert "gateway_record_ids" in direction_only["properties"]
    assert "Optional Gateway Record evidence" in (
        direction_only["properties"]["gateway_record_ids"]["description"]
    )
    assert "supporting_experiment_ids" not in direction_only["properties"]
    assert "hypothesis_status" not in direction_only["properties"]

    with_experiments = journal_schema(
        "direction_update",
        {"action": "abandon"},
        research_records=True,
        directions_enabled=True,
        experiments_enabled=True,
    )
    assert "supporting_experiment_ids" in with_experiments["properties"]
    assert with_experiments["properties"]["supporting_experiment_ids"]["minItems"] == 0
    assert "gateway_record_ids" not in with_experiments["properties"]


def test_http_error_feedback_uses_exact_research_module_selection(monkeypatch):
    captured = {}

    def project(result, **options):
        captured.update(options)
        return augment_error(result, **options)

    owner = SimpleNamespace(
        journal_factory=JournalFactory(tool_modules=("directions",)),
        root=Path("/private/runtime"),
        audit_root=Path("/private/audit"),
        config=SimpleNamespace(
            private_reference_dir=None,
            url="http://runtime.invalid",
            atrex_bench_root=None,
        ),
        environment={},
        plugins=SimpleNamespace(catalog=lambda: ()),
    )
    handler = SimpleNamespace(
        server=SimpleNamespace(owner=owner),
        capability=None,
        path="/v1/journal/execute",
        decoded_request={
            "operation": "direction_update",
            "request": {"action": "complete"},
        },
        close_connection=False,
        wfile=BytesIO(),
        send_response=lambda _status: None,
        send_header=lambda _name, _value: None,
        end_headers=lambda: None,
    )
    response = {
        "ok": False,
        "repairable": True,
        "error": {"code": "invalid_fields", "message": "invalid fields"},
    }
    monkeypatch.setattr("aka.light.runtime.server.augment_error", project)

    _Handler.reply(handler, 400, response)

    assert captured["research_records"] is True
    assert captured["directions_enabled"] is True
    assert captured["experiments_enabled"] is False
    payload = json.loads(handler.wfile.getvalue())
    properties = payload["request_schema"]["properties"]
    assert "gateway_record_ids" in properties
    assert "supporting_experiment_ids" not in properties


def test_direction_closure_before_start_returns_exact_recovery(tmp_path):
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    journal = evidence / "journal.json"
    initialize_journal(
        journal,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )
    direction_id = update_direction(
        journal,
        campaign,
        {
            "action": "propose",
            "name": "Capability probe",
            "change_scope": "incremental",
            "objective": "Resolve the consequential implementation choice with bounded probes.",
            "applicability": "The current workload.",
            "expected_benefit": "Avoid an unproductive implementation path.",
            "plan": ["Run one bounded probe."],
        },
    )["direction_id"]

    with pytest.raises(ValueError) as raised:
        update_direction(
            journal,
            campaign,
            {
                "action": "abandon",
                "direction_id": direction_id,
                "analysis": "The probe is no longer useful.",
            },
        )
    message = str(raised.value)
    assert f"submit action=start for {direction_id} first" in message
    assert "then retry action=abandon" in message

    update_direction(
        journal,
        campaign,
        {
            "action": "start",
            "direction_id": direction_id,
            "analysis": "Start the bounded investigation.",
        },
    )
    result = update_direction(
        journal,
        campaign,
        {
            "action": "abandon",
            "direction_id": direction_id,
            "analysis": "The probe is no longer useful.",
        },
    )
    assert result["status"] == "recorded"
    assert load_journal(journal)["direction_events"][-1]["action"] == "abandon"


def test_episode_may_sequentially_start_more_than_three_directions(tmp_path):
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    journal = evidence / "journal.json"
    initialize_journal(
        journal,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )

    for index in range(6):
        proposed = update_direction(
            journal,
            campaign,
            {
                "action": "propose",
                "name": f"Direction {index}",
                "change_scope": "incremental",
                "objective": f"Select and optimize the better Kernel path for regime {index}.",
                "applicability": "The current workload.",
                "expected_benefit": "Resolve one consequential implementation choice.",
                "plan": ["Run the discriminating probe."],
            },
        )
        direction_id = proposed["direction_id"]
        update_direction(
            journal,
            campaign,
            {
                "action": "start",
                "direction_id": direction_id,
                "analysis": "Highest-value unresolved choice.",
            },
        )
        update_direction(
            journal,
            campaign,
            {
                "action": "abandon",
                "direction_id": direction_id,
                "analysis": "The bounded probe leaves no worthwhile next step.",
            },
        )

    events = load_journal(journal)["direction_events"]
    assert sum(event["action"] == "start" for event in events) == 6


def test_paused_direction_can_restart_but_terminal_direction_cannot(tmp_path):
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    journal = evidence / "journal.json"
    initialize_journal(
        journal,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )
    direction_id = update_direction(
        journal,
        campaign,
        {
            "action": "propose",
            "name": "Capability probe",
            "change_scope": "major_redesign",
            "objective": "Determine whether the capability unlocks a faster Kernel architecture.",
            "applicability": "The target GPU.",
            "expected_benefit": "Enable the structural redesign.",
            "plan": ["Compile the bounded probe."],
        },
    )["direction_id"]
    update_direction(journal, campaign, {
        "action": "start", "direction_id": direction_id, "analysis": "Run the probe."
    })
    update_direction(journal, campaign, {
        "action": "defer",
        "direction_id": direction_id,
        "analysis": "The required compiler is temporarily unavailable.",
        "reopen_condition": "The compiler becomes available.",
    })
    assert update_direction(journal, campaign, {
        "action": "start", "direction_id": direction_id, "analysis": "Compiler restored."
    })["status"] == "recorded"
    update_direction(journal, campaign, {
        "action": "abandon",
        "direction_id": direction_id,
        "analysis": "The path is no longer relevant.",
    })
    with pytest.raises(ValueError, match="cannot start from status abandoned"):
        update_direction(journal, campaign, {
            "action": "start", "direction_id": direction_id, "analysis": "Try again."
        })


def test_second_local_direction_start_returns_precise_recovery(tmp_path):
    campaign = tmp_path / "scope" / "journals"
    evidence = campaign / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    journal = evidence / "journal.json"
    initialize_journal(
        journal,
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions",),
    )

    direction_ids = []
    for index in range(2):
        direction_ids.append(
            update_direction(
                journal,
                campaign,
                {
                    "action": "propose",
                    "name": f"Direction {index}",
                    "change_scope": "local",
                    "objective": f"Select and optimize the better Kernel path for regime {index}.",
                    "applicability": "The current workload.",
                    "expected_benefit": "Resolve one consequential implementation choice.",
                    "plan": ["Run the discriminating probe."],
                },
            )["direction_id"]
        )
    update_direction(
        journal,
        campaign,
        {
            "action": "start",
            "direction_id": direction_ids[0],
            "analysis": "Highest-value unresolved choice.",
        },
    )

    second = update_direction(
        journal,
        campaign,
        {
            "action": "start",
            "direction_id": direction_ids[1],
            "analysis": "Run another independent bounded investigation.",
        },
    )
    assert second["status"] == "recorded"
    starts = [
        event["direction_id"]
        for event in load_journal(journal)["direction_events"]
        if event["action"] == "start"
    ]
    assert starts == direction_ids










def test_progress_is_projected_even_when_research_modules_are_disabled(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=(),
    )
    source = b"def kernel():\n    return 1\n"
    (workspace / "kernel.py").write_bytes(source)
    agent_git = AgentGit(workspace, tmp_path / "agent.bundle")
    agent_git.prepare()
    record_id, _ = _measurement(
        MeasurementStore(tmp_path / "scope" / "measurements"),
        source,
        "unrecorded-mode",
        kernel_id=agent_git.head(),
    )

    service.publish_research_files()

    progress = (workspace / "PROGRESS.md").read_text()
    assert "# Current state" in progress
    assert "No accepted submission has eligible full-Evaluate evidence yet" in progress
    assert "## Directions" not in progress
    assert "directions/<status>" not in progress
    assert "experiments/<experiment_id>" not in progress
    assert "**Hypothesis:**" not in progress
    assert (workspace / "records").is_dir()
    assert (workspace / "records" / f"{record_id}.json").is_file()
    assert not (workspace / "directions").exists()
    assert not (workspace / "experiments").exists()


def test_measurements_checkpoint_working_and_best_lineage_commits(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        origin={"trajectory": 1, "episode": 1},
    )
    agent_git = AgentGit(workspace, tmp_path / "agent.bundle")
    (workspace / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    agent_git.prepare()
    first = agent_git.head()
    (workspace / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    slower = agent_git.checkpoint("slower measured Kernel")
    (workspace / "kernel.py").write_text("VALUE = 3\n", encoding="utf-8")
    diagnostic = agent_git.checkpoint("diagnostic Kernel")
    (workspace / "kernel.py").write_text("VALUE = 4\n", encoding="utf-8")
    comparison_candidate = agent_git.checkpoint("ABBA candidate")
    assert all(isinstance(value, str) for value in (
        first, slower, diagnostic, comparison_candidate
    ))
    service.observe_measurement({
        "operation": "evaluate",
        "status": "completed",
        "kernel_commit_id": first,
        "result": {
            "correct": True,
            "mode": "full",
            "input_scope": "contract",
            "latency_us_geomean": 10.0,
        },
    })
    service.observe_measurement({
        "operation": "evaluate",
        "status": "completed",
        "kernel_commit_id": slower,
        "result": {
            "correct": True,
            "mode": "full",
            "input_scope": "contract",
            "latency_us_geomean": 12.0,
        },
    })
    service.observe_measurement({
        "operation": "profile",
        "status": "failed",
        "kernel_commit_id": diagnostic,
        "result": {"error": {"error_class": "timeout"}},
    })
    # ABBA has two identities and must not replace the single working HEAD.
    service.observe_measurement({
        "operation": "evaluate",
        "status": "completed",
        "candidate_kernel_commit_id": comparison_candidate,
        "baseline_kernel_commit_id": first,
        "result": {"correct": True, "mode": "full", "input_scope": "contract"},
    })

    state = service.research.read()["lineages"]["T1"]
    assert state["working_kernel_id"] == diagnostic
    assert state["last_correct_kernel_id"] == slower
    assert "lineage_best_kernel_id" not in state
    assert "lineage_best_latency_us" not in state
    service.publish_research_files(synchronize=False)
    progress = (workspace / "PROGRESS.md").read_text()
    assert f"**Last measured working Kernel commit ID:** `{diagnostic}`" in progress
    assert "Best verified lineage Kernel" not in progress
    assert "Best measured passing Kernel" not in progress
    assert "Accepted-submission per-Shape Pareto frontier" in progress


def test_private_historical_kernel_snapshot_is_not_projected_without_git_commit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = b"def kernel():\n    return 1\n"
    (workspace / "kernel.py").write_bytes(source)
    AgentGit(workspace, tmp_path / "agent.bundle").prepare()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=(),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("experiments",),
    )
    store = MeasurementStore(tmp_path / "scope" / "measurements")
    record_id, private_kernel_id = _measurement(
        store,
        source,
        "legacy-private-snapshot",
        kernel_id="f" * 40,
    )

    service.publish_research_files(synchronize=False)

    assert store.read(record_id)["kernels"]["candidate"]["kernel_id"] == private_kernel_id
    assert not (workspace / "records" / f"{record_id}.json").exists()
    progress = (workspace / "PROGRESS.md").read_text(encoding="utf-8")
    assert private_kernel_id not in progress
    assert record_id not in progress


def test_progress_projection_mentions_only_enabled_experiment_module(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("experiments",),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("experiments",),
    )

    service.publish_research_files()

    progress = (workspace / "PROGRESS.md").read_text()
    assert "research-library/experiments/<experiment_id>/vN/experiment.json" in progress
    assert "experiments/index.json" not in progress
    assert not (workspace / "experiments").exists()
    assert "## Directions" not in progress
    assert "directions/<status>" not in progress
    assert "**Hypothesis:**" not in progress


def test_combined_research_mode_removes_stale_experiment_only_index(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    stale_index = workspace / "experiments" / "index.json"
    stale_index.write_text('[{"experiment_id":"stale"}]\n')
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json",
        episode=1,
        base_commit="a" * 40,
        branch="work",
        tool_modules=("directions", "experiments"),
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
    )

    service.publish_research_files()

    assert not stale_index.exists()


@pytest.mark.parametrize(
    ("modules", "required", "unavailable"),
    [
        (
            ("directions",),
            ('"action": "propose"', "derived_from_direction_ids", "gateway_record_ids"),
            "record-experiment",
        ),
        (
            ("experiments",),
            ('"interpretation"', "experiment_id", "wiki_usage_status"),
            "update-direction",
        ),
    ],
)
def test_partial_research_modes_install_complete_module_specific_reference(
    tmp_path, modules, required, unavailable
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from aka.light.workspace.assets import link_runtime

    link_runtime(workspace, research_records=True, research_modules=modules)
    journal = (
        workspace / "skills/runtime-records/references/journal.md"
    ).read_text()
    for marker in required:
        assert marker in journal
    assert unavailable not in journal


def test_runtime_install_does_not_mutate_tracked_gitignore(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (workspace / "tools").mkdir()
    (workspace / "tools/record.py").write_text("# retired wrapper\n", encoding="utf-8")
    from aka.light.workspace.agent_git import AgentGit, initialize_gitignore
    from aka.light.workspace.assets import link_runtime

    initialize_gitignore(workspace)
    link_runtime(workspace, research_records=False, research_modules=())
    assert not (workspace / "tools/record.py").exists()
    ignores = (workspace / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.atrex_long_horizon/" in ignores
    assert "/verification_artifacts/.atrex_long_horizon_verify/" in ignores
    assert "/memory/live.json" not in ignores
    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    repository.prepare()
    before = (workspace / ".gitignore").read_bytes()
    before_head = repository.head()

    link_runtime(workspace, research_records=False, research_modules=())

    assert (workspace / ".gitignore").read_bytes() == before
    assert repository.head() == before_head
    assert subprocess.run(
        ["git", "diff", "--quiet", "--", ".gitignore"],
        cwd=workspace,
    ).returncode == 0


def test_projected_record_is_directly_readable_without_wrapper(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from aka.light.workspace.assets import link_runtime

    link_runtime(workspace, research_records=False, research_modules=())
    record_id = "gateway-" + "1" * 32
    records = workspace / "records"
    records.mkdir()
    (records / f"{record_id}.json").write_text(
        json.dumps({
            "operation": "dev",
            "status": "completed",
            "gateway_record_id": record_id,
            "kernel_commit_id": "a" * 40,
            "result": {
                "correct": True,
                "latency_us_geomean": 4.5,
                "latency_us_by_shape": {"7": 5.25},
                "stdout": "x" * 70000,
                "stderr": "",
            },
        }),
        encoding="utf-8",
    )
    assert not (workspace / "tools/record.py").exists()
    record = json.loads((records / f"{record_id}.json").read_text(encoding="utf-8"))
    assert record["result"]["correct"] is True
    assert record["result"]["latency_us_by_shape"]["7"] == 5.25
    assert record["result"]["stdout"][100:228] == "x" * 128


def test_agent_git_keeps_single_file_auxiliary_metadata_private(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(
        "def run(x, y):\n    return x + y\n", encoding="utf-8"
    )
    (workspace / "solution.json").write_text(
        json.dumps(
            {
                "name": "vector_add",
                "spec": {
                    "languages": ["python", "triton"],
                    "dependencies": ["torch", "triton"],
                    "entry_point": "kernel.py::run",
                },
                "sources": [{"path": "kernel.py", "role": "candidate"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    repository.prepare()

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "solution.json"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    assert tracked.returncode != 0
    exclude = (workspace / ".git" / "info" / "exclude").read_text(
        encoding="utf-8"
    )
    assert "solution.json" in exclude
    assert "!/solution.json" not in exclude


def test_agent_git_reincludes_cuda_manifest_candidate_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from aka.light.workspace.sources.cuda_manifest import POLICY
    from aka.light.workspace.sources.private_state import bind

    bind(workspace, file_system="cuda_manifest", contract=POLICY)
    (workspace / "solution.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "vector_add",
                "version": "1",
                "source": {"type": "local", "files": ["kernel.cu"]},
                "entry_point": {
                    "type": "cuda",
                    "build": {
                        "toolchain": "torch_extension",
                        "sources": ["kernel.cu"],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "kernel.cu").write_text(
        'extern "C" __global__ void kernel() {}\n', encoding="utf-8"
    )

    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    repository.prepare()

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "solution.json"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    assert tracked.returncode == 0
    exclude = (workspace / ".git" / "info" / "exclude").read_text(
        encoding="utf-8"
    )
    assert "!/solution.json" in exclude




def test_light_ignores_legacy_journal_files_and_creates_no_transaction_file(tmp_path):
    workspace = tmp_path / "workspace"
    legacy_root = workspace / ".atrex_long_horizon"
    legacy_root.mkdir(parents=True)
    (legacy_root / "journal.json").write_text(
        json.dumps({"state": "candidate_ready", "experiments": [{"id": "legacy"}]}),
        encoding="utf-8",
    )
    journals = tmp_path / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
        tool_modules=("directions",),
    )
    result = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose",
            "name": "Ignore legacy state",
            "change_scope": "incremental",
            "objective": "Validate directly against the central Research Store.",
            "applicability": "Current Light campaigns.",
            "expected_benefit": "Avoid legacy transaction-state coupling.",
            "plan": ["Record one central Direction revision."],
        },
    })

    assert result["status"] == "recorded"
    assert ResearchStore(tmp_path / "research").read()["directions"]
    assert list(tmp_path.rglob("research-transaction.json")) == []


def test_publish_if_changed_skips_identical_projection_and_heals_changes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert publish_if_changed(workspace, "records/value.json", b"one") is True
    first = (workspace / "records/value.json").stat().st_mtime_ns
    assert publish_if_changed(workspace, "records/value.json", b"one") is False
    assert (workspace / "records/value.json").stat().st_mtime_ns == first
    assert publish_if_changed(workspace, "records/value.json", b"two") is True
    assert (workspace / "records/value.json").read_bytes() == b"two"


def test_existing_trajectory_session_resumes_on_first_invocation(tmp_path):
    resume_flags = []
    opened_session_ids = []

    class Native:
        supports_resume = True

        def __init__(self):
            self.environment = {"ATREX_TELEMETRY_ATTEMPT_ID": "invocation"}

        def invoke(self, _prompt, **kwargs):
            resume_flags.append(kwargs.get("resume"))
            return NativeInvocation(
                exit_status=0,
                timed_out=False,
                session_id="trajectory-session",
                stdout="done",
                stderr="",
                terminal_usage=TokenUsage(7, 0, 0, 0, 7, "exact"),
                events=(),
                capabilities=AgentRuntimeCapabilities(False, False, False),
                observation_errors=(),
                resume_usage_qualified=False,
            )

        def close(self):
            pass

    class Agent:
        def open_session(self, _runtime_id, _workspace, **kwargs):
            opened_session_ids.append(kwargs.get("session_id"))
            return Native()

    result = LongSessionRunner(
        agent=Agent(),
        agent_cli="claude",
        usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0,
    ).run(
        tmp_path,
        "Continue the next Episode.",
        episode_timeout_seconds=60,
        session_id="trajectory-session",
        resume_session=True,
        initial_purpose="trajectory_continuation",
    )

    assert result.session_id == "trajectory-session"
    assert opened_session_ids == ["trajectory-session"]
    assert resume_flags == [True]
    assert [row.purpose for row in result.invocations] == ["trajectory_continuation"]


def _compaction_connection_loss(session_id):
    return json.dumps({
        "type": "assistant",
        "session_id": session_id,
        "parent_tool_use_id": None,
        "isApiErrorMessage": True,
        "message": {
            "content": [{
                "type": "text",
                "text": (
                    "Prompt is too long · automatic compaction failed: "
                    "API Error: Connection lost mid-response. "
                    "The response above may be incomplete."
                ),
            }],
        },
    })


class _SequenceNative:
    supports_resume = True

    def __init__(self, agent):
        self.agent = agent
        self.environment = {"ATREX_TELEMETRY_ATTEMPT_ID": "invocation"}

    def invoke(self, prompt, **kwargs):
        self.agent.prompts.append(prompt)
        self.agent.resume_flags.append(bool(kwargs.get("resume")))
        return self.agent.invocations.pop(0)

    def close(self):
        pass


class _SequenceAgent:
    def __init__(self, invocations):
        self.invocations = list(invocations)
        self.prompts = []
        self.resume_flags = []
        self.opened_sessions = []

    def open_session(self, _runtime_id, _workspace, **kwargs):
        self.opened_sessions.append(kwargs.get("session_id"))
        return _SequenceNative(self)


def _invocation(exit_status, session_id, tokens, stdout="done"):
    return NativeInvocation(
        exit_status=exit_status,
        timed_out=False,
        session_id=session_id,
        stdout=stdout,
        stderr="",
        terminal_usage=TokenUsage(tokens, 0, 0, 0, tokens, "exact"),
        events=(),
        capabilities=AgentRuntimeCapabilities(False, False, False),
        observation_errors=(),
        resume_usage_qualified=False,
    )


def test_compaction_connection_loss_resumes_same_claude_session(
    tmp_path, monkeypatch
):
    agent = _SequenceAgent([
        _invocation(
            1,
            "session-1",
            11,
            _compaction_connection_loss("session-1"),
        ),
        _invocation(0, "session-1", 7),
    ])
    result = LongSessionRunner(
        agent=agent,
        agent_cli="claude",
        usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0,
    ).run(
        tmp_path,
        "Optimize.",
        episode_timeout_seconds=60,
    )

    assert result.exit_status == 0
    assert result.resume_count == 1
    assert agent.resume_flags == [False, True]
    assert "transient provider interruption" in agent.prompts[1]
    assert [row.purpose for row in result.invocations] == [
        "optimization", "session_recovery",
    ]


def test_repeated_compaction_loss_uses_one_fresh_recovery_session(
    tmp_path, monkeypatch
):
    failure = _compaction_connection_loss("session-1")
    agent = _SequenceAgent([
        _invocation(1, "session-1", 11, failure),
        _invocation(1, "session-1", 13, failure),
        _invocation(0, "session-2", 7),
    ])
    result = LongSessionRunner(
        agent=agent,
        agent_cli="claude",
        usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0,
    ).run(
        tmp_path,
        "Optimize.",
        episode_timeout_seconds=60,
    )

    assert result.exit_status == 0
    assert result.resume_count == 1
    assert agent.resume_flags == [False, True, False]
    assert len(agent.opened_sessions) == 2
    assert "Recover the SAME Episode" in agent.prompts[2]
    assert "do not reset or repeat completed measurements" in agent.prompts[2]
    assert [row.purpose for row in result.invocations] == [
        "optimization", "session_recovery", "compaction_recovery",
    ]


def test_stable_session_home_root_survives_episode_workspace_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "aka.light.agent.sandbox.sandbox_executable", lambda _environment: "/usr/bin/bwrap"
    )
    first = tmp_path / "episodes" / "e0001" / "workspace"
    second = tmp_path / "episodes" / "e0002" / "workspace"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    home_root = tmp_path / "trajectory-session-homes"
    environment = {
        "HOME": str(tmp_path / "operator-home"),
        SESSION_HOME_ROOT_ENV: str(home_root),
    }

    first_environment = prepare_agent_environment(
        first, dict(environment), "trajectory-session"
    )
    second_environment = prepare_agent_environment(
        second, dict(environment), "trajectory-session"
    )
    other_environment = prepare_agent_environment(
        second, dict(environment), "different-session"
    )

    assert first_environment["HOME"] == second_environment["HOME"]
    assert second_environment["HOME"] != other_environment["HOME"]
    assert Path(first_environment["HOME"]).is_relative_to(home_root.resolve())


def test_supervisor_state_persists_trajectory_session():
    state = SupervisorState(
        session_id="trajectory-session", session_backend="claude"
    )
    restored = SupervisorState.from_dict(state.as_dict())
    assert restored.session_id == "trajectory-session"
    assert restored.session_backend == "claude"


def _finalized_journal(path: Path, *, episode: int, suffix: str, record_id: str) -> tuple[str, str]:
    direction_id = "direction_" + suffix * 32
    experiment_id = "experiment_" + suffix * 32
    initialize_journal(
        path,
        episode=episode,
        base_commit="a" * 40,
        branch="episode",
        memory_version=1,
    )
    value = load_journal(path)
    value.update(state="pivot", finalized_at="2026-10-30T00:00:00+00:00")
    value["direction_events"] = [
        {
            "direction_event_id": "directionevent_" + suffix * 32,
            "direction_id": direction_id,
            "action": "propose",
            "name": f"direction-{suffix}",
            "change_scope": "major_redesign",
            # Historical sealed Journals can still contain the retired field.
            "hypothesis": "A shared optimization objective",
            "plan": ["implement", "measure"],
            "stop_conditions": [],
            "recorded_at": "2026-10-30T00:00:00+00:00",
            "supporting_experiment_ids": [],
        },
        {
            "direction_event_id": "directionevent_" + ("c" if suffix != "c" else "d") * 32,
            "direction_id": direction_id,
            "action": "start",
            "analysis": "Started",
            "recorded_at": "2026-10-30T00:01:00+00:00",
            "supporting_experiment_ids": [],
        },
        {
            "direction_event_id": "directionevent_" + ("e" if suffix != "e" else "f") * 32,
            "direction_id": direction_id,
            "action": "defer",
            "analysis": "Deferred after measurement",
            "recorded_at": "2026-10-30T00:02:00+00:00",
            "supporting_experiment_ids": [experiment_id],
        },
    ]
    value["experiments"] = [
        {
            "experiment_id": experiment_id,
            "sequence": 1,
            "recorded_at": "2026-10-30T00:01:30+00:00",
            "name": f"experiment-{suffix}",
            "hypothesis": "Measure the candidate",
            "change": "Changed the schedule",
            "gateway_record_ids": [record_id],
            "analysis": "The measurement is reusable evidence",
            "direction_id": direction_id,
            "claim_review_quality": "partially_supported",
        }
    ]
    private_write(path, value)
    return direction_id, experiment_id




def test_imported_measurement_is_readable_but_not_a_local_task_cache(tmp_path):
    source = MeasurementStore(tmp_path / "source")
    destination = MeasurementStore(tmp_path / "destination")
    record_id, kernel_id = _measurement(source, b"def kernel():\n    return 1\n", "source")

    imported = destination.import_record(source, record_id)

    assert imported == source.read(record_id)
    assert destination.read_kernel(kernel_id) == source.read_kernel(kernel_id)
    assert destination.kernel_records(kernel_id)[0]["gateway_record_id"] == record_id
    assert not (destination.root / "tasks").exists()


def test_same_source_under_distinct_git_commits_has_distinct_task_identity(tmp_path):
    store = MeasurementStore(tmp_path / "measurements")
    source = b"def kernel():\n    return 1\n"
    records = []
    for kernel_id in ("1" * 40, "2" * 40):
        identity = store.kernel(source, kernel_id)
        request = {
            "operation": "evaluate",
            "options": {},
            "inputs": {"kernel.py": hashlib.sha256(source).hexdigest()},
            "kernel_ids": {"candidate": kernel_id},
        }
        with store.reserve(request, {"candidate": identity}) as task:
            records.append(
                task.finish(
                    {"exit_code": 0, "stdout": "", "stderr": ""},
                    cacheable=True,
                )["gateway_record_id"]
            )
    assert records[0] != records[1]


def test_shared_research_validates_observational_submitted_winners(tmp_path):
    journals = tmp_path / "journals"
    other_accepted = "5" * 40
    winner = {
        "kernel_commit_id": "1" * 40,
        "gateway_record_id": "gateway-" + "2" * 32,
        "latency_us_geomean": 4.25,
        "trajectory": 2,
        "lineage": "T2",
    }
    private_write(
        shared_research_path(journals),
        {
            "schema_version": 1,
            "revision": "revision-winner",
            "through_epoch": 3,
            "directions": {},
            "experiments": [],
            "research_queue": [],
            "accepted_submission_kernel_ids": {
                "T2": [winner["kernel_commit_id"], other_accepted],
            },
            "lineage_winners": {"T2": winner},
            "global_best": winner,
        },
    )

    shared = _shared_research(journals)
    assert shared["accepted_submission_kernel_ids"] == {
        "T2": sorted([winner["kernel_commit_id"], other_accepted]),
    }
    assert shared["lineage_winners"] == {"T2": winner}
    assert shared["global_best"] == winner


def test_progress_labels_only_accepted_submissions_as_winners():
    accepted = "1" * 40
    measured_only = "2" * 40
    winner = {
        "kernel_commit_id": accepted,
        "gateway_record_id": "gateway-" + "3" * 32,
        "latency_us_geomean": 4.25,
        "lineage": "T1",
    }
    progress = _progress_markdown(
        {
            "current_lineage": "T1",
            "lineages": {
                "T1": {
                    "working_kernel_commit_id": measured_only,
                    # A migrated pre-change projection must not affect rendering.
                    "lineage_best_kernel_commit_id": measured_only,
                }
            },
            "submitted_winners": {
                "lineages": {"T1": winner},
                "global_best": winner,
            },
            "pareto_frontier": {
                "best_by_shape": {
                    "shape-a": {
                        "kernel_commit_id": accepted,
                        "gateway_record_id": "gateway-" + "4" * 32,
                        "latency_us": 3.5,
                    }
                },
                "shape_count": 1,
                "evaluation_count": 1,
                "winner_kernel_commit_ids": [accepted],
                "best_full_evaluation": {
                    "kernel_commit_id": accepted,
                    "gateway_record_id": "gateway-" + "4" * 32,
                    "latency_us_geomean": 3.5,
                },
            },
        },
        modules=frozenset(),
    ).decode()

    assert "**Global best accepted submission" not in progress
    assert f"**Best accepted submission for `T1`:** `{accepted}`" in progress
    assert f"| `T1` | `{measured_only}` | `{accepted}` |" in progress
    assert "Best verified lineage Kernel" not in progress
    assert "Best measured passing Kernel" not in progress
    assert f"| `shape-a` | 3.5 | `{accepted}` |" in progress
    assert "Accepted-submission per-Shape Pareto frontier" in progress


def test_progress_shows_global_best_only_with_multiple_lineages():
    winner = {
        "kernel_commit_id": "1" * 40,
        "gateway_record_id": "gateway-" + "2" * 32,
        "latency_us_geomean": 4.25,
        "lineage": "T1",
    }
    progress = _progress_markdown(
        {
            "current_lineage": "T1",
            "lineages": {"T1": {}, "T2": {}},
            "submitted_winners": {
                "lineages": {"T1": winner},
                "global_best": winner,
            },
        },
        modules=frozenset(),
    ).decode()

    assert f"**Global best accepted submission commit ID:** `{'1' * 40}`" in progress
    assert f"**Best accepted submission for `T1`:** `{'1' * 40}`" in progress


def test_shared_research_is_visible_without_agent_facing_origin_fields(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "scratch").mkdir(parents=True)
    journals = tmp_path / "journals"
    shared_direction = "direction_" + "1" * 32
    shared_experiment = "experiment_" + "2" * 32
    shared_direction_value = {
        "direction_id": shared_direction,
        "name": "shared direction",
        "change_scope": "major_redesign",
        "hypothesis": "shared hypothesis",
        "plan": ["test it"],
        "stop_conditions": [],
        "status": "proposed",
        "analysis": "shared analysis",
        "supporting_experiment_ids": [],
        "associated_experiment_ids": [shared_experiment],
    }
    shared_experiment_value = {
        "experiment_id": shared_experiment,
        "recorded_by": "T2",
        "sequence": 1,
        "name": "shared experiment",
        "objective": "shared optimization objective",
        "change": "shared change",
        "gateway_record_ids": [],
        "analysis": "shared result",
        "direction_id": shared_direction,
    }
    private_write(
        shared_research_path(journals),
        {
            "schema_version": 1,
            "revision": "revision-1",
            "through_epoch": 1,
            "directions": {
                shared_direction: shared_direction_value,
            },
            "experiments": [shared_experiment_value],
            "research_queue": [shared_direction],
        },
    )
    current = journals / "episodes" / "e00000002" / "journal.json"
    initialize_journal(
        current,
        episode=2,
        base_commit="b" * 40,
        branch="episode",
        memory_version=1,
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=current.parent,
        research_synchronization="asynchronous",
    )
    catalog = ResearchStore(service.research_catalog_root / "events")
    catalog.observe_direction(
        {
            **shared_direction_value,
            "owner_lineage": "T2",
            "access": "local_writable",
        },
        origin={"epoch": 1, "trajectory": 2, "episode": 1},
    )
    catalog.observe_experiment(
        {
            **shared_experiment_value,
            "owner_lineage": "T2",
            "access": "local_writable",
        },
        origin={"epoch": 1, "trajectory": 2, "episode": 1},
    )
    publish_catalog_shared_research(
        catalog_root=service.research_catalog_root,
        research_root=journals,
        trajectory=1,
        synchronization="asynchronous",
    )

    service.publish_research_files()
    loaded = json.loads(
        (
            workspace / "research-library" / "directions" / shared_direction
            / "v1" / "direction.json"
        ).read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (
            workspace / "research-library" / "experiments" / shared_experiment
            / "v1" / "experiment.json"
        ).read_text(encoding="utf-8")
    )

    assert loaded["direction_id"] == shared_direction
    assert "associated_experiment_ids" not in loaded
    assert loaded["access"] == "peer_read_only"
    assert experiment["experiment_id"] == shared_experiment
    assert experiment["owner_lineage"] == "T2"
    assert experiment["access"] == "peer_read_only"
    assert "origin" not in loaded and "shared" not in loaded
    assert "origin" not in experiment and "shared" not in experiment
    with pytest.raises(AgentRequestError) as rejected:
        service.execute(
            {
                "operation": "direction_update",
                "request": {
                    "action": "start",
                    "direction_id": shared_direction,
                    "analysis": "try to mutate shared state",
                },
            }
        )
    assert rejected.value.response["error"]["code"] == "shared_direction_read_only"
    with pytest.raises(AgentRequestError) as peer_mutation:
        service.execute({
            "operation": "direction_prioritize",
            "request": {
                "direction_id": shared_direction,
                "reason": "try to rank peer work locally",
            },
        })
    assert peer_mutation.value.response["error"]["code"] == "shared_direction_read_only"
    with pytest.raises(AgentRequestError) as experiment_rejected:
        service.execute(
            {
                "operation": "experiment_record",
                "request": {
                    "name": "local measurement on shared direction",
                    "hypothesis": "must be rejected before evidence lookup",
                    "change": "none",
                    "gateway_record_ids": [],
                    "analysis": "none",
                    "direction_id": shared_direction,
                },
            }
        )
    assert experiment_rejected.value.response["error"]["code"] == "shared_direction_read_only"
    progress = (workspace / "PROGRESS.md").read_text(encoding="utf-8")
    assert "access `peer_read_only`" in progress
    assert "### Local work queue\n\n_None._" in progress
    assert "### Shared direction pool" in progress
    assert "shared rank #1" in progress
    assert "absorb compatible peer mechanisms and evidence" in progress
    assert "`derived_from_direction_ids`" in progress
    assert "Experiment files declare `owner_lineage` and `access`" in progress

    proposed = service.execute(
        {
            "operation": "direction_update",
            "request": {
                "action": "propose",
                "name": "local continuation",
                "change_scope": "major_redesign",
                "objective": "refine the shared optimization objective",
                "applicability": "The shared Direction's target regimes.",
                "expected_benefit": "Retain the peer gain while refining its bottleneck.",
                "plan": ["implement locally"],
                "relationship": "refinement",
                "derived_from_direction_ids": [shared_direction],
                "derived_from_experiment_ids": [],
            },
        }
    )
    assert proposed["direction_id"] != shared_direction


def test_search_epoch_seals_other_trajectory_research_and_measurements(tmp_path):
    driver = EpisodeSearchDriver.__new__(EpisodeSearchDriver)
    controller = tmp_path / "controller"
    controller.mkdir()
    campaign_root = supervisor_campaign_root(controller)
    initialize_campaign_root(
        campaign_root,
        name="pooled-sharing-test",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(),
    )
    driver.campaign = SimpleNamespace(workspace=controller)
    driver.root = campaign_root
    driver.children = {}
    identities = {}
    records = {}
    local_only_records = {}
    passing_full_records = {}
    kernels = {}
    store = MeasurementStore(supervisor_measurement_root(controller))
    for number, suffix in ((1, "1"), (2, "2")):
        workspace = campaign_root / "lineages" / f"t{number:04d}" / "agent-workspace"
        _install(workspace, "def run():\n    return None\n")
        AgentGit(workspace).prepare()
        child = SimpleNamespace(workspace=workspace)
        driver.children[number] = child
        runtime_root = supervisor_campaign_root(workspace)
        record_id, kernel_id = _measurement(
            store, f"def kernel_{number}():\n    return {number}\n".encode(), f"t{number}"
        )
        local_only_record_id, _ = _measurement(
            store,
            f"def local_probe_{number}():\n    return {number}\n".encode(),
            f"t{number}-local-only",
        )
        passing_full_record_id, _ = _passing_full_measurement(
            store,
            f"def full_candidate_{number}():\n    return {number}\n".encode(),
            f"t{number}-passing-full",
            {"shape_a": float(number), "shape_b": float(number + 1)},
        )
        journal = runtime_root / "journals" / "episodes" / "e00000001" / "journal.json"
        identities[number] = _finalized_journal(
            journal, episode=1, suffix=suffix, record_id=record_id
        )
        research = ResearchStore(runtime_root / "research")
        research.checkpoint_lineage(
            {
                "gateway_record_ids": [
                    record_id,
                    local_only_record_id,
                    passing_full_record_id,
                ],
            },
            origin={"epoch": 1, "trajectory": number, "episode": 1},
        )
        research.observe_direction(
            {
                "direction_id": identities[number][0],
                "name": f"direction-{suffix}",
                "change_scope": "major_redesign",
                "objective": "A shared optimization objective",
                "applicability": "Long-context shapes.",
                "expected_benefit": "Reduce full-workload latency.",
                "status": "deferred",
                "plan": ["implement", "measure"],
                "supporting_experiment_ids": [identities[number][1]],
            },
            origin={"epoch": 1, "trajectory": number, "episode": 1},
        )
        research.observe_experiment(
            {
                "experiment_id": identities[number][1],
                "name": f"experiment-{suffix}",
                "intent": "Measure the bounded candidate.",
                "interpretation": "The measured result supports the scoped decision.",
                "gateway_record_ids": [record_id],
                "direction_id": identities[number][0],
            },
            origin={"epoch": 1, "trajectory": number, "episode": 1},
        )
        if number == 2:
            research.prioritize(
                identities[number][0],
                reason="continue the highest-impact redesign first",
                origin={"epoch": 1, "trajectory": number, "episode": 1},
            )
        records[number], kernels[number] = record_id, kernel_id
        local_only_records[number] = local_only_record_id
        passing_full_records[number] = passing_full_record_id

    jobs = [
        {"epoch": 1, "trajectory": 1, "episode": 1, "status": "completed"},
        {"epoch": 1, "trajectory": 2, "episode": 1, "status": "completed"},
    ]
    driver.seal_epoch(1, jobs)
    driver.seal_epoch(1, jobs)

    for number in (1, 2):
        other = 2 if number == 1 else 1
        runtime_root = supervisor_campaign_root(driver.children[number].workspace)
        shared = json.loads(shared_research_path(runtime_root / "research").read_text())
        assert set(shared["directions"]) == {identities[other][0]}
        shared_direction = shared["directions"][identities[other][0]]
        assert shared_direction["owner_lineage"] == f"T{other}"
        assert shared_direction["objective"] == "A shared optimization objective"
        assert "hypothesis" not in shared_direction
        assert "hypothesis_status" not in shared_direction
        assert shared["research_queue"] == sorted(
            [identities[1][0], identities[2][0]]
        )
        assert {row["experiment_id"] for row in shared["experiments"]} == {identities[other][1]}
        assert shared["experiments"][0]["owner_lineage"] == f"T{other}"
        assert "origin" not in shared_direction
        assert not (runtime_root / "measurements" / "identity.key").exists()
        assert set(store.record_ids()) == (
            set(records.values())
            | set(local_only_records.values())
            | set(passing_full_records.values())
        )
        assert store.read_kernel(kernels[other]).startswith(b"def kernel_")
        assert local_only_records[other] not in shared["gateway_record_ids"]
        assert records[other] in shared["gateway_record_ids"]
        assert records[number] not in shared["gateway_record_ids"]
        assert passing_full_records[other] in shared["gateway_record_ids"]
        assert passing_full_records[number] in shared["gateway_record_ids"]

    publication = read_state(publication_path(driver.root, 1))
    assert publication["synchronization"] == "synchronous"
    assert publication["watermark"]["trajectory_episodes"] == {
        "T1": 1,
        "T2": 1,
    }
    assert [source["trajectory"] for source in publication["sources"]] == [
        "T1",
        "T2",
    ]
    for number in (1, 2):
        git_publication = publication_source_root(campaign_root, 1, number)
        assert (git_publication / "history.bundle").is_file()
        receipt = json.loads((git_publication / "receipt.json").read_text())
        assert receipt["trajectory"] == number
        assert receipt["publication"] == 1
        assert "epoch" not in receipt
    assert not (campaign_root / "peer-git").exists()
    receipt = publication["evidence"]
    assert set(receipt["gateway_record_ids"]) == (
        set(records.values()) | set(passing_full_records.values())
    )
    assert not (driver.root / "epochs").exists()
    assert not (driver.root / "async-publications").exists()
    assert driver.epoch_sealed(1)


def test_pooled_private_state_uses_one_campaign_root_with_lineage_namespaces(tmp_path):
    controller = tmp_path / "controller"
    controller.mkdir()
    campaign_root = legacy_supervisor_campaign_root(controller)
    trajectory = campaign_root / "search" / "trajectories" / "t0002"
    (trajectory / ".git").mkdir(parents=True)
    (trajectory / ".git" / "atrex-search-origin.json").write_text(
        json.dumps({
            "bootstrap": "bootstrap",
            "trajectory": 2,
            "campaign": str(controller),
        }),
        encoding="utf-8",
    )

    assert supervisor_campaign_base_root(trajectory) == campaign_root
    assert supervisor_campaign_root(trajectory) == campaign_root / "lineages" / "t0002"
    assert supervisor_measurement_root(trajectory) == campaign_root / "measurements"


def test_direct_campaign_root_contains_private_state_and_only_agent_workspace_has_git(
    tmp_path,
):
    campaign = initialize_campaign_root(
        tmp_path / "campaign-A",
        name="operator-a",
        launch_composition=_test_launch_composition(profile="light-sandbox"),
        source=_test_source_binding(),
    )
    agent = campaign / "lineages" / "t0001" / "agent-workspace"
    agent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(agent)], check=True)

    assert not (campaign / ".git").exists()
    assert not (campaign / ".atrex-supervisor-runtime").exists()
    assert supervisor_campaign_base_root(campaign) == campaign
    assert supervisor_campaign_base_root(agent) == campaign
    assert supervisor_campaign_root(agent) == campaign / "lineages" / "t0001"
    assert supervisor_measurement_root(agent) == campaign / "measurements"
    assert CampaignStore(agent).root == campaign / "lineages" / "t0001" / "segments"


def test_pooled_private_state_rejects_origin_marker_path_redirection(tmp_path):
    controller = tmp_path / "controller"
    controller.mkdir()
    workspace = tmp_path / "not-a-campaign-trajectory"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "atrex-search-origin.json").write_text(
        json.dumps({
            "bootstrap": "bootstrap",
            "trajectory": 1,
            "campaign": str(controller),
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        supervisor_campaign_root(workspace)


def test_pooled_recovery_moves_legacy_trajectory_state_into_campaign_lineage(tmp_path):
    controller = tmp_path / "controller"
    controller.mkdir()
    campaign_root = legacy_supervisor_campaign_root(controller)
    trajectory = campaign_root / "search" / "trajectories" / "t0001"
    (trajectory / ".git").mkdir(parents=True)
    (trajectory / ".git" / "atrex-search-origin.json").write_text(
        json.dumps({
            "bootstrap": "bootstrap",
            "trajectory": 1,
            "campaign": str(controller),
        }),
        encoding="utf-8",
    )
    legacy = legacy_supervisor_campaign_root(trajectory)
    (legacy / "journals").mkdir(parents=True)
    (legacy / "journals" / "legacy.json").write_text("{}\n", encoding="utf-8")
    driver = EpisodeSearchDriver.__new__(EpisodeSearchDriver)

    driver._migrate_trajectory_private_root(trajectory)

    destination = campaign_root / "lineages" / "t0001"
    assert not legacy.exists()
    assert (destination / "journals" / "legacy.json").is_file()


def test_shared_direction_queue_averages_normalized_trajectory_ranks():
    directions = {
        name: {"direction_id": name, "status": "proposed"}
        for name in ("a", "b", "c", "new")
    }
    states = {
        1: {"queue": ["a", "b", "new", "c"]},
        2: {"queue": ["b", "a", "c"]},
        3: {"queue": ["b", "c", "a"]},
    }

    # b wins the average vote.  The new Direction is not treated as last by
    # trajectories that had no opportunity to observe it; it keeps T1's rank.
    assert _aggregate_direction_queue(states, directions) == ["b", "a", "new", "c"]


def test_shared_direction_queue_ties_are_deterministic_and_ignore_closed_items():
    directions = {
        "a": {"direction_id": "a", "status": "proposed"},
        "b": {"direction_id": "b", "status": "in_progress"},
        "done": {"direction_id": "done", "status": "completed"},
    }
    states = {
        1: {"queue": ["a", "b", "done"]},
        2: {"queue": ["b", "a"]},
    }

    assert _aggregate_direction_queue(states, directions) == ["a", "b"]


def test_shared_direction_queue_uses_peer_pool_votes_without_overwriting_local_order():
    directions = {
        name: {"direction_id": name, "status": "proposed"}
        for name in ("a", "b", "c")
    }
    states = {
        1: {"queue": ["a"], "shared_queue": ["b", "c"]},
        2: {"queue": ["b"], "shared_queue": ["a", "c"]},
        3: {"queue": ["c"], "shared_queue": ["b", "a"]},
    }

    assert _aggregate_direction_queue(states, directions) == ["a", "b", "c"]
    assert states[1]["queue"] == ["a"]
    assert states[2]["queue"] == ["b"]
    assert states[3]["queue"] == ["c"]


def test_shared_direction_queue_preserves_unvoted_inherited_order():
    directions = {
        name: {"direction_id": name, "status": "proposed"}
        for name in ("a", "b", "c")
    }
    states = {
        1: {"queue": []},
        2: {"queue": []},
        3: {"queue": []},
    }

    assert _aggregate_direction_queue(
        states,
        directions,
        inherited_queue=["c", "a", "b"],
    ) == ["c", "a", "b"]


def test_epoch_merge_keeps_owner_latest_revision_regardless_of_review_quality():
    direction_id = "direction_" + "a" * 32
    owner_latest = {
        "direction_id": direction_id,
        "revision": 2,
        "claim_review_quality": "partially_supported",
    }
    peer_cached = {
        "direction_id": direction_id,
        "revision": 1,
        "claim_review_quality": "supported",
    }

    merged, origins = EpisodeSearchDriver._merge_named(
        {1: {direction_id: owner_latest}, 2: {direction_id: peer_cached}},
        {direction_id: {"epoch": 1, "trajectory": 1, "episode": 1}},
        {1: {direction_id: 2}, 2: {}},
        [
            {"epoch": 2, "trajectory": 1, "episode": 2},
            {"epoch": 2, "trajectory": 2, "episode": 2},
        ],
        "Direction",
    )

    assert merged[direction_id] == owner_latest
    assert origins[direction_id]["trajectory"] == 1


def test_multiple_active_directions_require_explicit_experiment_binding(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journals = tmp_path / "scope" / "journals"
    evidence = journals / "episodes" / "e00000001"
    evidence.mkdir(parents=True)
    initialize_journal(
        evidence / "journal.json", episode=1, base_commit="a" * 40, branch="work"
    )
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=evidence,
        research_root=tmp_path / "research",
    )
    directions = []
    for name in ("incremental", "redesign"):
        direction_id = service.execute({
            "operation": "direction_update",
            "request": {
                "action": "propose",
                "name": name,
                "change_scope": "incremental" if name == "incremental" else "major_redesign",
                "objective": f"Optimize the {name} path across applicable regimes.",
                "applicability": "The target workload.",
                "expected_benefit": "Resolve a consequential optimization choice.",
                "plan": ["Run one bounded probe."],
            },
        })["direction_id"]
        service.execute({
            "operation": "direction_update",
            "request": {
                "action": "start",
                "direction_id": direction_id,
                "analysis": "Start bounded work.",
            },
        })
        directions.append(direction_id)

    service.execute({
        "operation": "direction_prioritize",
        "request": {
            "direction_id": directions[0],
            "before_direction_id": directions[1],
            "reason": "Prefer the cheaper discriminator first.",
        },
    })
    with pytest.raises(AgentRequestError) as ambiguous:
        service.execute({
            "operation": "experiment_record",
            "request": {
                "name": "Unbound observation",
                "intent": "Record a result without choosing concurrent work.",
                "interpretation": "The intended Direction is ambiguous.",
            },
        })
    assert ambiguous.value.response["error"]["code"] == "direction_ambiguous"
    assert ambiguous.value.response["error"]["active_direction_ids"] == directions

    recorded = service.execute({
        "operation": "experiment_record",
        "request": {
            "direction_id": directions[0],
            "name": "Bound observation",
            "intent": "Record evidence for the incremental path.",
            "interpretation": "The observation belongs only to that Direction.",
        },
    })
    projected = json.loads((workspace / recorded["archive_path"]).read_text())
    assert projected["direction_id"] == directions[0]






def test_pooled_search_seals_peer_evidence_without_promotion(tmp_path):
    calls = []

    class Driver:
        def bootstrap(self):
            return {"latency_us_geomean": 10.0}

        def initialize(self, seed, trajectories):
            calls.append(("initialize", trajectories))

        def prepare_epoch(self, epoch):
            calls.append(("prepare", epoch))

        def run_episode(self, epoch, trajectory, episode):
            return {
                "epoch": epoch,
                "trajectory": trajectory,
                "episode": episode,
                "agent_head_commit_id": "a" * 40,
                "best_submitted_evaluation": None,
                "tokens": 1,
            }

        def seal_epoch(self, epoch, jobs):
            calls.append(("seal", epoch))

        def epoch_sealed(self, epoch):
            return ("seal", epoch) in calls

        def tokens(self):
            return 1

    root = tmp_path / "campaign-A"
    result = PooledSearch(
        root,
        SearchOptions(epochs=1, trajectories=1, episodes_per_trajectory=1, max_parallel_episodes=1),
        Driver(),
    ).run()

    assert result == "search: completed Epoch budget"
    assert calls == [("initialize", 1), ("prepare", 1), ("seal", 1)]
    coordinator = read_state(root / COORDINATOR_STATE)
    assert set(coordinator) == {
        "schema_version",
        "epoch",
        "active",
        "completed_epochs",
        "global_best",
    }
    assert (root / "coordinator.lock").is_file()
    assert not (root / "search").exists()
    assert read_state(lineage_state_path(root, 1))["completed_episodes"] == 1
    publication = read_state(publication_path(root, 1))
    assert publication["synchronization"] == "synchronous"
    assert publication["scheduler"]["phase"] == "completed"
    assert not (root / "epochs").exists()
    assert not (root / "async-publications").exists()


def test_search_synchronization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="synchronous or asynchronous"):
        SearchOptions(synchronization="eventually")


def test_search_role_specialization_is_optional_and_defaults_disabled():
    assert SearchOptions().role_specialization is False
    assert SearchOptions(role_specialization=True).role_specialization is True
    with pytest.raises(TypeError, match="role_specialization must be a boolean"):
        SearchOptions(role_specialization=1)


@pytest.mark.parametrize("enabled", [True, False])
def test_episode_search_role_specialization_prompt_switch(tmp_path, enabled):
    base = SimpleNamespace(
        workspace=tmp_path,
        platform="L20D",
        arch="sm_103",
        notes="none",
        research_records=False,
        research_modules=(),
        search_coordinate={
            "epoch": 1,
            "trajectory": 1,
            "episode": 1,
            "role_specialization": enabled,
        },
        module_options=SimpleNamespace(goal_episodes=False),
        specify_dsl=True,
        framework="Cuda",
        dsl_context="- DSL: `Cuda`",
        plugin_directive=lambda stage: "",
        tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
    )
    prompt = LongHorizonCampaign(base)._prompt(
        episode=1,
        segment=CandidateHandle(1, "a" * 40, "work", tmp_path),
    )
    assert "Your trajectory: T1. Search location: Epoch 1, Episode 1" in prompt
    assert ("T1 role: bounded exploitation" in prompt) is enabled


@pytest.mark.parametrize(
    ("synchronization", "expected"),
    [
        ("synchronous", "at synchronous Epoch barriers"),
        ("asynchronous", "after each completed audit segment"),
    ],
)
def test_episode_search_prompt_describes_actual_publication_boundary(
    tmp_path, synchronization, expected
):
    base = SimpleNamespace(
        workspace=tmp_path,
        platform="L20D",
        arch="sm_103a",
        notes="none",
        research_records=False,
        research_modules=(),
        search_coordinate={
            "epoch": 1,
            "trajectory": 1,
            "episode": 1,
            "synchronization": synchronization,
            "role_specialization": False,
        },
        module_options=SimpleNamespace(goal_episodes=False),
        specify_dsl=True,
        framework="Cuda",
        dsl_context="- DSL: `Cuda`",
        plugin_directive=lambda stage: "",
        tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
    )
    prompt = LongHorizonCampaign(base)._prompt(
        episode=1,
        segment=CandidateHandle(1, "a" * 40, "work", tmp_path),
    )
    assert expected in prompt


def test_asynchronous_search_advances_without_waiting_for_slow_peer(tmp_path):
    seed = {"latency_us_geomean": 10.0, "name": "seed"}
    first_trajectory_advanced = Event()
    slow_peer_released = Event()
    lock = Lock()
    calls = []
    token_count = 0

    class Driver:
        def bootstrap(self):
            return seed

        def initialize(self, initial, trajectories):
            assert initial == seed and trajectories == 2

        def prepare_trajectory(
            self,
            epoch,
            trajectory,
            episode,
            shared_publication,
        ):
            assert type(shared_publication) is int
            with lock:
                calls.append(("prepare", trajectory, episode))

        def run_episode(self, epoch, trajectory, episode):
            nonlocal token_count
            if (trajectory, episode) == (2, 1):
                assert first_trajectory_advanced.wait(2)
                slow_peer_released.set()
            elif (trajectory, episode) == (1, 2):
                first_trajectory_advanced.set()
            with lock:
                calls.append(("run", trajectory, episode))
                token_count += 1
            return {
                "epoch": epoch,
                "trajectory": trajectory,
                "episode": episode,
                "agent_head_commit_id": (str(trajectory) * 40)[:40],
                "best_submitted_evaluation": None,
                "tokens": 1,
                "session_id": f"session-t{trajectory}",
            }

        def seal_epoch(self, publication, jobs):
            assert jobs
            assert all(job["status"] == "completed" for job in jobs)
            with lock:
                calls.append(("seal", publication))

        def tokens(self):
            return token_count

        def interrupt(self):
            slow_peer_released.set()

    root = tmp_path / "search"
    result = PooledSearch(
        root,
        SearchOptions(
            epochs=2,
            trajectories=2,
            episodes_per_trajectory=1,
            max_parallel_episodes=2,
            synchronization="asynchronous",
        ),
        Driver(),
    ).run()

    assert result == "search: completed asynchronous Episode budget"
    assert slow_peer_released.is_set()
    assert calls.index(("run", 1, 2)) < calls.index(("run", 2, 1))
    state = read_state(root / COORDINATOR_STATE)
    assert "lineages" not in state
    assert read_state(lineage_state_path(root, 1))["completed_episodes"] == 2
    assert read_state(lineage_state_path(root, 2))["completed_episodes"] == 2
    assert state["completed_epochs"] == 2
    assert state["asynchronous"]["latest_publication"] == 4
    publications = sorted((root / "publications").glob("p*/publication.json"))
    assert [path.parent.name for path in publications] == [
        "p00000001",
        "p00000002",
        "p00000003",
        "p00000004",
    ]
    assert all(
        read_state(path)["synchronization"] == "asynchronous"
        for path in publications
    )
    assert not (root / "epochs").exists()
    assert not (root / "async-publications").exists()


def test_wiki_query_missing_input_file_is_a_structured_agent_error(tmp_path):
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.config = SimpleNamespace(wiki_enabled=True)
    runtime.wiki_provider = SimpleNamespace(cli_root=tmp_path / "wiki")
    capability = SimpleNamespace(workspace=tmp_path)

    with pytest.raises(AgentRequestError) as raised:
        runtime._wiki_command(
            capability,
            "query_nl",
            ["--file", "wiki-question.txt"],
            tmp_path / "staged",
            {},
        )

    response = raised.value.response
    assert response["error"]["code"] == "file_not_found"
    assert response["error"]["requested_path"] == "wiki-question.txt"
    assert response["issues"] == [
        {
            "path": "--file",
            "code": "file_not_found",
            "message": "Wiki query input file does not exist: wiki-question.txt",
        }
    ]
    assert "pass the query text directly" in response["error"]["next_action"]


def _source(text: str) -> bytes:
    policy, files = default_tree({"kernel.py": text.encode()})
    return pack(files, policy)


def _install(workspace: Path, text: str) -> bytes:
    source = _source(text)
    install(workspace, source)
    return source


def _controller_repository(workspace: Path, text: str) -> Path:
    _install(workspace, text)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Supervisor"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "supervisor@localhost"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"], cwd=workspace, check=True
    )
    return workspace


def test_episode_prompt_uses_reachable_agent_comparison_base(tmp_path):
    controller = _controller_repository(
        tmp_path / "controller", "def run():\n    return 1\n"
    )
    controller_base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=controller,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    agent_workspace = tmp_path / "agent-workspace"
    _install(agent_workspace, "def run():\n    return 1\n")
    agent_repository = AgentGit(agent_workspace, tmp_path / "agent.bundle")
    agent_repository.prepare()
    agent_base = agent_repository.head()
    assert agent_base is not None and agent_base != controller_base
    base = SimpleNamespace(
        workspace=controller,
        platform="L20D",
        arch="sm_103a",
        notes="none",
        research_records=False,
        research_modules=(),
        module_options=SimpleNamespace(goal_episodes=False),
        specify_dsl=True,
        framework="Cuda",
        dsl_context="- DSL: `Cuda`",
        plugin_directive=lambda _stage: "",
        reference_knowledge_prompt=lambda: "",
        tool_runtime=SimpleNamespace(pause_agent_during_requests=True),
        acceptance_measurement_context=lambda _workspace, _source: {
            "baseline_kernel_id": agent_base,
            "measurement_repetitions": 1,
        },
    )

    prompt = LongHorizonCampaign(base)._prompt(
        episode=1,
        segment=CandidateHandle(1, controller_base, "work", controller),
        agent_workspace=agent_workspace,
    )

    assert f"Agent-visible comparison base: `{agent_base}`" in prompt
    assert f"Agent-visible comparison base: `{controller_base}`" not in prompt
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{agent_base}^{{commit}}"],
        cwd=agent_workspace,
        capture_output=True,
        check=False,
    ).returncode == 0




def test_pooled_trajectories_restore_one_canonical_agent_v0(tmp_path):
    controller = _controller_repository(
        tmp_path / "controller", "def run():\n    return 0\n"
    )
    controller_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=controller,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    v0_source = snapshot(controller)

    driver = EpisodeSearchDriver.__new__(EpisodeSearchDriver)
    driver.campaign = SimpleNamespace(workspace=controller)
    driver.root = tmp_path / "search"
    driver.children = {}
    record = {
        "quality_gate": {"result": "PASS"},
        "performance": {"latency_us_geomean": 1.0},
    }
    bootstrap = driver._kernel(
        controller, controller_commit, record, trajectory=0, episode=0
    )
    canonical = driver._canonical_agent_root(bootstrap)

    v0_commit = canonical["agent_v0_commit_id"]
    campaign_root = supervisor_campaign_base_root(controller)
    assert driver._canonical_agent_bundle == campaign_root / "bootstrap" / "agent-v0.bundle"
    assert driver._canonical_agent_receipt == campaign_root / "bootstrap" / "agent-v0.json"
    assert driver._publication_source_root(2, 3) == publication_source_root(
        campaign_root, 3, 2
    )
    assert driver._publication_import_receipt(2, 3) == publication_import_path(
        campaign_root, 3, 2
    )
    assert not (campaign_root / "peer-git").exists()
    assert canonical == {
        "schema_version": 2,
        "source_sha256": hashlib.sha256(v0_source).hexdigest(),
        "agent_v0_commit_id": controller_commit,
    }
    assert driver._canonical_agent_root(bootstrap) == canonical

    restored_heads = []
    for number in (1, 2, 3):
        child_runtime_root = tmp_path / f"trajectory-{number}"
        child_runtime_root.mkdir()
        child_agent_workspace = child_runtime_root / "agent-workspace"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-hardlinks",
                str(driver._canonical_agent_bundle),
                str(child_agent_workspace),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", "-B", "work", v0_commit],
            cwd=child_agent_workspace,
            check=True,
        )
        subprocess.run(
            ["git", "remote", "remove", "origin"],
            cwd=child_agent_workspace,
            check=True,
        )
        source_state = read_source_state(controller)
        assert source_state is not None
        file_system, contract = source_state
        bind_source_state(
            child_agent_workspace,
            file_system=file_system,
            contract=contract,
        )
        child_agent = AgentGit(child_agent_workspace)
        child_agent.prepare()
        restored_heads.append(child_agent.head())
        assert child_agent.commit_reachable(v0_commit)
        assert snapshot(child_agent_workspace) == v0_source
        assert not (child_runtime_root / "adaptive-state").exists()

    assert restored_heads == [v0_commit, v0_commit, v0_commit]


def test_research_store_rebuilds_queue_ignores_legacy_progress_and_adoption(tmp_path):
    store = ResearchStore(tmp_path / "research")
    first = "direction_" + "1" * 32
    second = "direction_" + "2" * 32
    peer = "direction_" + "3" * 32
    for direction_id, name in ((first, "local"), (second, "redesign")):
        store.observe_direction(
            {
                "direction_id": direction_id,
                "name": name,
                "change_scope": "major_redesign" if direction_id == second else "local",
                "status": "proposed",
                "plan": ["probe", "measure"],
                # Older projections may feed these derived fields back into
                # an observed Direction. Runtime must retire both fields.
                "completed_plan_steps": [0],
                "next_step": "measure",
            },
            origin={"trajectory": 1},
        )
    store.prioritize(second, before_direction_id=first, reason="higher impact")
    # Historical logs may still contain the retired direction.progressed event.
    store.append(
        "direction.progressed",
        {"direction_id": second, "plan_step": 0},
        origin={"trajectory": 1},
    )
    store.observe_direction(
        {
            "direction_id": second,
            "name": "redesign",
            "change_scope": "major_redesign",
            "status": "in_progress",
            "owner": "T1",
            "plan": ["probe", "measure"],
        },
        origin={"trajectory": 1},
    )
    store.observe_direction(
        {
            "direction_id": second,
            "name": "redesign",
            "change_scope": "major_redesign",
            "status": "deferred",
            "owner": "T1",
            "plan": ["probe", "measure"],
        },
        origin={"trajectory": 1},
    )
    # Reopening may produce exactly the same visible in-progress projection as
    # an earlier lease.  Historical deduplication must not leave it deferred.
    store.observe_direction(
        {
            "direction_id": second,
            "name": "redesign",
            "change_scope": "major_redesign",
            "status": "in_progress",
            "owner": "T1",
            "plan": ["probe", "measure"],
        },
        origin={"trajectory": 1},
    )
    # Historical logs may still contain the retired kernel.adopted event.
    # Rebuilds must remain compatible even though Agents can no longer create it.
    store.append(
        "kernel.adopted",
        {
            "kernel_id": "3" * 40,
            "source": {"git_ref": "refs/atrex/shared/T2/e0001/head"},
            "reason": "historical event",
        },
        origin={"trajectory": 1},
    )
    store.observe_direction(
        {
            "direction_id": peer,
            "name": "peer redesign",
            "change_scope": "major_redesign",
            "status": "proposed",
            "plan": ["probe peer idea"],
            "proposed_by": "T2",
            "access": "peer_read_only",
        },
        origin={"trajectory": 2},
    )
    store.synchronize_shared_queue([peer, first, second], origin={"epoch": 2})
    store.prioritize(second, before_direction_id=first, reason="local durable reorder")
    store.synchronize_shared_queue([peer, first, second], origin={"epoch": 3})

    rebuilt = ResearchStore(tmp_path / "research").read()
    assert rebuilt["queue"] == [second, first]
    assert rebuilt["shared_queue"] == [peer]
    assert "completed_plan_steps" not in rebuilt["directions"][second]
    assert "next_step" not in rebuilt["directions"][second]
    assert rebuilt["lineages"]["T1"]["working_kernel_id"] == "3" * 40
    assert rebuilt["lineages"]["T1"]["adopted_from"] == {
        "git_ref": "refs/atrex/shared/T2/e0001/head",
        "kernel_commit_id": "3" * 40,
    }
    board = store.board(trajectory=1)
    assert [item["direction_id"] for item in board["open_queue"]] == [second, first]
    assert board["open_queue"][0]["status"] == "in_progress(T1)"
    assert "next_step" not in board["open_queue"][0]
    assert [item["direction_id"] for item in board["shared_direction_pool"]] == [peer]
    assert board["lineages"]["T1"]["working_kernel_commit_id"] == "3" * 40
    assert "working_kernel_id" not in board["lineages"]["T1"]


def test_research_store_terminal_projection_excludes_newer_live_episode(tmp_path):
    store = ResearchStore(tmp_path / "research")
    direction_id = "direction_" + "4" * 32
    store.observe_direction(
        {
            "direction_id": direction_id,
            "name": "terminal direction",
            "change_scope": "major_redesign",
            "status": "in_progress",
            "plan": ["probe", "measure"],
        },
        origin={"trajectory": 1, "episode": 1},
    )
    # Replay remains compatible with historical progress events even though
    # Agents can no longer create them.
    store.append(
        "direction.progressed",
        {"direction_id": direction_id, "plan_step": 0},
        origin={"trajectory": 1, "episode": 2},
    )

    terminal = store.read_through_episode(1)

    assert "completed_plan_steps" not in terminal["directions"][direction_id]
    assert "next_step" not in terminal["directions"][direction_id]
    assert len(store.events_through_episode(1)) == 1
    assert "completed_plan_steps" not in store.read()["directions"][direction_id]
    assert "next_step" not in store.read()["directions"][direction_id]


def test_research_store_tracks_multiple_active_directions_per_lineage(tmp_path):
    store = ResearchStore(tmp_path / "research")
    first = "direction_" + "a" * 32
    second = "direction_" + "b" * 32
    for direction_id in (first, second):
        store.observe_direction(
            {
                "direction_id": direction_id,
                "name": direction_id,
                "status": "in_progress",
                "owner": "T1",
                "plan": ["probe"],
            },
            origin={"trajectory": 1},
        )

    state = store.read()
    assert state["lineages"]["T1"]["active_direction_ids"] == [first, second]
    assert state["lineages"]["T1"]["active_direction_id"] == first

    store.observe_direction(
        {
            "direction_id": first,
            "name": first,
            "status": "completed",
            "owner": "T1",
            "plan": ["probe"],
        },
        origin={"trajectory": 1},
    )
    state = store.read()
    assert state["lineages"]["T1"]["active_direction_ids"] == [second]
    assert state["lineages"]["T1"]["active_direction_id"] == second


def test_research_store_uses_snapshot_tail_and_skips_unchanged_lineage(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path / "research")
    values = {
        "working_kernel_id": "a" * 40,
        # Historical callers may still replay these retired fields. They must
        # not reintroduce an unsubmitted Evaluate as a lineage winner.
        "lineage_best_kernel_id": "b" * 40,
        "lineage_best_latency_us": 1.0,
    }
    first = store.checkpoint_lineage(values, origin={"trajectory": 1, "episode": 1})
    second = store.checkpoint_lineage(values, origin={"trajectory": 1, "episode": 1})
    assert second["sequence"] == first["sequence"]

    # A valid compact snapshot serves ordinary reads without replaying every
    # historical event. Tail validation still happens inside read().
    monkeypatch.setattr(store, "_events", lambda: (_ for _ in ()).throw(
        AssertionError("complete event replay should not run")
    ))
    lineage = store.read()["lineages"]["T1"]
    assert lineage["working_kernel_id"] == "a" * 40
    assert "lineage_best_kernel_id" not in lineage
    assert "lineage_best_latency_us" not in lineage


def test_research_store_rotates_bounded_segments_and_rebuilds(tmp_path, monkeypatch):
    import aka.light.research.store as research_store

    monkeypatch.setattr(research_store, "MAX_EVENT_SEGMENT_BYTES", 512)
    store = ResearchStore(tmp_path / "research")
    for index in range(12):
        store.checkpoint_lineage(
            {"working_kernel_id": f"{index:040x}"},
            origin={"trajectory": 1, "episode": index + 1},
        )
    assert list((tmp_path / "research").glob("events-*.jsonl"))

    # Deleting the convenience snapshot exercises recovery across rotated
    # segments and the active tail without losing sequence history.
    (tmp_path / "research" / "snapshot.json").unlink()
    rebuilt = ResearchStore(tmp_path / "research").read()
    assert rebuilt["sequence"] == 12
    assert rebuilt["lineages"]["T1"]["working_kernel_id"] == f"{11:040x}"


@pytest.mark.parametrize(
    ("modules", "included", "excluded"),
    [
        (
            ("directions",),
            "research-library/directions/<direction_id>/vN/direction.json",
            "research-library/experiments/<experiment_id>/vN/experiment.json",
        ),
        (
            ("experiments",),
            "research-library/experiments/<experiment_id>/vN/experiment.json",
            "research-library/directions/<direction_id>/vN/direction.json",
        ),
        (
            (),
            "records/<gateway_record_id>.json",
            "research-library/experiments/<experiment_id>/vN/experiment.json",
        ),
    ],
)
def test_journal_repair_hint_mentions_only_enabled_modules(modules, included, excluded):
    hint = _journal_evidence_hint(modules)
    assert included in hint
    assert excluded not in hint


def test_agent_git_persists_work_and_installs_runtime_source_refs(tmp_path):
    workspace = tmp_path / "workspace"
    source = _install(workspace, "def run():\n    return 1\n")
    bundle = tmp_path / "state" / "agent.bundle"
    repo = AgentGit(workspace, bundle)
    repo.prepare()
    first = repo.head()
    assert first
    (workspace / "impl").mkdir()
    (workspace / "impl" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    persisted = repo.persist()
    assert persisted == first
    assert "impl/helper.py" in subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], cwd=workspace,
        text=True, capture_output=True, check=True,
    ).stdout
    subprocess.run(["git", "add", "impl/helper.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Agent: preserve helper"],
        cwd=workspace,
        check=True,
    )
    persisted = repo.persist()
    assert persisted != first
    runtime_commit = repo.install_runtime_ref(
        "refs/tags/atrex/incumbent", source,
        message="Runtime: incumbent kernel-test", commit_id=first,
    )
    assert subprocess.run(
        ["git", "rev-parse", "refs/tags/atrex/incumbent"], cwd=workspace,
        text=True, capture_output=True, check=True,
    ).stdout.strip() == runtime_commit
    with pytest.raises(ValueError, match="existing local or imported peer"):
        repo.install_runtime_ref(
            "refs/tags/atrex/historical-private",
            source,
            message="must not expose a synthetic historical commit",
            commit_id="f" * 40,
        )

    restored = tmp_path / "restored"
    restored_source = _install(restored, "def run():\n    return 1\n")
    assert restored_source == source
    resumed = AgentGit(restored, bundle)
    resumed.prepare()
    assert "Runtime: preserve Episode WIP checkpoint" not in subprocess.run(
        ["git", "log", "--all", "--format=%s"], cwd=restored,
        text=True, capture_output=True, check=True,
    ).stdout
    assert (restored / "impl" / "helper.py").read_text() == "VALUE = 2\n"


def test_agent_git_kernel_discovery_skips_arbitrary_non_kernel_commits(tmp_path):
    workspace = tmp_path / "workspace"
    source = _install(workspace, "def run():\n    return 1\n")
    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    repository.prepare()
    kernel_commit = repository.head()

    subprocess.run(["git", "rm", "-q", "kernel.py"], cwd=workspace, check=True)
    (workspace / "design-notes.txt").write_text(
        "This engineering checkpoint is intentionally not a Kernel snapshot.\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "design-notes.txt"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "checkpoint incomplete redesign"],
        cwd=workspace,
        check=True,
    )

    assert repository.find_kernel_commit(source) == kernel_commit


def test_agent_git_kernel_discovery_returns_none_for_non_kernel_history(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "work"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Atrex Agent"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "agent@localhost"],
        cwd=workspace,
        check=True,
    )
    (workspace / "design-notes.txt").write_text(
        "No Kernel has been created yet.\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "design-notes.txt"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "record design"], cwd=workspace, check=True
    )

    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    assert repository.find_kernel_commit(b"def run():\n    return 1\n") is None


def test_agent_git_head_is_kernel_id_only_for_clean_exact_source(tmp_path):
    workspace = tmp_path / "workspace"
    source = _install(workspace, "def run():\n    return 1\n")
    repository = AgentGit(workspace, tmp_path / "agent.bundle")
    repository.prepare()

    first = repository.kernel_commit(source)
    assert len(first) in {40, 64}

    direction = (
        workspace / "research-library" / "directions"
        / ("direction_" + "1" * 32) / "v0"
    )
    direction.mkdir(parents=True)
    (direction / "direction.json").write_text("{}\n", encoding="utf-8")
    experiment = (
        workspace / "research-library" / "experiments"
        / ("experiment_" + "2" * 32) / "v1"
    )
    experiment.mkdir(parents=True)
    (experiment / "experiment.json").write_text("{}\n", encoding="utf-8")
    (workspace / "PROGRESS.md").write_text("# Runtime projection\n", encoding="utf-8")
    assert repository.kernel_commit(source) == first

    (workspace / "kernel.py").write_text(
        "def run():\n    return 2\n", encoding="utf-8"
    )
    with pytest.raises(DirtyKernelWorktree) as dirty:
        repository.kernel_commit()
    assert "kernel.py" in dirty.value.paths

    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "candidate"], cwd=workspace, check=True
    )
    second = repository.kernel_commit()
    assert second != first
    assert repository.kernel_commit(snapshot(workspace)) == second




def test_agent_workspace_does_not_project_controller_memory(tmp_path):
    controller = _controller_repository(
        tmp_path / "controller", "def run():\n    return 1\n"
    )
    controller_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=controller,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    (controller / "memory").mkdir()
    (controller / "memory" / "v1.json").write_text(
        json.dumps({
            "version": "v1",
            "git_commit_hash": controller_commit,
            "long_horizon": {"candidate_commit": controller_commit},
        }),
        encoding="utf-8",
    )
    agent_root = tmp_path / "runtime" / "agent-workspace"
    (agent_root / "memory").mkdir(parents=True)
    (agent_root / "memory" / "v1.json").write_text(
        "stale projected summary\n", encoding="utf-8"
    )
    view = EpisodeWorkspace(
        controller,
        tmp_path / "evidence",
        persistent_root=agent_root,
    )

    view.prepare()

    agent_commit = view.agent_git.head()
    assert agent_commit and agent_commit != controller_commit
    assert not (agent_root / "memory" / "v1.json").exists()
    retired = list(
        (tmp_path / "runtime" / "retired-agent-memory").glob("*/memory/v1.json")
    )
    assert len(retired) == 1
    assert retired[0].read_text(encoding="utf-8") == "stale projected summary\n"


def test_agent_workspace_is_stable_across_episode_controller_worktrees(tmp_path):
    first_controller = _controller_repository(
        tmp_path / "controller-1", "def run():\n    return 1\n"
    )
    second_controller = tmp_path / "controller-2"
    subprocess.run(
        ["git", "clone", "-q", str(first_controller), str(second_controller)],
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Supervisor"],
        cwd=second_controller,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "supervisor@localhost"],
        cwd=second_controller,
        check=True,
    )
    (second_controller / "kernel.py").write_text(
        "def run():\n    return 2\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "kernel.py"], cwd=second_controller, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "new incumbent"],
        cwd=second_controller,
        check=True,
    )
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "agent-workspace-state.json").write_text(
        json.dumps({"workspace_mode": "persistent", "kernel_digest": "legacy"})
        + "\n",
        encoding="utf-8",
    )
    agent_root = runtime_root / "agent-workspace"

    first = EpisodeWorkspace(
        first_controller,
        tmp_path / "evidence-1",
        persistent_root=agent_root,
    )
    assert first.prepare() == agent_root
    assert "records/<gateway_record_id>.json" in (
        agent_root / "WORKSPACE.md"
    ).read_text(encoding="utf-8")
    assert not (agent_root / "input" / "evidence").exists()
    assert "conversation.jsonl" not in (
        agent_root / "WORKSPACE.md"
    ).read_text(encoding="utf-8")
    (agent_root / "input" / "evidence").mkdir(parents=True)
    (agent_root / "input" / "evidence" / "conversation.jsonl").write_text(
        "legacy derived view\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "switch", "-c", "experiment/persistent"],
        cwd=agent_root,
        check=True,
        text=True,
        capture_output=True,
    )
    (agent_root / "kernel.py").write_text(
        "def run():\n    return 7\n", encoding="utf-8"
    )
    (agent_root / "scratch" / "probe.txt").write_text(
        "keep across episodes\n", encoding="utf-8"
    )
    (agent_root / "tools" / "custom_probe.py").write_text(
        "print('persistent tool')\n", encoding="utf-8"
    )
    custom_skill = agent_root / "skills" / "custom-optimization"
    custom_skill.mkdir()
    (custom_skill / "SKILL.md").write_text(
        "# Persistent optimization skill\n", encoding="utf-8"
    )
    first.publish()

    second = EpisodeWorkspace(
        second_controller,
        tmp_path / "evidence-2",
        persistent_root=agent_root,
    )
    assert second.prepare() == agent_root
    assert not (agent_root / "input" / "evidence").exists()
    assert (agent_root / "kernel.py").read_text(encoding="utf-8").endswith(
        "return 7\n"
    )
    assert (agent_root / "scratch" / "probe.txt").read_text(encoding="utf-8") == (
        "keep across episodes\n"
    )
    assert (agent_root / "tools" / "custom_probe.py").read_text(
        encoding="utf-8"
    ) == "print('persistent tool')\n"
    assert (custom_skill / "SKILL.md").read_text(encoding="utf-8") == (
        "# Persistent optimization skill\n"
    )
    assert subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=agent_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == "experiment/persistent"
    assert not (runtime_root / "agent-workspace-state.json").exists()
    assert not (runtime_root / "state.json").exists()
    assert not (runtime_root / "adaptive-state").exists()


def test_runtime_reuses_agent_repository_without_episode_worktree(tmp_path):
    campaign_root = initialize_campaign_root(
        tmp_path / "campaign",
        name="persistent-agent-repository",
        launch_composition=_test_launch_composition(),
        source=_test_source_binding(b"def run():\n    return 1\n"),
    )
    agent = _controller_repository(
        campaign_root / "lineages" / "t0001" / "agent-workspace",
        "def run():\n    return 1\n",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=agent,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    lineage_state = campaign_root / "lineages" / "t0001" / "state.json"
    merge_lineage_state(
        lineage_state,
        1,
        completed_episodes=0,
        session_id="persistent-session",
        lineage_winner={"kernel_commit_id": head, "latency_us_geomean": 5.0},
    )
    runtime = SupervisorRuntime(
        RuntimeConfig(hardware="L20D", workspace=agent, wiki_enabled=False),
        workspace_provider=ManagedCandidateWorkspace(),
    )
    try:
        runtime_configuration = campaign_runtime_configuration(agent)
        assert runtime_configuration is not None
        assert runtime_configuration["hardware"] == "L20D"
        assert runtime_configuration["transport"] == "agate_sdk"
        assert runtime_configuration["source_checkout"]["git_commit_id"]
        manifest = json.loads((campaign_root / "campaign.json").read_text())
        assert manifest["source"] == _test_source_binding(
            b"def run():\n    return 1\n"
        )
        assert not (
            campaign_root
            / "lineages"
            / "t0001"
            / "runtime-config.json"
        ).exists()
        first = runtime.register_episode(
            agent,
            episode=1,
            base_commit=head,
            branch="work",
            memory_version=1,
            managed_agent_workspace=True,
        )
        assert first == agent.resolve()
        merged = json.loads(lineage_state.read_text())
        assert merged["session_id"] == "persistent-session"
        assert merged["lineage_winner"]["kernel_commit_id"] == head
        assert merged["current_episode"] == 1
        (agent / "kernel.py").write_text(
            "def run():\n    return 2\n", encoding="utf-8"
        )
        (agent / "scratch" / "probe.txt").write_text(
            "unfinished probe\n", encoding="utf-8"
        )

        second = runtime.register_episode(
            agent,
            episode=2,
            base_commit=head,
            branch="work",
            memory_version=2,
            managed_agent_workspace=True,
        )

        assert second == first
        assert (agent / "kernel.py").read_text(encoding="utf-8").endswith(
            "return 2\n"
        )
        assert (agent / "scratch" / "probe.txt").read_text(
            encoding="utf-8"
        ) == "unfinished probe\n"
        assert not (campaign_root / "lineages" / "t0001" / "segment-worktrees").exists()
        assert not (campaign_root / "search" / "trajectories").exists()
        assert runtime.journals[agent.resolve()].service.episode == 2
        assert not (campaign_root / "lineages" / "t0001" / "journals").exists()
        lineage_state = json.loads(
            (campaign_root / "lineages" / "t0001" / "state.json").read_text()
        )
        assert lineage_state["current_episode"] == 2
        assert lineage_state["segment_status"] == "active"
        assert not {
            "agent_git",
            "kernel_digest",
            "controller_kernel_digest",
            "workspace_mode",
            "research_records",
            "research_modules",
            "runtime_plugins",
        }.intersection(lineage_state)
    finally:
        runtime.close()


def test_legacy_episode_submissions_migrate_once_into_campaign_store(tmp_path):
    legacy = tmp_path / "legacy-journals"
    journal = legacy / "episodes" / "e00000003" / "journal.json"
    journal.parent.mkdir(parents=True)
    accepted_commit = "a" * 40
    rejected_commit = "b" * 40
    journal.write_text(
        json.dumps(
            {
                "kernel_submissions": [
                    {
                        "kernel_commit_id": accepted_commit,
                        "accepted": True,
                        "submitted_at": "2026-11-01T00:00:00+00:00",
                        "evaluation_evidence": {
                            "status": "passed",
                            "gateway_record_id": "gateway-" + "1" * 32,
                        },
                    },
                    {
                        "kernel_commit_id": rejected_commit,
                        "accepted": False,
                        "submitted_at": "2026-11-01T00:01:00+00:00",
                        "diagnostics": ["final-source review rejected the commit"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    store = tmp_path / "submissions"

    assert migrate_legacy_journals(legacy, store, lineage="T2") == 2
    first = submissions(store, lineage="T2")
    assert len(first) == 2
    assert accepted_submission_kernel_ids(store, lineage="T2") == {
        accepted_commit
    }

    # Re-registering or moving a Campaign must not duplicate immutable audit
    # rows. The migration identity does not depend on the absolute path.
    assert migrate_legacy_journals(legacy, store, lineage="T2") == 0
    moved = tmp_path / "moved-journals"
    moved.mkdir()
    moved_journal = moved / "episodes" / "e00000003" / "journal.json"
    moved_journal.parent.mkdir(parents=True)
    moved_journal.write_bytes(journal.read_bytes())
    assert migrate_legacy_journals(moved, store, lineage="T2") == 0
    assert submissions(store, lineage="T2") == first


def test_runtime_rebinds_stable_agent_workspace_to_next_episode_journal(tmp_path):
    campaign = _controller_repository(
        tmp_path / "campaign", "def run():\n    return 1\n"
    )
    first_controller = _controller_repository(
        tmp_path / "controller-1", "def run():\n    return 1\n"
    )
    second_controller = tmp_path / "controller-2"
    subprocess.run(
        ["git", "clone", "-q", str(first_controller), str(second_controller)],
        check=True,
    )
    runtime = SupervisorRuntime(
        RuntimeConfig(hardware="L20D", workspace=campaign, wiki_enabled=False),
        workspace_provider=ManagedCandidateWorkspace(),
    )
    try:
        first = runtime.register_episode(
            first_controller,
            episode=1,
            base_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=first_controller,
                text=True, capture_output=True, check=True,
            ).stdout.strip(),
            branch="episode-1",
            memory_version=1,
            managed_agent_workspace=True,
        )
        first_baseline = runtime.baseline_kernel_commit(first_controller)
        assert runtime.baseline_kernel_commit(first) == first_baseline
        assert first_baseline == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=first,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip() == first_baseline
        assert subprocess.run(
            ["git", "show-ref", "--verify", "refs/atrex/baselines/e00000001"],
            cwd=first,
            text=True,
            capture_output=True,
        ).returncode != 0
        commits_before = subprocess.run(
            ["git", "rev-list", "--all", "--count"], cwd=first,
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        assert runtime.measurements.read_kernel(first_baseline) == (
            from_git(first_controller, subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=first_controller,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip())
        )
        campaign = SimpleNamespace(
            start_runtime=lambda: None,
            _supervisor_runtime=runtime,
        )
        acceptance = Campaign.acceptance_measurement_context(
            campaign,
            first,
            runtime.measurements.read_kernel(first_baseline),
        )
        assert acceptance["baseline_kernel_id"] == first_baseline
        second = runtime.register_episode(
            second_controller,
            episode=2,
            base_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=second_controller,
                text=True, capture_output=True, check=True,
            ).stdout.strip(),
            branch="episode-2",
            memory_version=2,
            baseline_kernel_commit_id=first_baseline,
            managed_agent_workspace=True,
        )
        assert second == first
        assert runtime.baseline_kernel_commit(second) == first_baseline
        assert subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=second,
            text=True, capture_output=True, check=True,
        ).stdout.strip() == first_baseline
        assert subprocess.run(
            ["git", "rev-list", "--all", "--count"], cwd=second,
            text=True, capture_output=True, check=True,
        ).stdout.strip() == commits_before
        assert runtime.journals[second].service.episode == 2
        assert runtime.journals[first_controller.resolve()].service.episode == 1
        assert not (runtime.audit_root / "journals").exists()
    finally:
        runtime.close()


def test_agent_git_restores_unmerged_branches_and_active_checkout(tmp_path):
    workspace = tmp_path / "workspace"
    source = _install(workspace, "def run():\n    return 1\n")
    bundle = tmp_path / "state" / "agent.bundle"
    repo = AgentGit(workspace, bundle)
    repo.prepare()
    subprocess.run(
        ["git", "switch", "-c", "experiment/ksplit4"], cwd=workspace, check=True,
        text=True, capture_output=True,
    )
    (workspace / "impl").mkdir()
    (workspace / "impl" / "ksplit.py").write_text(
        "KSPLIT = 4\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "impl/ksplit.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Agent: try KSPLIT=4"],
        cwd=workspace,
        check=True,
    )
    repo.persist()

    restored = tmp_path / "restored"
    restored_source = _install(restored, "def run():\n    return 1\n")
    assert restored_source == source
    resumed = AgentGit(restored, bundle)
    resumed.prepare()

    assert subprocess.run(
        ["git", "branch", "--show-current"], cwd=restored,
        text=True, capture_output=True, check=True,
    ).stdout.strip() == "experiment/ksplit4"
    assert subprocess.run(
        ["git", "show-ref", "--verify", "refs/heads/work"], cwd=restored,
        text=True, capture_output=True, check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "show-ref", "--verify", "refs/heads/experiment/ksplit4"], cwd=restored,
        text=True, capture_output=True, check=True,
    ).returncode == 0
    assert (restored / "impl" / "ksplit.py").read_text() == "KSPLIT = 4\n"


def test_epoch_git_exchange_filters_paths_and_preserves_peer_ref(tmp_path):
    source_workspace = tmp_path / "source"
    _install(source_workspace, "def run():\n    return 1\n")
    source = AgentGit(source_workspace, tmp_path / "source.bundle")
    source.prepare()
    (source_workspace / "impl").mkdir()
    (source_workspace / "impl" / "peer.py").write_text("KSPLIT = 2\n", encoding="utf-8")
    (source_workspace / "scratch").mkdir()
    (source_workspace / "scratch" / "secret.txt").write_text("private\n", encoding="utf-8")
    subprocess.run(["git", "add", "impl/peer.py"], cwd=source_workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Agent: peer candidate"],
        cwd=source_workspace,
        check=True,
    )
    source.persist()

    publication_root = tmp_path / "publications" / "p00000001"
    export = publication_root / "history.bundle"
    receipt = export_publication(
        source_workspace,
        export,
        trajectory=2,
        publication=1,
        audit_path=publication_root / "receipt.json",
    )
    assert receipt["commit_count"] >= 1
    assert b"scratch/secret.txt" not in export.read_bytes()
    assert subprocess.run(
        ["git", "show-ref", "--verify", "refs/remotes/atrex/T2"],
        cwd=source_workspace,
        capture_output=True,
        check=False,
    ).returncode != 0

    destination = tmp_path / "destination"
    _install(destination, "def run():\n    return 0\n")
    local = AgentGit(destination)
    local.prepare()
    imported = import_exports_into_repository(
        destination,
        [export],
        notes={
            "refs/remotes/atrex/T2": {
                "measurement_status": "measured", "kernel_id": "4" * 40
            }
        },
        audit_path=tmp_path / "import.json",
    )
    assert imported["imports"] == ["T2@p00000001"]
    assert imported["shared_refs"] == ["refs/remotes/atrex/T2"]
    assert imported["commit_mappings"] == [
        {
            "trajectory": 2,
            "publication": 1,
            "source_ref": "refs/remotes/atrex/T2",
            "original_agent_commit": receipt["original_agent_commit"],
            "imported_commit": imported["commit_mappings"][0]["imported_commit"],
        }
    ]
    assert imported["commit_mappings"][0]["imported_commit"]
    assert imported["commit_mappings"][0]["imported_commit"] == receipt["original_agent_commit"]

    # A trajectory-stable repository imports peer refs in place; it does not
    # need a recovery bundle or newly projected Episode workspace.
    assert subprocess.run(
        ["git", "show-ref", "--verify", "refs/remotes/atrex/T2"],
        cwd=destination,
        text=True,
        capture_output=True,
        check=True,
    ).returncode == 0
    remote_branches = subprocess.run(
        ["git", "branch", "-r", "--format=%(refname:short)"],
        cwd=destination,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert "atrex/T2" in remote_branches

    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "refs/remotes/atrex/T2"],
        cwd=destination, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert tree == [
        ".gitignore",
        "impl/peer.py",
        "kernel.py",
    ]
    messages = subprocess.run(
        ["git", "log", "--format=%s", "refs/remotes/atrex/T2"],
        cwd=destination, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    original_messages = subprocess.run(
        ["git", "log", "--format=%s", receipt["original_agent_commit"]],
        cwd=source_workspace,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert messages == original_messages


def test_epoch_git_exchange_keeps_stable_ref_on_published_head(tmp_path):
    source_workspace = tmp_path / "source"
    _install(source_workspace, "def run():\n    return 1\n")
    source = AgentGit(source_workspace, tmp_path / "source.bundle")
    source.prepare()
    winner = source.head()
    assert winner

    (source_workspace / "kernel.py").write_text(
        "def run():\n    return 2\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "kernel.py"], cwd=source_workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Agent: continue after submitted winner"],
        cwd=source_workspace,
        check=True,
    )
    head = source.head()
    assert head and head != winner

    publication_root = tmp_path / "publications" / "p00000001"
    export = publication_root / "history.bundle"
    receipt = export_publication(
        source_workspace,
        export,
        trajectory=2,
        publication=1,
        lineage_winner_commit_id=winner,
        audit_path=publication_root / "receipt.json",
    )
    assert receipt["source_ref"] == "refs/remotes/atrex/T2"
    assert receipt["original_agent_commit"] == head
    assert receipt["lineage_winner_commit_id"] == winner

    destination = tmp_path / "destination"
    _install(destination, "def run():\n    return 0\n")
    local = AgentGit(destination)
    local.prepare()
    imported = import_exports_into_repository(
        destination,
        [export],
        audit_path=tmp_path / "import.json",
    )
    mapping = imported["commit_mappings"][0]
    assert mapping["source_ref"] == "refs/remotes/atrex/T2"
    assert mapping["imported_commit"] == head
    assert subprocess.run(
        ["git", "rev-parse", "refs/remotes/atrex/T2"],
        cwd=destination,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == head
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{winner}^{{commit}}"],
        cwd=destination,
        capture_output=True,
        check=False,
    ).returncode == 0
    assert local.head() != winner

    local_head = local.head()
    (source_workspace / "kernel.py").write_text(
        "def run():\n    return 3\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "kernel.py"], cwd=source_workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Agent: publish newer HEAD"],
        cwd=source_workspace,
        check=True,
    )
    newer_head = source.head()
    newer_root = tmp_path / "publications" / "p00000002"
    newer_export = newer_root / "history.bundle"
    export_publication(
        source_workspace,
        newer_export,
        trajectory=2,
        publication=2,
        lineage_winner_commit_id=winner,
        audit_path=newer_root / "receipt.json",
    )
    import_exports_into_repository(
        destination,
        [newer_export],
        audit_path=tmp_path / "import-2.json",
    )
    assert subprocess.run(
        ["git", "rev-parse", "atrex/T2"],
        cwd=destination,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == newer_head
    assert local.head() == local_head


def test_peer_in_progress_direction_does_not_block_local_lease(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "scratch").mkdir(parents=True)
    journals = tmp_path / "journals"
    peer = "direction_" + "1" * 32
    private_write(
        shared_research_path(journals),
        {
            "schema_version": 1,
            "revision": "r1",
            "through_epoch": 1,
            "directions": {
                peer: {
                    "direction_id": peer,
                    "name": "peer major redesign",
                    "change_scope": "major_redesign",
                    "objective": "peer optimization objective",
                    "plan": ["continue"],
                    "stop_conditions": [],
                    "status": "in_progress",
                    "owner": "T2",
                    "supporting_experiment_ids": [],
                    "associated_experiment_ids": [],
                }
            },
            "experiments": [],
        },
    )
    journal = journals / "episodes" / "e00000002" / "journal.json"
    initialize_journal(journal, episode=2, base_commit="a" * 40, branch="episode")
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=journals,
        evidence_root=journal.parent,
        research_root=tmp_path / "research",
        origin={"epoch": 2, "trajectory": 1, "episode": 2},
    )
    local = service.execute({
        "operation": "direction_update",
        "request": {
            "action": "propose", "name": "local probe", "change_scope": "local",
            "objective": "local optimization objective",
            "applicability": "The local trajectory's target workload.",
            "expected_benefit": "Resolve the local performance uncertainty.",
            "plan": ["measure"],
        },
    })["direction_id"]
    assert service.execute({
        "operation": "direction_update",
        "request": {"action": "start", "direction_id": local, "analysis": "independent lease"},
    })["status"] == "recorded"


def test_epoch_projection_replays_local_active_direction_beside_peer_active_direction(
    tmp_path,
):
    journals = tmp_path / "journals"
    peer = "direction_" + "1" * 32
    local = "direction_" + "2" * 32
    private_write(
        shared_research_path(journals),
        {
            "schema_version": 1,
            "revision": "epoch-1",
            "through_epoch": 1,
            "directions": {
                peer: {
                    "direction_id": peer,
                    "name": "peer redesign",
                    "change_scope": "major_redesign",
                    "objective": "peer optimization objective",
                    "plan": ["continue peer work"],
                    "stop_conditions": [],
                    "status": "in_progress",
                    "owner": "T2",
                    "supporting_experiment_ids": [],
                    "associated_experiment_ids": [],
                }
            },
            "experiments": [],
            "research_queue": [peer],
        },
    )

    previous_path = journals / "episodes" / "e00000001" / "journal.json"
    previous = initialize_journal(
        previous_path,
        episode=1,
        base_commit="a" * 40,
        branch="episode-1",
        tool_modules=("directions",),
    )
    previous.update(state="pivot", finalized_at="2026-11-01T00:00:00+00:00")
    previous["direction_events"] = [
        {
            "direction_event_id": "directionevent_" + "3" * 32,
            "direction_id": local,
            "action": "propose",
            "name": "local redesign",
            "change_scope": "major_redesign",
            "objective": "local optimization objective",
            "plan": ["continue local work"],
            "stop_conditions": [],
            "recorded_at": "2026-11-01T00:00:00+00:00",
            "supporting_experiment_ids": [],
        },
        {
            "direction_event_id": "directionevent_" + "4" * 32,
            "direction_id": local,
            "action": "start",
            "analysis": "Keep the redesign active across Episodes.",
            "recorded_at": "2026-11-01T00:01:00+00:00",
            "supporting_experiment_ids": [],
        },
    ]
    private_write(previous_path, previous)

    current_path = journals / "episodes" / "e00000002" / "journal.json"
    current = initialize_journal(
        current_path,
        episode=2,
        base_commit="b" * 40,
        branch="episode-2",
        tool_modules=("directions",),
    )
    current.update(state="pivot", finalized_at="2026-11-01T01:00:00+00:00")
    private_write(current_path, current)

    projection = research_projection(current_path, journals)

    assert projection["directions"][peer]["status"] == "in_progress"
    assert projection["directions"][peer]["owner"] == "T2"
    assert projection["directions"][local]["status"] == "in_progress"


def test_pooled_search_derives_observational_winners_without_promoting(tmp_path):
    seed = {"latency_us_geomean": 10.0, "name": "seed"}

    class Driver:
        def bootstrap(self):
            return seed

        def initialize(self, initial, trajectories):
            assert initial == seed and trajectories == 2

        def prepare_epoch(self, epoch):
            assert epoch == 1

        def run_episode(self, epoch, trajectory, episode):
            candidate = (
                {
                    "latency_us_geomean": 5.0,
                    "kernel_commit_id": "1" * 40,
                    "gateway_record_id": "gateway-" + "1" * 32,
                }
                if trajectory == 1 else None
            )
            return {"epoch": epoch, "trajectory": trajectory, "episode": episode,
                    "agent_head_commit_id": str(trajectory) * 40,
                    "best_submitted_evaluation": candidate, "tokens": 1,
                    "session_id": f"session-t{trajectory}"}

        def seal_epoch(self, epoch, jobs):
            pass

        def epoch_sealed(self, epoch):
            return True

        def tokens(self):
            return 2

    root = tmp_path / "search"
    result = PooledSearch(
        root,
        SearchOptions(epochs=1, trajectories=2, episodes_per_trajectory=1,
                      max_parallel_episodes=2),
        Driver(),
    ).run()
    assert result == "search: completed Epoch budget"
    state = read_state(root / COORDINATOR_STATE)
    assert state["global_best"]["kernel_commit_id"] == "1" * 40
    assert state["global_best"]["trajectory"] == 1
    assert "lineages" not in state
    first = read_state(lineage_state_path(root, 1))
    second = read_state(lineage_state_path(root, 2))
    assert (
        first["lineage_winner"]["kernel_commit_id"]
        == "1" * 40
    )
    assert second["lineage_winner"] is None
    assert first["agent_head_commit_id"] == "1" * 40
    assert second["agent_head_commit_id"] == "2" * 40
    assert first["session_id"] == "session-t1"
    assert second["session_id"] == "session-t2"


def test_pooled_search_rejects_pre_submitted_winner_checkpoint(tmp_path):
    seed = {"latency_us_geomean": 10.0, "name": "legacy-seed"}
    options = SearchOptions(
        epochs=2,
        trajectories=2,
        episodes_per_trajectory=1,
        max_parallel_episodes=2,
        role_specialization=True,
    )

    class Driver:
        pass

    root = tmp_path / "search"
    root.mkdir()
    (root / COORDINATOR_STATE).write_text(json.dumps({
        "schema_version": 1,
        "options": {
            "epochs": 2,
            "trajectories": 2,
            "episodes_per_trajectory": 1,
            "max_parallel_episodes": 2,
        },
        "binding": "binding",
        "token_budget": 0,
        "bootstrap": seed,
        "seed": seed,
        "epoch": 2,
        "active": None,
        "completed_epochs": 1,
        "stop_reason": "stopped",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="predates persistent submitted-winner sharing"):
        PooledSearch(root, options, Driver()).run()
