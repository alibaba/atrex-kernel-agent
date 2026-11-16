import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.light.evaluation import agate_native, response_policy, ssh_devices
from aka.light.evaluation import ssh_native, ssh_worker
from aka.light.experiment import records
from aka.light.agent.session_capture import SessionCapture
from aka.light.agent.session_usage import estimate_usage_from_conversation
from aka.light.runtime.paths import supervisor_campaign_root, supervisor_submission_root
from aka.light.workflow.token_checkpoints import capture_token_checkpoint
from aka.light.workflow.bootstrap import Campaign
from aka.light.runtime.server import SupervisorRuntime
from aka.light.acceptance.review import validate_light_production_review
from aka.legacy.acceptance.review import _validate_production_review
from aka.legacy.agent import dsh
from observability._internal.token_usage import snapshot as token_time_snapshot


def _jsonl(*events):
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_conversation_token_fallback_is_explicitly_estimated(tmp_path):
    conversation = _jsonl(
        {
            "type": "message",
            "source": "runtime_input",
            "role": "user",
            "content": [{"type": "text", "text": "Implement vector add."}],
        },
        {
            "type": "provider_event",
            "path": "provider/stdout.stream-json",
            "event": {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Use one CUDA thread per element."},
                        {"type": "text", "text": "Implemented and measured."},
                    ],
                },
            },
        },
    )
    usage, detail = estimate_usage_from_conversation(conversation)
    assert usage.measurement == "estimated"
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.reasoning_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.cache_read_tokens is None
    assert detail["method"] == "conversation_text_cjk_plus_utf8_bytes_v1"

    capture = SessionCapture(
        tmp_path / "sessions",
        backend="claude",
        command=["claude", "Count this archived prompt."],
        context={"workspace": str(tmp_path)},
        native_environment={},
    )
    capture.finish(exit_status=0)
    report = json.loads((capture.root / "token-usage.json").read_text())
    assert report["total"]["measurement"] == "estimated"
    assert report["total"]["total_tokens"] > 0
    assert report["accounting_basis"] == "conversation_text_estimate"
    assert report["terminal"]["total_tokens"] is None

    exact = SessionCapture(
        tmp_path / "exact-sessions",
        backend="claude",
        command=["claude", "Use the provider receipt."],
        context={"workspace": str(tmp_path)},
        native_environment={},
    )
    exact._chunks["stdout"].append(_jsonl({
        "type": "result",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }))
    exact.finish(exit_status=0)
    exact_report = json.loads((exact.root / "token-usage.json").read_text())
    assert exact_report["total"]["measurement"] == "exact"
    assert exact_report["total"]["total_tokens"] == 15
    assert "conversation_estimate" not in exact_report


def test_kernel_milestone_token_checkpoint_separates_main_and_reviews(tmp_path):
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    audit_root = supervisor_campaign_root(workspace)
    capture_root = audit_root / "segments/episodes/e0002/sessions"
    first = audit_root / "segments/episodes/e0001/sessions/run-first"
    current = capture_root / "run-current"
    first.mkdir(parents=True)
    current.mkdir(parents=True)
    (first / "token-usage.json").write_text(json.dumps({
        "session_id": "first",
        "started_at": "2026-11-12T01:00:00+00:00",
        "finished_at": "2026-11-12T01:10:00+00:00",
        "state": "finished",
        "accounting_basis": "provider_result_usage",
        "total": {
            "input_tokens": 80,
            "output_tokens": 20,
            "reasoning_tokens": 10,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 100,
            "measurement": "exact",
        },
    }), encoding="utf-8")
    (current / "token-usage.json").write_text(json.dumps({
        "session_id": "current",
        "started_at": "2026-11-12T02:00:00+00:00",
        "state": "running",
        "accounting_basis": "main_response_fallback",
        "total": {
            "input_tokens": 40,
            "output_tokens": 10,
            "reasoning_tokens": 5,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "total_tokens": 50,
            "measurement": "estimated",
        },
    }), encoding="utf-8")
    review = audit_root / "review-usage/final_source_policy_review/review.json"
    review.parent.mkdir(parents=True)
    review.write_text(json.dumps({
        "schema": "atrex.light-review-usage.v1",
        "review_kind": "final_source_policy_review",
        "label": "source review",
        "recorded_at": "2026-11-12T02:04:00+00:00",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "reasoning_tokens": 2,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 25,
            "measurement": "exact",
        },
    }), encoding="utf-8")

    reference, stored = capture_token_checkpoint(
        audit_root=audit_root,
        session_capture_root=capture_root,
        event="accepted-submit-kernel",
        checkpoint_id="submission-test",
        occurred_at="2026-11-12T02:05:00+00:00",
        identity={"kernel_commit_id": "a" * 40},
    )

    assert reference["main_agent_total_tokens"] == 150
    assert reference["review_total_tokens"] == 25
    assert reference["total_tokens"] == 175
    assert reference["reasoning_tokens"] == 17
    assert reference["measurement"] == "estimated"
    assert stored["reviews_by_kind"]["final_source_policy_review"]["usage"][
        "total_tokens"
    ] == 25
    report = token_time_snapshot(workspace)
    assert report["accepted_submission_token_checkpoints"][0]["usage"][
        "total_tokens"
    ] == 175


def test_passing_full_evaluate_records_a_token_checkpoint(tmp_path):
    from aka.light.workflow.journal_service import SupervisorJournalService

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign_root = tmp_path / "supervisor" / "journals"
    evidence_root = campaign_root / "episodes/e00000001"
    evidence_root.mkdir(parents=True)
    capture_root = tmp_path / "sessions/e0001/sessions"
    usage = capture_root / "run-main/token-usage.json"
    usage.parent.mkdir(parents=True)
    usage.write_text(json.dumps({
        "session_id": "main",
        "started_at": "2026-11-12T01:00:00+00:00",
        "state": "running",
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
    service = SupervisorJournalService(
        workspace=workspace,
        campaign_root=campaign_root,
        evidence_root=evidence_root,
        research_root=tmp_path / "research",
        session_capture_root=capture_root,
    )
    record_id = "gateway-" + "1" * 32
    kernel_id = "a" * 40
    result = {
        "all_pass": True,
        "mode": "full",
        "latency_us_by_shape": {"shape-0": 4.0, "shape-1": 9.0},
        "latency_us_geomean": 6.0,
    }
    record = {
        "gateway_record_id": record_id,
        "operation": "evaluate",
        "cacheable": True,
        "created_at": "2026-11-12T01:05:00+00:00",
        "kernels": {"candidate": {"kernel_id": kernel_id}},
        "request": {
            "operation": "evaluate",
            "options": {
                "command": ["python3", "test_kernel.py"],
                "evaluation_mode": "full",
                "evaluation_policy": {"mode": "full"},
            },
        },
        "response": {
            "exit_code": 0,
            "truncated": False,
            "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
        },
    }

    reference = service.record_passing_full_evaluate_checkpoint(record)

    assert reference is not None
    assert reference["checkpoint_id"] == record_id
    assert reference["total_tokens"] == 100
    saved = json.loads((campaign_root.parent / reference["path"]).read_text())
    assert saved["event"] == "passing-full-evaluate"
    assert saved["identity"]["kernel_commit_id"] == kernel_id
    assert saved["identity"]["latency_us_geomean"] == 6.0


def test_dsh_runtime_connection_is_private_and_ephemeral(tmp_path):
    environment = {
        "DSH_HOME": str(tmp_path / "dsh"),
        "ATREX_AKA_RUNTIME_URL": "http://runtime.invalid",
        "ATREX_AKA_RUNTIME_TOKEN": "secret-token",
    }

    prepared = dsh.prepare_runtime_connection(environment)
    connection = Path(prepared[dsh.CONNECTION_ENV])
    assert connection.is_file()
    assert connection.stat().st_mode & 0o777 == 0o600
    assert json.loads(connection.read_text()) == {
        "url": "http://runtime.invalid",
        "token": "secret-token",
    }
    assert dsh.CONNECTION_ENV not in environment

    dsh.release_runtime_connection(prepared)
    assert not connection.exists()


def test_dsh_stream_reports_session_failure_and_exact_usage():
    stdout = _jsonl(
        {"type": "session", "sessionId": "session_1"},
        {"type": "status", "phase": "step_start", "turn": 0, "step": 0},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 0,
            "step": 0,
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadTokens": 2,
                "cacheWriteTokens": 1,
                "totalTokens": 17,
            },
        },
        {
            "type": "final",
            "sessionId": "session_1",
            "reason": {"kind": "completed"},
        },
    )

    assert dsh.session_id_from_stream(stdout) == "session_1"
    assert dsh.terminal_failure(stdout) == (None, None)
    events, usage = dsh.normalize_stream(stdout)
    assert [event.kind for event in events] == ["usage_delta", "terminal_usage"]
    assert usage.total_tokens == 17
    assert usage.measurement == "exact"

    limited = _jsonl(
        {"type": "session", "sessionId": "session_1"},
        {
            "type": "final",
            "sessionId": "session_1",
            "reason": {"kind": "max-tokens"},
        },
    )
    assert dsh.terminal_failure(limited) == ("output_limit", "max_output_tokens")


def test_dsh_settings_reject_paths_and_unknown_authority():
    assert dsh.settings_values('{"provider":"anthropic","profile":"production"}') == {
        "provider": "anthropic",
        "profile": "production",
    }
    with pytest.raises(ValueError, match="profile name"):
        dsh.settings_values('{"profile":"../private/profile.json"}')
    with pytest.raises(ValueError, match="accept only"):
        dsh.settings_values('{"token":"secret"}')


def _abba_state(phase, endpoint, payload, *, succeeded=False):
    state = {
        "phase": phase,
        "identity": {
            "kind": "eval",
            "url_fingerprint": endpoint,
            "payload": payload,
        },
    }
    if phase == "terminal":
        job = {
            "status": "succeeded" if succeeded else "failed",
            "result": {"complete": True},
        }
        state["process"] = {
            "returncode": 0 if succeeded else 1,
            "stdout": json.dumps(job),
            "stderr": "",
        }
    return state


def test_abba_replacement_evidence_requires_exact_payload_and_new_gateway(monkeypatch):
    monkeypatch.setattr(
        records,
        "cacheable_job",
        lambda job, identity: job["status"] == "succeeded",
    )
    payload = {"abba": {"repeats": 2}, "candidate": "source", "shape_ids": ["s0"]}
    accepted = _abba_state("accepted", "old-gateway", payload)
    replacement = _abba_state("terminal", "new-gateway", payload, succeeded=True)

    assert records._abba_acceptance_states([accepted, replacement]) == [replacement]

    changed = _abba_state(
        "terminal",
        "new-gateway",
        {**payload, "shape_ids": ["s1"]},
        succeeded=True,
    )
    assert records._abba_acceptance_states([accepted, changed]) == [accepted, changed]

    same_gateway = _abba_state("terminal", "old-gateway", payload, succeeded=True)
    assert records._abba_acceptance_states([accepted, same_gateway]) == [accepted, same_gateway]


def test_agate_native_placement_is_forwarded_to_diagnostic(monkeypatch):
    option = agate_native.NativeOption(
        flag="--cluster-label",
        dest="cluster_label",
        commands=frozenset({"eval"}),
        action="append",
        choices=None,
        value_type=str,
        metavar="KEY=VALUE",
        help=None,
        schema_type="string",
    )
    monkeypatch.setattr(agate_native, "catalog", lambda: {option.flag: option})
    args = Namespace(agate_native__cluster_label=["pool=L20D", "zone=cn-a"])
    request = {"__atrex_source_diagnostic": {}}

    result = agate_native.apply_request(request, args, "run")

    expected = {"cluster_selector": {"pool": "L20D", "zone": "cn-a"}}
    assert result["placement"] == expected
    assert result["__atrex_source_diagnostic"]["placement"] == expected


def test_unknown_gateway_outcome_is_not_misclassified_as_known():
    assert response_policy.known_gateway_error(
        {"error": {"error_class": "timeout", "message": "deadline"}}
    )
    assert not response_policy.known_gateway_error(
        {"error": {"error_class": "new_server_failure", "message": "opaque"}}
    )


def test_ssh_worker_materializes_source_tree_and_rejects_traversal(tmp_path):
    request = {
        "__atrex_eval_archives": {
            "candidate.tar.gz": {
                "kernel.py": "from impl.kernel import Model\n",
                "impl/kernel.py": "class Model: pass\n",
            }
        }
    }
    entry = ssh_worker.materialize(
        {"archive": "candidate.tar.gz", "entry_point": "kernel.py"},
        request,
        tmp_path,
    )
    assert entry == tmp_path / "kernel.py"
    assert (tmp_path / "impl/kernel.py").read_text() == "class Model: pass\n"

    with pytest.raises(ValueError, match="relative"):
        ssh_worker.write(tmp_path, "../escape.py", "bad")


def test_ssh_native_manifest_fails_closed_before_compilation(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ssh_worker", ssh_worker)
    manifest = {
        "targets": {"hardware": ["L20D"]},
        "source": {"archive": "native.tar.gz", "files": {}},
    }
    request = {
        "spec": {"target_hardware": ["L20N"]},
        "reference": {"shapes": {}},
        "__atrex_eval_archives": {"native.tar.gz": {}},
    }
    with pytest.raises(ValueError, match="targets do not cover"):
        ssh_native.materialize(manifest, request, tmp_path)


def test_ssh_device_map_uses_explicit_character_devices():
    assignment = ssh_devices.resolve(
        0,
        device_map={
            "0": {
                "vendor": "nvidia",
                "uuid": "GPU-test",
                "devices": ["/dev/null"],
                "environment": {"CUDA_VISIBLE_DEVICES": "GPU-test"},
            }
        },
    )
    assert assignment["index"] == 0
    assert assignment["devices"] == ["/dev/null"]
    with pytest.raises(ValueError, match="device or runtime"):
        ssh_devices.resolve(0, isolation="process", device_map={})


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


def test_light_reuses_upstream_review_and_ignores_only_non_pinned_dsl():
    review = _upstream_source_review(
        framework_decision="reject",
        framework_category="alternate_framework",
    )
    upstream_errors, _ = _validate_production_review(
        review,
        candidate_files=frozenset({"kernel.py"}),
    )
    guided_errors, _, guided = validate_light_production_review(
        review,
        candidate_files=frozenset({"kernel.py"}),
        dsl_policy="guided",
    )
    pinned_errors, _, pinned = validate_light_production_review(
        review,
        candidate_files=frozenset({"kernel.py"}),
        dsl_policy="pinned",
    )

    assert upstream_errors
    assert guided_errors == []
    assert guided["verdict"] == "allow"
    assert [item["id"] for item in guided["checks"]] == [
        "compute_provenance",
        "dependency_inventory",
        "external_code_loading",
        "solution_manifest",
    ]
    assert "reviewer_verdict" not in guided
    assert all("reviewer_decision" not in item for item in guided["checks"])
    assert pinned["verdict"] == "reject"
    assert any("framework_compliance" in error for error in pinned_errors)


def test_upstream_review_allows_framework_runtime_and_support_plumbing():
    review = _upstream_source_review()
    review["dependencies"].append({
        "name": "ninja",
        "decision": "allow",
        "category": "toolchain_plumbing",
        "reason": "Ninja only schedules candidate compilation",
        "evidence": ["candidate/kernel.py:1"],
    })
    errors, _ = _validate_production_review(
        review,
        candidate_files=frozenset({"kernel.py"}),
    )
    assert errors == []


def test_fast_submission_skips_independent_final_source_reviewer(tmp_path):
    (tmp_path / "kernel.py").write_text(
        "def custom_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    campaign = Campaign.__new__(Campaign)
    campaign.__dict__["source_checks_enabled"] = True
    campaign.optimization_mode = "fast"
    campaign.specify_dsl = False
    campaign.framework = ""

    def unexpected_review(*_args, **_kwargs):
        raise AssertionError("Fast must not start the independent final-source Reviewer")

    campaign._review_production_candidate = unexpected_review
    assert campaign._production_kernel_violations(tmp_path) == []


def test_submit_kernel_receipt_reports_only_the_failed_policy_axis(
    monkeypatch, tmp_path
):
    commit_id = "b" * 40
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.production_review_required = True
    runtime.submission_specify_dsl = True
    runtime.config = SimpleNamespace(optimization_mode="ultra")
    runtime.measurements = object()
    def policy_validator(*_args, **kwargs):
        assert kwargs["submission_id"] == "submission-" + "1" * 32
        assert kwargs["submission_revision"] == 2
        assert kwargs["resume_session_id"] == "review-session-v1"
        return {
            "violations": ["external operator performs target computation"],
            "source_contract": "pass",
            "implementation_review": {"availability": "available"},
            "review": {
                "schema_version": 2,
                "verdict": "reject",
                "checks": [{
                    "id": "compute_provenance",
                    "decision": "reject",
                    "category": "prebuilt_compute",
                    "reason": "external operator performs target computation",
                    "evidence": ["candidate/kernel.py:1"],
                }],
                "dependencies": [],
                "summary": "external compute remains",
            },
        }

    runtime.production_policy_validator = policy_validator
    monkeypatch.setattr(
        "aka.light.workspace.sources.from_git",
        lambda _workspace, _commit: b"candidate",
    )
    monkeypatch.setattr(
        "aka.light.runtime.server.kernel_full_evaluation_evidence",
        lambda _store, _commit: {"status": "passed"},
    )
    agent_git = SimpleNamespace(
        workspace=tmp_path,
        commit_reachable=lambda value: value == commit_id,
    )
    binding = SimpleNamespace(
        service=SimpleNamespace(git_workspace=tmp_path),
        require_gluon=False,
    )

    result = runtime._review_submitted_kernel(
        SimpleNamespace(),
        binding,
        agent_git,
        commit_id,
        "submission-" + "1" * 32,
        2,
        "review-session-v1",
    )

    assert result["accepted"] is False
    assert result["source_contract"] == "pass"
    assert result["review"] == {
        "schema_version": 2,
        "verdict": "reject",
        "checks": [{
            "id": "compute_provenance",
            "decision": "reject",
            "category": "prebuilt_compute",
            "reason": "external operator performs target computation",
            "evidence": ["candidate/kernel.py:1"],
        }],
        "dependencies": [],
        "summary": "external compute remains",
    }


def test_submission_policy_feedback_preserves_observed_implementation_facts(
    monkeypatch, tmp_path
):
    campaign = Campaign.__new__(Campaign)
    campaign.__dict__["source_checks_enabled"] = True
    campaign.optimization_mode = "performance"
    campaign.specify_dsl = False

    monkeypatch.setattr(
        "aka.light.workspace.sources.install",
        lambda workspace, source: None,
    )

    def review(_workspace, *, require_gluon, review_record_sink):
        assert require_gluon is False
        review_record_sink({
            "dsl_policy": "guided",
            "framework": None,
            "required_phase": "open_dsl",
            "review": {
                "schema_version": 2,
                "verdict": "allow",
            },
            "policy_result": {
                "verdict": "allow",
                "checks": [
                    {"id": "compute_provenance", "decision": "allow"},
                ],
                "dependencies": [],
                "summary": "self-authored compute",
            },
        })
        return []

    campaign._production_kernel_violations = review
    result = campaign.production_policy_feedback(tmp_path, b"candidate")

    assert result["violations"] == []
    assert result["source_contract"] == "pass"
    assert result["review"] == {
        "verdict": "allow",
        "checks": [{"id": "compute_provenance", "decision": "allow"}],
        "dependencies": [],
        "summary": "self-authored compute",
    }
    assert result["implementation_review"] == {
        "availability": "available",
        "dsl_policy": "guided",
        "configured_dsl": None,
        "required_phase": "open_dsl",
    }


def test_submit_kernel_review_keeps_implementation_facts_private_result(
    monkeypatch, tmp_path
):
    commit_id = "a" * 40
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.production_review_required = True
    runtime.submission_specify_dsl = False
    runtime.config = SimpleNamespace(optimization_mode="fast")
    runtime.measurements = object()
    runtime.production_policy_validator = lambda *_args, **_kwargs: {
        "violations": [],
        "implementation_review": {
            "availability": "available",
            "technologies": ["triton"],
            "dsl_policy": "open",
        },
    }
    monkeypatch.setattr(
        "aka.light.workspace.sources.from_git",
        lambda _workspace, _commit: b"candidate",
    )
    monkeypatch.setattr(
        "aka.light.runtime.server.kernel_full_evaluation_evidence",
        lambda _store, _commit: {"status": "passed"},
    )
    agent_git = SimpleNamespace(
        workspace=tmp_path,
        commit_reachable=lambda value: value == commit_id,
    )
    binding = SimpleNamespace(
        service=SimpleNamespace(git_workspace=tmp_path),
        require_gluon=False,
    )

    result = runtime._review_submitted_kernel(
        SimpleNamespace(), binding, agent_git, commit_id
    )

    assert result["accepted"] is True
    assert result["implementation_review"] == {
        "availability": "available",
        "technologies": ["triton"],
        "dsl_policy": "open",
    }


def test_final_source_review_is_attached_to_submission(tmp_path):
    submission_id = "submission-" + "a" * 32
    conversation = tmp_path / "conversation.jsonl"
    token_usage = tmp_path / "token-usage.json"
    conversation.write_text('{"type":"session_end"}\n', encoding="utf-8")
    token_usage.write_text(
        json.dumps({"total": {"total_tokens": 42}}) + "\n",
        encoding="utf-8",
    )

    errors = Campaign._persist_production_review_record(
        tmp_path,
        submission_id,
        1,
        {
            "review_execution": "executed",
            "accepted": True,
            "session_id": "review-session",
        },
        conversation=conversation,
        token_usage=token_usage,
    )

    assert errors == []
    root = supervisor_submission_root(tmp_path) / submission_id / "v1" / "review"
    assert json.loads((root / "result.json").read_text())["accepted"] is True
    assert (root / "conversation.jsonl").read_text() == '{"type":"session_end"}\n'
    usage = json.loads((root / "token-usage.json").read_text())
    assert usage["total"]["total_tokens"] == 42


def test_upstream_reviewer_rejects_inconsistent_decision_and_category():
    review = _upstream_source_review()
    review["checks"][1]["decision"] = "reject"
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_production_review(
            review,
            candidate_files=frozenset({"kernel.py"}),
        )


def test_light_adapter_enforces_pinned_framework_review_decision():
    review = _upstream_source_review(
        framework_decision="reject",
        framework_category="alternate_framework",
    )
    errors, _, adapted = validate_light_production_review(
        review,
        candidate_files=frozenset({"kernel.py"}),
        dsl_policy="pinned",
    )
    assert adapted["verdict"] == "reject"
    assert any("framework_compliance" in error for error in errors)


def test_final_source_reviewer_prompt_uses_upstream_verdict_schema():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "src/aka/light/workflow/prompts/dependency_review.md"
    ).read_text(encoding="utf-8")
    assert '"schema_version": 2' in prompt
    assert '"checks"' in prompt
    assert '"decision": "allow | reject"' in prompt
    assert '"verdict": "allow | reject"' in prompt
    assert "Fast mode does not invoke this review" in prompt
