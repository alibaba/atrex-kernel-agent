"""Reviewer lifecycle tests use fake CLI executables, never a model or GPU."""
import json
import sys
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest

from aka.agent.provider import Agent, AgentOptions
from aka.agent.execution.native import NativeExecution
from aka.contracts.agent import AgentRunRequest
from aka.workflow.light.campaign import ManagedCampaign as IntegratedCampaign
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.long_horizon.review_archive import (
    review_root, review_revision_root, reviewer_environment, reviewer_workspace,
    bound_review_usage_rows,
)
from aka.workflow.light.support.paths import supervisor_measurement_root
from aka.agent.session_capture import latest_complete_session_archive


FAKE_CLI = r'''
import json, os, sys
from pathlib import Path
backend = Path(sys.argv[0]).name
args = sys.argv[1:]
resume = "resume" in args or "--resume" in args or (backend == "pi" and "--session" in args)
if backend == "codex":
    thread = "00000000-0000-4000-8000-000000000001"
    home = Path(os.environ["CODEX_HOME"])
    rollout = home / "sessions" / ("rollout-" + thread + ".jsonl")
elif backend == "pi":
    flag = "--session" if resume else "--session-id"
    thread = args[args.index(flag) + 1]
    home = Path(os.environ["PI_CODING_AGENT_DIR"])
    rollout = home / "sessions" / "project" / ("2026-11-17T00-00-00-000Z_" + thread + ".jsonl")
else:
    flag = "--resume" if resume else "--session-id"
    thread = args[args.index(flag) + 1]
    home = Path(os.environ["CLAUDE_CONFIG_DIR"])
    project = str(Path.cwd()).replace("/", "-")
    rollout = home / "projects" / project / (thread + ".jsonl")
if resume and not rollout.exists():
    print("no rollout found for thread id " + thread, file=sys.stderr)
    sys.exit(1)
rollout.parent.mkdir(parents=True, exist_ok=True)
with rollout.open("a") as stream:
    if backend == "pi":
        if not resume:
            stream.write(json.dumps({"type":"session", "version":3, "id":thread, "cwd":str(Path.cwd())}) + "\n")
        else:
            assert "remember-this" in rollout.read_text(), "native history was lost"
        stream.write(json.dumps({"type":"message", "id":"m-" + str(resume), "message":{
            "role":"assistant", "content":[{"type":"text", "text":"reviewed fixture remember-this"}],
            "usage":{"input":7,"output":3,"cacheRead":0,"cacheWrite":0,"totalTokens":10}}}) + "\n")
    else:
        stream.write(json.dumps({"type":"session_meta", "payload":{"id":thread, "cwd":str(Path.cwd())}}) + "\n")
Path("dependency_review.json").write_text(json.dumps({"fixture":True}))
Path("review.json").write_text(json.dumps({"verdict":"supported", "primary_issue":"Fixture evidence",
    "findings":[{"claim":"Fixture", "judgment":"supported", "decision_critical":True,
                 "reason":"Observed fixture", "evidence":[]}],
    "scope_safe_interpretation":"Fixture only", "next_check":"Measure next", "confidence":"high"}))
if backend == "pi":
    events = [{"type":"session", "version":3, "id":thread, "cwd":str(Path.cwd())},
              {"type":"message_end", "message":{"role":"assistant",
               "content":[{"type":"text", "text":"reviewed fixture remember-this"}],
               "usage":{"input":7,"output":3,"cacheRead":0,"cacheWrite":0,"totalTokens":10}}},
              {"type":"agent_settled", "aborted":False}]
elif backend == "codex":
    events = [{"type":"thread.started", "thread_id":thread},
              {"type":"item.completed", "item":{"type":"agent_message", "text":"reviewed fixture"}},
              {"type":"turn.completed", "usage":{"input_tokens":7,"output_tokens":3}}]
else:
    events = [{"type":"assistant", "message":{"id":"m-" + str(resume), "role":"assistant",
              "content":[{"type":"text", "text":"reviewed fixture"}], "usage":{"input_tokens":7,"output_tokens":3}}},
              {"type":"result", "subtype":"success", "session_id":thread,
               "usage":{"input_tokens":7,"output_tokens":3}}]
for event in events:
    print(json.dumps(event), flush=True)
'''


def owner(tmp_path, backend, sandbox):
    root = tmp_path.resolve()
    binary = root / "bin" / backend
    binary.parent.mkdir(exist_ok=True)
    binary.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
    binary.chmod(0o700)
    host = root / "host"
    host.mkdir(exist_ok=True)
    # Exercise the real bwrap preparation/identity contract on macOS too;
    # only the OS launcher is replaced, not prepare_home or the CLI runtime.
    with patch("aka.agent.execution.bubblewrap.BubblewrapExecution", return_value=NativeExecution()):
        agent = Agent(AgentOptions(sandbox=sandbox), environment={
            "HOME": str(host), "PATH": str(binary.parent) + ":/usr/bin:/bin",
        })
    return agent


@pytest.mark.parametrize("backend", ["codex", "claude", "pi"])
@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
def test_review_resume_after_owner_restart_and_complete_capture(tmp_path, backend, sandbox):
    root = review_root(tmp_path.resolve() / "measurements", "submission-fixture")
    homes, ids = [], []
    for revision in (1, 2):
        agent = owner(tmp_path, backend, sandbox)
        try:
            with reviewer_workspace(root) as workspace:
                env = {**reviewer_environment(root),
                       "ATREX_SESSION_CAPTURE_DIR": str(root / f"v{revision}" / "captures")}
                result = agent.runtime(backend).run(AgentRunRequest(
                    workspace, "review fixture", None, environment=env,
                    session_id=ids[-1] if ids else None, resume_session=bool(ids),
                ))
                assert result.exit_status == 0, result.stderr_tail
                ids.append(result.session_id)
                native = list((root / "home").rglob("*" + result.session_id + ".jsonl"))
                assert len(native) == 1
                homes.append(native[0])
                capture = latest_complete_session_archive(Path(env["ATREX_SESSION_CAPTURE_DIR"]))
                assert capture is not None
                assert "reviewed fixture" in capture.conversation.read_text()
                assert json.loads(capture.token_usage.read_text())["total"]["total_tokens"] >= 10
                if backend == "pi":
                    usage = json.loads(capture.token_usage.read_text())
                    assert usage["session_id"] == result.session_id
                    assert usage["total"]["total_tokens"] == 10
                    assert result.terminal_usage.total_tokens == 10
        finally:
            agent.close()
    assert ids[0] == ids[1]
    assert homes[0] == homes[1]
    assert homes[0].exists()


def make_campaign(tmp_path, monkeypatch, agent, backend="codex"):
    campaign = Campaign("fixture", "fixture", "L20N", "cuda", work_dir=str(tmp_path),
                        agent_cli=backend, optimization_mode="ultra")
    campaign.workspace.mkdir(parents=True)
    campaign.agent = agent
    campaign.agent_access = MethodType(IntegratedCampaign.agent_access, campaign)
    campaign._run_session = MethodType(IntegratedCampaign._run_session, campaign)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text("fixture = 1\n")
    module = "aka.workflow.light.orchestration.campaign"
    monkeypatch.setattr(module + "._production_review_candidate_paths", lambda w: [w / "kernel.py"])
    monkeypatch.setattr(module + "._production_review_digest", lambda *args: "fixture-digest")
    monkeypatch.setattr(module + ".validate_light_production_review", lambda *a, **kw: ([], "allowed", {"verdict":"allow"}))
    return campaign, candidate


@pytest.mark.parametrize("backend", ["codex", "pi"])
def test_submission_revisions_archive_and_resume_in_measurements(tmp_path, monkeypatch, backend):
    agent = owner(tmp_path, backend, "none")
    try:
        campaign, candidate = make_campaign(tmp_path, monkeypatch, agent, backend)
        sid = "submission-" + "a" * 32
        session = None
        for revision in (1, 2):
            campaign._production_review_cache.clear()
            errors = campaign._review_production_candidate_once(candidate, "cuda", False,
                candidate_digest="fixture-digest", submission_id=sid, submission_revision=revision,
                resume_session_id=session)
            assert errors == []
            archive = review_revision_root(supervisor_measurement_root(campaign.workspace), sid, revision)
            result = json.loads((archive / "result.json").read_text())
            assert result["accepted"] is True
            assert result.get("resumed_from_session_id") == session
            session = result["session_id"]
            assert "reviewed fixture" in (archive / "conversation.jsonl").read_text()
            assert (archive / "token-usage.json").is_file()
        assert not list(tmp_path.rglob("review-session-homes"))
    finally:
        agent.close()


def test_failed_resume_keeps_conversation_usage_and_failure(tmp_path, monkeypatch):
    from aka.workflow.light.orchestration.infrastructure_retry import ReviewerOutputUnavailable
    agent = owner(tmp_path, "codex", "none")
    try:
        campaign, candidate = make_campaign(tmp_path, monkeypatch, agent)
        sid = "submission-" + "b" * 32
        with pytest.raises(ReviewerOutputUnavailable, match="resume storage"):
            campaign._review_production_candidate_once(candidate, "cuda", False,
                candidate_digest="fixture-digest", submission_id=sid,
                resume_session_id="00000000-0000-4000-8000-000000000001")
        archive = review_revision_root(supervisor_measurement_root(campaign.workspace), sid, 1)
        result = json.loads((archive / "result.json").read_text())
        assert result["accepted"] is False
        assert result["exit_status"] == 1
        assert "no rollout found" in result["stderr_tail"]
        assert (archive / "conversation.jsonl").is_file()
        assert (archive / "token-usage.json").is_file()
    finally:
        agent.close()


def test_all_final_source_attempts_count_once(tmp_path):
    sid = "submission-" + "c" * 32
    submission = tmp_path / "submissions" / sid / "v1"
    submission.mkdir(parents=True)
    (submission / "submission.json").write_text(json.dumps({"lineage":"T1"}))
    archive = review_revision_root(tmp_path / "measurements", sid, 1)
    archive.mkdir(parents=True)
    (archive / "result.json").write_text(json.dumps({"review_execution":"executed"}))
    for name in ("one", "two"):
        path = archive / "captures" / name / "run-1"
        path.mkdir(parents=True)
        (path / "token-usage.json").write_text(json.dumps({"total":{"total_tokens":10}}))
    (archive / "token-usage.json").write_text(json.dumps({"total":{"total_tokens":10}}))
    rows = list(bound_review_usage_rows(tmp_path, lineage="T1"))
    assert len(rows) == 2
    assert sum(row["usage"]["total_tokens"] for row in rows) == 20


@pytest.mark.parametrize("claim_type", ["direction", "experiment"])
def test_claim_revision_resumes_and_archives_in_measurements(tmp_path, monkeypatch, claim_type):
    from aka.workflow.light.host.journal import SupervisorJournalService
    from aka.workflow.light.long_horizon.review_archive import archive_review
    agent = owner(tmp_path, "claude", "none")
    try:
        campaign, _ = make_campaign(tmp_path, monkeypatch, agent)
        claim_id = claim_type + "_" + "a" * 32
        store = supervisor_measurement_root(campaign.workspace)
        service = SupervisorJournalService.__new__(SupervisorJournalService)
        service.measurement_root = store
        service.campaign_root = campaign.workspace
        session = None
        # Direction v2 is a lifecycle-only start, so its next Review is v3.
        for revision in ((1, 3) if claim_type == "direction" else (1, 2)):
            source = tmp_path / f"inputs-{revision}"
            (source / "submissions").mkdir(parents=True)
            target = {claim_type + "_id": claim_id, "revision": revision}
            (source / "submissions/target.json").write_text(json.dumps(target))
            (source / "evidence-manifest.json").write_text(json.dumps({"claim_type":claim_type}))
            archive = review_revision_root(store, claim_id, revision)
            review = campaign.review_research_claim(source, archive_root=archive, resume_session_id=session)
            assert review["verdict"] == "supported"
            attempts = review.pop("_review_session_audit")["attempts"]
            assert len(attempts) == 1
            assert attempts[0]["resumed"] == (revision > 1)
            if session:
                assert attempts[0]["session_id"] == session
            session = attempts[0]["session_id"]
            archive_review(archive, result={"claim_type":claim_type,
                "review":{"session_audit":{"attempts":attempts}}}, conversation=None, token_usage=None)
            assert service._prior_claim_review_session(claim_type, target) == session
            assert "reviewed fixture" in (archive / "conversation.jsonl").read_text()
            assert (archive / "attempts/a1/token-usage.json").is_file()
    finally:
        agent.close()


def test_missing_old_rollout_retries_fresh_once_and_keeps_both_attempts(tmp_path, monkeypatch):
    agent = owner(tmp_path, "codex", "none")
    try:
        campaign, candidate = make_campaign(tmp_path, monkeypatch, agent)
        monkeypatch.setattr("aka.task.source.formats.seed_baseline_metadata", lambda *a, **kw: None)
        sid = "submission-" + "d" * 32
        assert campaign._review_production_candidate(candidate, "cuda", False,
            submission_id=sid, resume_session_id="00000000-0000-4000-8000-000000000001") == []
        archive = review_revision_root(supervisor_measurement_root(campaign.workspace), sid, 1)
        assert json.loads((archive / "result.json").read_text())["accepted"] is True
        tokens = [json.loads(path.read_text()) for path in archive.glob("captures/*/run-*/token-usage.json")]
        assert sorted(row["exit_status"] for row in tokens) == [0, 1]
    finally:
        agent.close()


@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
def test_pi_native_checkpoint_restores_history_after_owner_restart(tmp_path, sandbox):
    root = review_root(tmp_path.resolve() / "measurements", "main-session-fixture")
    output = tmp_path / "private-agent-state"
    agent = owner(tmp_path, "pi", sandbox)
    try:
        with reviewer_workspace(root) as workspace:
            session = agent.open_session("pi", workspace, output_directory=output,
                                         environment=reviewer_environment(root))
            assert session.supports_resume
            first = session.invoke("remember-this")
            assert first.exit_status == 0, first.stderr
            session.close()
            snapshot = agent.export_checkpoint(output, tmp_path / "checkpoint")
            files = json.loads((snapshot.directory / "manifest.json").read_text())["files"]
            assert len(files) == 1
            assert files[0]["runtime"] == "pi"
            native = Path(files[0]["target"])
            original = native.read_bytes()
            native.unlink()
    finally:
        agent.close()
    agent = owner(tmp_path, "pi", sandbox)
    try:
        agent.restore_checkpoint(snapshot)
        assert native.read_bytes() == original
        session = agent.open_session("pi", workspace, session_id=first.session_id,
                                     output_directory=output, environment=reviewer_environment(root))
        second = session.invoke("continue", resume=True)
        assert second.exit_status == 0, second.stderr
        assert second.session_id == first.session_id
        assert second.terminal_usage.total_tokens == first.terminal_usage.total_tokens == 10
        session.close()
        assert native.read_bytes().startswith(original)
        assert not list((output / ".aka-agent-session-errors").glob("*"))
    finally:
        agent.close()


def test_pi_resume_requires_existing_native_history(tmp_path):
    from aka.agent.native import resume_session_command, supports_same_session_resume
    command = resume_session_command("continue", "missing-session", "high", "pi", environment={
        "ATREX_PI_SESSION_SETTINGS": json.dumps({"provider":"fixture", "model":"fixture-model"})})
    assert supports_same_session_resume("pi")
    assert command[command.index("--session") + 1] == "missing-session"
    assert "--session-id" not in command
    assert "--resume" not in command
    assert command[command.index("--provider") + 1] == "fixture"
    assert command[-1] == "continue"
    root = review_root(tmp_path.resolve() / "measurements", "missing-session-fixture")
    agent = owner(tmp_path, "pi", "none")
    try:
        with reviewer_workspace(root) as workspace:
            result = agent.runtime("pi").run(AgentRunRequest(workspace, "continue", None,
                session_id="missing-session", resume_session=True, environment=reviewer_environment(root)))
        assert result.exit_status == 1
        assert "no rollout found" in result.stderr_tail
        assert not list((root / "home").rglob("*missing-session.jsonl"))
    finally:
        agent.close()
