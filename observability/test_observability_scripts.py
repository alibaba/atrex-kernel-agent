from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from aka.contracts.workflow_export import (
    CampaignRef,
    CandidateFact,
    EpisodeFact,
    ProgressFact,
    ResultFact,
    SessionFact,
    WorkflowExportSnapshot,
)
from observability.common import Scope
from observability import export_light_evidence
from observability.light_acceptance import snapshot as acceptance_snapshot
from observability.light_agate_status import (
    _batch_progress,
    _error,
    _retry_count,
    _timing,
    snapshot as agate_snapshot,
)
from observability.light_performance import _shape_rows, snapshot as performance_snapshot
from observability.light_runtime_status import _conversation_phase, snapshot as runtime_snapshot
from observability.light_token_time import _session_rows
from observability.light_tool_usage import _duration, _tool
from observability.light_wiki_usage import _journal_adoption, _knowledge_ids


def result(*, state="pass", latency=10.0):
    return ResultFact(
        state,
        "PASS" if state == "pass" else "FAIL",
        "PASS" if state == "pass" else "FAIL",
        state == "pass",
        latency,
        {"s1": latency} if latency else {},
        ("s1",),
        "candidate",
        "latency_us_geomean",
        None,
    )


def public_snapshot(workspace: Path):
    candidate = CandidateFact(0, None, "a" * 40, "a" * 40, True, None, result())
    episode = EpisodeFact(1, 1, "candidate_ready", True, "b" * 40, "session-1", "done", "PASS")
    session = SessionFact("session-1", "claude", 1, "observed", "2026-11-06T00:00:00+00:00", {"total_tokens": 10})
    return WorkflowExportSnapshot(
        3,
        CampaignRef("campaign", workspace, workspace.name),
        "cuda",
        (candidate,),
        candidate,
        candidate,
        (episode,),
        (session,),
        ProgressFact("exploring", 1, 1, 0, 10, "2026-11-06T00:00:00+00:00", True),
        (),
    )


class ObservabilityScriptsTest(unittest.TestCase):
    def test_runtime_status_reports_scope_episode_and_session(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            scope = Scope(workspace, workspace / "runtime", "campaign", None)
            session = workspace / ".atrex_long_horizon/episodes/e0001/sessions/run-live"
            session.mkdir(parents=True)
            (session / "token-usage.json").write_text(
                '{"session_id":"session-1","backend":"claude","state":"running",'
                '"started_at":"2026-11-06T00:00:00Z"}'
            )
            (session / "conversation.jsonl").write_text(
                '{"type":"provider_event","event":{"type":"assistant","message":{"content":['
                '{"type":"tool_use","id":"tool-1","name":"Bash","input":'
                '{"command":"python3 tools/sandbox.py --kind evaluate"}}]}}}\n'
            )
            with mock.patch("observability.light_runtime_status.discover_scopes", return_value=[scope]), \
                    mock.patch("observability.light_runtime_status.load_snapshot", return_value=public_snapshot(workspace)), \
                    mock.patch("observability.light_runtime_status.search_state", return_value=None):
                value = runtime_snapshot(workspace, stale_after=10**9)
        self.assertEqual(value["scopes"][0]["current_episode"], 1)
        self.assertEqual(value["scopes"][0]["episodes"][0]["category"], "completed")
        self.assertTrue(value["scopes"][0]["sessions"][0]["active"])
        self.assertEqual(value["scopes"][0]["active_sessions"][0]["episode"], 1)
        self.assertEqual(value["scopes"][0]["active_sessions"][0]["phase"], "waiting_agate")

    def test_runtime_session_phase_returns_to_thinking_after_tool_result(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            conversation = Path(temporary) / "conversation.jsonl"
            conversation.write_text(
                '{"type":"provider_event","event":{"message":{"content":['
                '{"type":"tool_use","id":"w1","name":"Bash","input":'
                '{"command":"python tools/sandbox.py wiki-query cache"}}]}}}\n'
                '{"type":"provider_event","event":{"message":{"content":['
                '{"type":"tool_result","tool_use_id":"w1","content":"done"}]}}}\n'
            )
            phase, tool, _ = _conversation_phase(conversation)
        self.assertEqual(phase, "thinking")
        self.assertIsNone(tool)

    def test_runtime_status_reports_framework_baseline_session_before_episode(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "campaign"
            workspace.mkdir()
            scope = Scope(workspace, workspace.parent / "runtime", "campaign", None)
            trace = workspace.parent / ".atrex-session-traces" / workspace.name / "run-live"
            trace.mkdir(parents=True)
            (trace / "token-usage.json").write_text(
                '{"session_id":"baseline-session","backend":"claude","state":"running",'
                f'"context":{{"workspace":"{workspace}"}},'
                '"started_at":"2026-11-06T00:00:00Z"}'
            )
            (trace / "conversation.jsonl").write_text(
                '{"type":"provider_event","event":{"message":{"content":['
                '{"type":"tool_use","id":"tool-1","name":"Bash","input":'
                '{"command":"python tools/sandbox.py --kind run"}}]}}}\n'
            )
            (workspace / "memory").mkdir()
            (workspace / "memory/live.json").write_text(
                '{"phase":"framework_baseline"}'
            )
            public = public_snapshot(workspace)
            public = replace(
                public,
                progress=replace(
                    public.progress,
                    phase="preparing",
                    episode=None,
                    writer_active=False,
                ),
                episodes=(),
                sessions=(),
            )
            with mock.patch("observability.light_runtime_status.discover_scopes", return_value=[scope]), \
                    mock.patch("observability.light_runtime_status.load_snapshot", return_value=public), \
                    mock.patch("observability.light_runtime_status.search_state", return_value=None):
                value = runtime_snapshot(workspace, stale_after=10**9)
        row = value["scopes"][0]
        self.assertEqual(row["active_session_count"], 1)
        self.assertEqual(row["health"], "healthy")
        self.assertEqual(row["active_sessions"][0]["episode"], 0)
        self.assertEqual(row["active_sessions"][0]["stage"], "framework_baseline")
        self.assertEqual(row["active_sessions"][0]["phase"], "waiting_agate")

    def test_performance_status_combines_headlines(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            scope = Scope(workspace, workspace / "runtime", "campaign", None)
            with mock.patch("observability.light_performance.discover_scopes", return_value=[scope]), \
                    mock.patch("observability.light_performance.load_snapshot", return_value=public_snapshot(workspace)), \
                    mock.patch("observability.light_performance.iter_records", return_value=[]), \
                    mock.patch("observability.light_performance.accepted_best", return_value={"latency_us_geomean": 10.0}), \
                    mock.patch("observability.light_performance.best_passing_full_eval", return_value={"latency_us_geomean": 9.0}):
                value = performance_snapshot(workspace)
        self.assertEqual(value["accepted_best"]["latency_us_geomean"], 10.0)
        self.assertEqual(value["episodes"][0]["result"]["state"], "pass")

    def test_per_shape_status_keeps_explicit_partial_passes(self):
        rows = _shape_rows(
            {
                "all_pass": False,
                "latency_us_by_shape": {"s1": 1.0, "s2": 2.0},
                "failures": ["sid=s2: correctness mismatch"],
            },
            "gateway-" + "1" * 32,
            "kernel-" + "2" * 32,
            "t01",
        )
        self.assertEqual({row["shape_id"]: row["correctness"] for row in rows}, {"s1": "pass", "s2": "fail"})

    def test_acceptance_reports_verification_record(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            scope = Scope(workspace, workspace / "runtime", "campaign", None)
            snapshot = public_snapshot(workspace)
            candidate = CandidateFact(1, 1, "b" * 40, "b" * 40, True, None, result(latency=8.0))
            snapshot = replace(snapshot, candidates=(snapshot.candidates[0], candidate))
            attempt = {"episode": 1, "version": 1, "accepted": True, "status": "candidate_ready",
                       "verification": {"gate": "PASS", "candidate_latency_us": 8.0,
                                        "incumbent_latency_us": 10.0,
                                        "gateway_record_id": "gateway-" + "1" * 32}}
            with mock.patch("observability.light_acceptance.discover_scopes", return_value=[scope]), \
                    mock.patch("observability.light_acceptance.load_snapshot", return_value=snapshot), \
                    mock.patch("observability.light_acceptance.attempt_documents", return_value=[attempt]):
                value = acceptance_snapshot(workspace)
        promoted = next(row for row in value["candidates"] if row["version"] == 1)
        self.assertTrue(promoted["accepted"])
        self.assertIn("gateway-" + "1" * 32, promoted["evidence"])
        self.assertEqual(promoted["incumbent_latency_us"], 10.0)
        self.assertEqual(promoted["improvement_pct"], 20.0)

    def test_agate_helpers_classify_and_time(self):
        error = _error({"status": "failed", "error": {"error_class": "infra", "reason": "queue_timeout"}}, None)
        self.assertEqual(error["category"], "infrastructure")
        self.assertEqual(_retry_count({"infra": 2, "device": [1]}), 3)
        timing = _timing({"created_at": "2026-11-06T00:00:00Z", "started_at": "2026-11-06T00:00:02Z", "finished_at": "2026-11-06T00:00:05Z"})
        self.assertEqual(timing, {"queue_seconds": 2.0, "execution_seconds": 3.0, "total_seconds": 5.0})

    def test_agate_batch_progress_distinguishes_reused_shapes(self):
        record = {"response": {"reused": False}}
        jobs = [
            {"submission": 0, "identity": {"kind": "eval", "batch_checkpoint": "b1"}},
            {"submission": 1, "identity": {"kind": "eval"}},
        ]
        with mock.patch("observability.light_agate_status.evaluation_result", return_value={
            "shape_batch_count": 2, "completed_shape_batch_count": 2,
        }):
            value = _batch_progress(record, jobs)
        self.assertEqual(value["reused_shape_batches"], 1)
        self.assertEqual(value["submitted_shape_batches"], 1)

    def test_agate_status_includes_pending_trusted_job(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "campaign"
            workspace.mkdir()
            scope = Scope(workspace, Path(temporary) / "runtime", "campaign", None)
            audit_root = scope.audit_root
            state = audit_root / "trusted-jobs/jobs/checkpoint/state.json"
            state.parent.mkdir(parents=True)
            state.write_text(
                '{"phase":"accepted","submission":1,"retries":{},'
                '"identity":{"kind":"eval"},"process":{"stdout":'
                '"{\\"job_id\\":\\"ev_pending\\",\\"status\\":\\"running\\",'
                '\\"execution_phase\\":\\"backend_queue\\",'
                '\\"phase_elapsed_ms\\":4200}","stderr":"","returncode":0}}'
            )
            with mock.patch("observability.light_agate_status.discover_scopes", return_value=[scope]), \
                    mock.patch("observability.light_agate_status.iter_records", return_value=[]), \
                    mock.patch("observability.light_agate_status.search_state", return_value=None):
                value = agate_snapshot(workspace)
        self.assertEqual(value["summary"]["trusted_operation_counts"], {"evaluate": 1})
        self.assertEqual(value["summary"]["physical_job_count"], 1)
        self.assertEqual(value["pending_jobs"][0]["source"], "trusted")
        self.assertEqual(value["pending_jobs"][0]["job_id"], "ev_pending")
        self.assertEqual(value["pending_jobs"][0]["execution_phase"], "backend_queue")
        self.assertEqual(value["pending_jobs"][0]["pending_seconds"], 4.2)

    def test_token_sessions_read_episode_and_usage(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            path = workspace / ".atrex_long_horizon/episodes/e0003/sessions/run-" / "token-usage.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"session_id":"s","backend":"claude","state":"completed",'
                            '"started_at":"2026-11-06T00:00:00Z","finished_at":"2026-11-06T00:00:02Z",'
                            '"total":{"input_tokens":2,"output_tokens":3,"cache_read_tokens":4,'
                            '"cache_write_tokens":5,"total_tokens":14,"measurement":"exact"}}')
            scope = Scope(workspace, workspace / "runtime", "campaign", None)
            rows = _session_rows(scope)
        self.assertEqual(rows[0]["episode"], 3)
        self.assertEqual(rows[0]["stage"], "episode")
        self.assertEqual(rows[0]["usage"]["total_tokens"], 14)
        self.assertIsNone(rows[0]["usage"]["reasoning_tokens"])
        self.assertEqual(rows[0]["wall_seconds"], 2.0)

    def test_token_sessions_include_framework_baseline_capture(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "campaign"
            workspace.mkdir()
            trace = workspace.parent / ".atrex-session-traces" / workspace.name / "run-baseline"
            trace.mkdir(parents=True)
            (trace / "token-usage.json").write_text(
                '{"session_id":"baseline","backend":"claude","state":"completed",'
                f'"context":{{"workspace":"{workspace}"}},'
                '"started_at":"2026-11-06T00:00:00Z",'
                '"finished_at":"2026-11-06T00:00:05Z",'
                '"total":{"input_tokens":2,"output_tokens":3,"cache_read_tokens":4,'
                '"cache_write_tokens":5,"total_tokens":14,"measurement":"exact"}}'
            )
            (workspace / "memory").mkdir()
            (workspace / "memory/live.json").write_text(
                '{"episode_branch":"framework-baseline","phase":"recorded"}'
            )
            rows = _session_rows(Scope(workspace, workspace.parent / "runtime", "campaign", None))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode"], 0)
        self.assertEqual(rows[0]["stage"], "framework_baseline")
        self.assertEqual(rows[0]["usage"]["total_tokens"], 14)
        self.assertEqual(rows[0]["wall_seconds"], 5.0)

    def test_tool_helpers_parse_kind_and_duration(self):
        self.assertEqual(_tool("gateway", ["--kind", "profile"]), "profile")
        self.assertEqual(_duration({"nested": {"duration_seconds": 1.25}}), 1.25)

    def test_wiki_helpers_extract_and_adopt(self):
        self.assertEqual(_knowledge_ids({"records": {"wiki-a": {}}, "x": {"wiki_id": "wiki-b"}}), {"wiki-a", "wiki-b"})
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary)
            path = workspace / ".atrex_long_horizon/episodes/e0001/episode_runtime/journal.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"experiments":[{"experiment_id":"e1","wiki_query_ids":["q1"],'
                            '"wiki_usage":[{"query_id":"q1","wiki_id":"wiki-a","use":"adopted"}]}]}')
            scope = Scope(workspace, workspace / "runtime", "campaign", None)
            query_ids, wiki_ids, usages = _journal_adoption(scope)
        self.assertEqual(query_ids, {"q1"})
        self.assertEqual(wiki_ids, {"wiki-a"})
        self.assertEqual(usages[0]["experiment_id"], "e1")

    def test_evidence_export_writes_four_artifacts(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "campaign"
            workspace.mkdir()
            output = Path(temporary) / "export"
            store = SimpleNamespace(root=Path(temporary) / "measurements")
            record = {
                "operation": "evaluate", "created_at": "now", "cacheable": True,
                "kernels": {"candidate": {"kernel_id": "kernel-" + "a" * 32}},
                "response": {"exit_code": 0, "stdout": "visible", "stderr": ""},
                "result_digest": "digest",
            }

            def fake_source(located, destination):
                destination.mkdir()
                (destination / "kernel.py").write_text("pass\n")
                return {"output_directory": str(destination), "files": ["kernel.py"]}

            with mock.patch("observability.export_light_evidence._record", return_value=((workspace, store, record), [(workspace, store, record)])), \
                    mock.patch("observability.export_light_evidence.locate_gateway_record", return_value=object()), \
                    mock.patch("observability.export_light_evidence.export_source", side_effect=fake_source), \
                    mock.patch("observability.export_light_evidence.job_state_rows", return_value=[{
                        "path": "state.json", "job": {"job_id": "job-1", "status": "succeeded"},
                        "phase": "terminal", "submission": 1, "retries": {}, "process": {"stdout": "{}"},
                    }]):
                manifest = export_light_evidence.export_evidence(
                    workspace, "gateway-" + "1" * 32, output
                )
            self.assertEqual(manifest["agate_job_ids"], ["job-1"])
            self.assertTrue((output / "kernel/kernel.py").is_file())
            self.assertTrue((output / "agate-jobs.json").is_file())
            self.assertTrue((output / "agent-response.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
