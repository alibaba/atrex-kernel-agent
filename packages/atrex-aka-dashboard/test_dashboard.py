"""Run from the checkout: PYTHONPATH=src:. python -m unittest discover -s packages/atrex-aka-dashboard."""
from __future__ import annotations

import json
import socket
import tempfile
import unittest
from threading import Event
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from aka.bootstrap.profile import resolve_profile
from aka.bootstrap.continuation import capture
from aka.core.boot import boot
from aka.core.errors import BootFailure
from aka.dashboard.demo import create_demo_workspace
from aka.dashboard.reader import WorkspaceReader
from aka.dashboard.startup import DashboardStartup

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "src/aka/dashboard/profiles/dashboard.json"


class WorkspaceFixture:
    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def write(self, relative, value):
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def memory(self, n, score, *, passed=True, **performance):
        return self.write(f"memory/v{n}.json", {
            "version": f"v{n}", "correctness": {"status": "PASS" if passed else "FAIL"},
            "quality_gate": {"result": "PASS" if passed else "FAIL"},
            "performance": {"performance_objective": "shape_speedup_arithmetic_mean",
                            "performance_score": score, "latency_us_geomean": 10 / score,
                            "latency_us_by_shape": {"private-shape-id": 10 / score},
                            "measurement_status": "complete", **performance}})

    def attempt(self, episode, n, *, tokens=100, status="candidate_ready", accepted=True, mode="full"):
        value = {"episode": episode, "version": n, "mode": mode, "status": status,
                 "accepted": accepted, "tokens": tokens, "summary": f"episode {episode}",
                 "verification": {"gate": "PASS" if accepted else "FAIL", "candidate_latency_us": 5,
                                  "incumbent_latency_us": 10, "error": None}}
        self.write(f".atrex_long_horizon/episodes/e{episode:04d}/attempt.json", value)
        return value

    def telemetry(self, episode, n, *, total=100, control=100, measurement="exact", reasons=()):
        terminal = {"input_tokens": total, "output_tokens": 0 if total is not None else None,
                    "cache_read_tokens": None, "cache_write_tokens": None,
                    "total_tokens": total, "measurement": measurement}
        self.write(f".atrex_long_horizon/episodes/e{episode:04d}/telemetry.summary.json", {
            "episode": episode, "version": f"v{n}", "control_tokens": control,
            "measurement": measurement, "reason_codes": list(reasons),
            "phase_tokens": {"terminal_usage": terminal,
                             "phases": {"implementation": {"usage": terminal}},
                             "measurement": "partial", "reason_codes": list(reasons)}})

    def populated(self):
        self.write("framework_baseline.json", {"version": "v1", "framework": "Cuda", "platform": "H20"})
        self.memory(0, 1); self.memory(1, 1.2); self.memory(2, 1.5, measurement_subject="candidate")
        self.memory(3, 9, passed=False, measurement_subject="episode_head")
        first = self.attempt(1, 2, mode="fast")
        second = self.attempt(2, 3, accepted=False, tokens=200)
        self.telemetry(1, 2); self.telemetry(2, 3, total=200, control=200)
        self.write(".atrex_long_horizon/state.json", {"episodes": 2, "accepted": 1, "rejected": 1,
                   "tokens": 300, "usage_receipts": {"invocation-1": 100, "invocation-2": 200},
                   "attempts": [first, second]})
        return self


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = WorkspaceFixture(self.root / "kernel_opt_cuda").populated()
        self.reader = WorkspaceReader(self.root)
        self.identifier = self.reader.campaigns()["campaigns"][0]["id"]

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self):
        return self.reader.campaign(self.identifier)

    def test_canonical_promotion_and_rejection(self):
        self.fixture.memory(3, 9, passed=True, measurement_subject="episode_head")
        value = self.snapshot()
        self.assertEqual(value["canonical"]["incumbent"]["value"], 1.5)
        self.assertEqual([h["incumbent"]["value"] for h in value["canonical"]["history"]], [1, 1.2, 1.5, 1.5])
        self.assertEqual([e["status"] for e in value["episodes"]], ["accepted", "rejected"])
        self.assertEqual(self.reader.episode(self.identifier, 2)["status"], "rejected")
        self.assertNotIn("private-shape-id", json.dumps(value))

    def test_receipts_and_duplicate_archives_do_not_add_tokens(self):
        for _ in range(3):
            value = self.snapshot()
            self.assertEqual(value["tokens"]["total"], 300)
            self.assertEqual(value["tokens"]["receipt_count"], 2)
            self.assertEqual(len(value["episodes"]), 2)

    def test_partial_resume_and_control_mismatch(self):
        self.fixture.telemetry(2, 3, total=150, control=200,
                               reasons=["same_session_resume_usage_semantics_unqualified"])
        value = self.snapshot()
        self.assertEqual(value["tokens"]["total"], 300)
        self.assertEqual(value["tokens"]["measurement"], "partial")
        detail = self.reader.episode(self.identifier, 2)
        self.assertEqual(detail["tokens"]["total"], 150)
        self.assertEqual(detail["tokens"]["control_tokens"], 200)
        self.assertIn("control_token_total_mismatch", detail["tokens"]["reason_codes"])
        self.assertEqual(detail["phases"][0]["usage"]["total_tokens"], None)

    def test_control_mismatch_without_resume_is_partial(self):
        self.fixture.telemetry(1, 2, total=99, control=100)
        value = self.snapshot()
        self.assertEqual(value["tokens"]["total"], 300)
        self.assertEqual(value["tokens"]["measurement"], "partial")
        self.assertIn("control_token_total_mismatch", value["tokens"]["reason_codes"])

    def test_qoder_zero_is_unavailable_and_multitask_is_subtotal(self):
        qoder = WorkspaceFixture(self.root / "kernel_opt_qoder")
        qoder.attempt(1, 2, tokens=0, accepted=False)
        qoder.telemetry(1, 2, total=None, control=0, measurement="unavailable")
        qoder.write(".atrex_long_horizon/state.json", {"tokens": 0})
        listing = self.reader.campaigns()
        self.assertEqual(listing["tokens"]["total"], 300)
        self.assertEqual(listing["tokens"]["measurement"], "partial")
        unknown = next(c for c in listing["campaigns"] if c["name"] == "kernel_opt_qoder")
        self.assertIsNone(unknown["tokens"]["total"])
        self.assertEqual(unknown["tokens"]["measurement"], "unavailable")

    def test_progress_pending_and_worktree_journal_identity(self):
        from long_horizon.journal import append_experiment, initialize
        worktree = self.root / "episode-worktree"
        journal = worktree / ".atrex_long_horizon/journal.json"
        initialize(journal, episode=3, memory_version=4, base_commit="base", branch="episode-3")
        append_experiment(journal, {"name": "real experiment", "result": "improved"},
                          live_path=self.fixture.path / "memory/live.json")
        self.fixture.write(".atrex_long_horizon/active_episode.json", {
            "episode": 3, "memory_version": 4, "worktree": str(worktree), "phase": "exploring", "mode": "full"})
        value = self.snapshot()
        self.assertEqual(value["progress"]["experiment_count"], 1)
        self.assertTrue(value["tokens"]["pending"])
        self.assertEqual(value["tokens"]["total"], 300)
        self.assertEqual(value["budget"]["status"], "not_recorded")
        detail = self.reader.episode(self.identifier, 3)
        self.assertEqual(detail["experiments"][0]["name"], "real experiment")
        journal.unlink()
        detail = self.reader.episode(self.identifier, 3)
        self.assertFalse(detail["detail_available"])
        self.assertEqual(detail["latest_experiment"]["name"], "real experiment")

    def test_live_identity_transition_retains_previous_progress(self):
        self.fixture.write("memory/live.json", {"episode": 3, "version": "v4", "phase": "exploring", "experiment_count": 2})
        self.fixture.write(".atrex_long_horizon/active_episode.json", {"episode": 3, "memory_version": 4, "phase": "exploring"})
        self.assertEqual(self.snapshot()["progress"]["experiment_count"], 2)
        self.fixture.write(".atrex_long_horizon/active_episode.json", {"episode": 4, "memory_version": 5, "phase": "preparing"})
        value = self.snapshot()
        self.assertTrue(value["progress"]["record_updating"])
        self.assertEqual(value["progress"]["episode"], 3)

    def test_invalid_file_keeps_valid_value_then_recovers(self):
        self.assertEqual(self.snapshot()["tokens"]["total"], 300)
        state = self.fixture.path / ".atrex_long_horizon/state.json"
        state.write_text('{"tokens":', encoding="utf-8")
        value = self.snapshot()
        self.assertEqual(value["tokens"]["total"], 300)
        self.assertTrue(any(i["code"] == "record_updating" for i in value["issues"]))
        self.fixture.write(".atrex_long_horizon/state.json", {"tokens": 350})
        self.assertEqual(self.snapshot()["tokens"]["total"], 350)

    def test_older_terminal_never_overrides_activity(self):
        self.fixture.write("trace-retention-manifest.json", {"status": "completed", "generated_at": "2020-01-01T00:00:00Z"})
        self.fixture.write(".atrex_long_horizon/active_episode.json", {"episode": 3, "memory_version": 4, "phase": "exploring"})
        value = self.snapshot()
        self.assertEqual(value["progress"]["phase"], "exploring")
        self.assertTrue(value["progress"]["last_terminal"]["stale"])
        self.assertFalse(value["progress"]["last_terminal"]["current_run_confirmed"])

    def test_outcomes_identity_and_comparability(self):
        for n, status in enumerate(("pivot", "blocked", "interrupted"), 3):
            self.fixture.attempt(n, n + 1, status=status, accepted=False, mode="goal")
        self.fixture.write(".atrex_long_horizon/episodes/e0006/attempt.json", {"episode": 99})
        self.fixture.telemetry(1, 99)
        self.fixture.memory(4, 2, latency_us_by_shape={"different-workload": 5})
        self.fixture.attempt(6, 4)
        value = self.snapshot()
        self.assertEqual(value["episodes"][2:5][0]["status"], "pivot")
        self.assertTrue(any(i["code"] == "telemetry_identity_mismatch" for i in value["issues"]))
        self.assertEqual(value["canonical"]["incumbent"]["series"], 2)

    def test_initializer_and_read_only_discovery(self):
        empty = WorkspaceFixture(self.root / "kernel_opt_empty")
        nested = WorkspaceFixture(empty.path / "kernel_opt_private")
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        listing = self.reader.campaigns()
        self.assertEqual(len(listing["campaigns"]), 2)
        task = next(c for c in listing["campaigns"] if c["name"] == "kernel_opt_empty")
        self.assertIsNone(task["tokens"]["total"])
        self.reader.campaign(task["id"])
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((empty.path / ".atrex_long_horizon").exists())

    def test_real_telemetry_producer_schema(self):
        from long_horizon.models import InvocationObservation
        from long_horizon.telemetry import summarize_episode
        from orchestrator.agent_runtime.model import AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage
        terminal = TokenUsage(70, 30, 0, 0, 100, "exact")
        observation = InvocationObservation(terminal, (
            NormalizedAgentEvent(0, "phase_marker", phase="implementation", action="start"),
            NormalizedAgentEvent(1, "usage_delta", usage=terminal),
            NormalizedAgentEvent(2, "phase_marker", phase="implementation", action="end"),
        ), AgentRuntimeCapabilities(True, True, True, True))
        telemetry = summarize_episode(episode=1, version=2, status="candidate_ready", accepted=True,
                                      control_tokens=100, resume_count=0, invocations=[observation])
        self.fixture.write(".atrex_long_horizon/episodes/e0001/telemetry.summary.json", telemetry)
        detail = self.reader.episode(self.identifier, 1)
        self.assertEqual(detail["tokens"]["measurement"], "exact")
        self.assertEqual(detail["tokens"]["terminal_usage"]["input_tokens"], 70)
        implementation = next(p for p in detail["phases"] if p["phase"] == "implementation")
        self.assertEqual(implementation["usage"]["total_tokens"], 100)

    def test_incomplete_promotion_and_v1_crash_snapshot(self):
        self.fixture.memory(2, 9, expected_shape_count=2)
        value = self.snapshot()
        self.assertEqual(value["canonical"]["incumbent"]["value"], 1.2)
        self.assertIsNone(value["canonical"]["history"][2]["incumbent"])
        self.fixture.write(".atrex_long_horizon/framework_baseline/progress.json", {"status": "interrupted"})
        self.assertEqual(self.snapshot()["progress"]["phase"], "v1_interrupted")

    def test_archived_journal_and_missing_telemetry(self):
        self.fixture.write(".atrex_long_horizon/episodes/e0001/episode_runtime/journal.json", {
            "episode": 1, "memory_version": 2, "experiments": [{"name": "archived experiment", "shapes": {"secret": 123}}]})
        (self.fixture.path / ".atrex_long_horizon/episodes/e0001/telemetry.summary.json").unlink()
        detail = self.reader.episode(self.identifier, 1)
        self.assertEqual(detail["tokens"]["total"], 100)
        self.assertEqual(detail["tokens"]["measurement"], "partial")
        self.assertEqual(detail["experiments"][0]["name"], "archived experiment")
        self.assertNotIn("secret", json.dumps(detail))


class LifecycleAndHTTPTest(unittest.TestCase):
    def test_boot_is_lazy_and_dispose_releases_server(self):
        selection, _ = capture(resolve_profile(PROFILE))
        with patch("aka.dashboard.startup.create_server") as create, patch.object(Path, "iterdir", side_effect=AssertionError("boot scanned workspace")):
            report = boot(selection.composition, tokens=selection.tokens, required_services=selection.required_services)
            startup = report.service("startup")
            create.assert_not_called()
        report.verify_composition()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = WorkspaceFixture(Path(tmp) / "kernel_opt_cuda").populated()
            before = {p: p.read_bytes() for p in fixture.path.rglob("*") if p.is_file()}
            url = startup.start(fixture.path, port=0)
            def read(route):
                with urlopen(url + route, timeout=3) as response:
                    return json.loads(response.read())
            listing = read("api/campaigns")
            self.assertFalse(listing["demo"])
            identifier = listing["campaigns"][0]["id"]
            self.assertEqual(read(f"api/campaigns/{identifier}")["tokens"]["total"], 300)
            self.assertEqual(read(f"api/campaigns/{identifier}/episodes/1")["tokens"]["total"], 100)
            with urlopen(url, timeout=3) as response:
                self.assertIn(b"<title>AKA", response.read())
            for font in ("geist-sans.woff2", "geist-mono.woff2"):
                with urlopen(url + f"assets/{font}", timeout=3) as response:
                    self.assertEqual(response.headers["Content-Type"], "font/woff2")
                    self.assertEqual(response.read(), (ROOT / f"src/aka/dashboard/static/fonts/{font}").read_bytes())
            for route in ("api/campaigns/does-not-exist", f"api/campaigns/{identifier}/episodes/99", "../.git/config", "api/campaigns/../../etc/passwd", "assets/../../reader.py", "assets/unknown.woff2"):
                with self.assertRaises(HTTPError) as error:
                    read(route)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url + "api/campaigns", data=b"{}", method="POST"), timeout=3)
            self.assertEqual(error.exception.code, 405)
            error.exception.close()
            server = startup._server
            report.dispose(); report.dispose(); startup.close()
            self.assertEqual(server.socket.fileno(), -1)
            self.assertEqual(before, {p: p.read_bytes() for p in fixture.path.rglob("*") if p.is_file()})

    def test_demo_uses_real_reader_and_labels_sample_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = create_demo_workspace(Path(tmp))
            startup = DashboardStartup(port=0)
            url = startup.start(workspace, demo=True)
            try:
                with urlopen(url + "api/campaigns", timeout=3) as response:
                    listing = json.loads(response.read())
                self.assertTrue(listing["demo"])
                self.assertEqual(len(listing["campaigns"]), 2)
                self.assertEqual(listing["tokens"]["total"], 1280000)
                cuda = next(c for c in listing["campaigns"] if c["framework"] == "Cuda")
                with urlopen(url + f"api/campaigns/{cuda['id']}", timeout=3) as response:
                    campaign = json.loads(response.read())
                self.assertEqual(campaign["canonical"]["incumbent"]["value"], 1.54)
                self.assertEqual(campaign["tokens"]["measurement"], "partial")
                self.assertTrue(campaign["tokens"]["pending"])
                self.assertEqual(campaign["progress"]["episode"], 9)
                self.assertEqual(campaign["progress"]["experiment_count"], 5)
                self.assertEqual(len(campaign["episodes"]), 8)
                with urlopen(url + f"api/campaigns/{cuda['id']}/episodes/7", timeout=3) as response:
                    detail = json.loads(response.read())
                self.assertEqual(sum(p["usage"]["total_tokens"] for p in detail["phases"]), 190000)
                triton = next(c for c in listing["campaigns"] if c["framework"] == "Triton")
                self.assertIsNone(triton["tokens"]["total"])
                self.assertEqual(triton["tokens"]["measurement"], "unavailable")
            finally:
                startup.close()

    def test_port_conflict_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = DashboardStartup(port=0), DashboardStartup()
            first.start(Path(tmp))
            try:
                with self.assertRaises(OSError):
                    second.start(Path(tmp), port=first._server.server_port)
                with self.assertRaises(ValueError):
                    second.start(Path(tmp) / "missing")
            finally:
                first.close(); second.close()

    def test_configuration_rejects_nonlocal_or_invalid_values(self):
        for config in ({"host": "0.0.0.0"}, {"port": -1}, {"port": 65536}, {"refresh_ms": 0}):
            selection = resolve_profile(PROFILE, patches=[{"id": "dashboard", "config": config}])
            with self.assertRaises(BootFailure):
                boot(selection.composition, tokens=selection.tokens, required_services=selection.required_services)

    def test_close_releases_idle_browser_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            startup = DashboardStartup(port=0)
            startup.start(Path(tmp))
            server = startup._server
            accepted = Event()
            original = server.get_request
            def accept():
                connection, address = original()
                accepted.set()
                return connection, address
            with patch.object(server, "get_request", accept), socket.create_connection(server.server_address, timeout=3) as connection:
                connection.sendall(b"GET /")
                self.assertTrue(accepted.wait(1))
                startup.close()
                self.assertEqual(server.socket.fileno(), -1)
                self.assertEqual(connection.recv(1024), b"")


if __name__ == "__main__":
    unittest.main()
