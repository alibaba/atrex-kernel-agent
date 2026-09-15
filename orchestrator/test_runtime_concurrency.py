from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator.supervisor_runtime import RuntimeCapability, SupervisorRuntime, SupervisorRuntimeConfig


class RuntimeConcurrencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runtime = SupervisorRuntime(SupervisorRuntimeConfig(
            repository_root=Path(__file__).resolve().parents[1], hardware="test", agent_sandbox="none", sandbox_timeout=5,
        ))
        self.addCleanup(self.runtime.close)
        self.a = self.capability("a")
        self.b = self.capability("b")

    def capability(self, name):
        root = self.root / name
        root.mkdir()
        cap = RuntimeCapability(name, root, self.root, "claude", 0, root / "evidence")
        self.runtime._capabilities[name] = cap
        return cap

    def spawn(self, action):
        done, failures = threading.Event(), []

        def run():
            try:
                action()
            except BaseException as error:
                failures.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return done, thread, failures

    def test_slow_journal_does_not_block_other_session_authorize_audit_or_journal(self):
        entered, release = threading.Event(), threading.Event()

        def execute(service, request):
            if service.workspace == self.a.workspace:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test journal was not released")
            return {"ok": True}

        with patch("supervisor.journal.SupervisorJournalService.execute", autospec=True, side_effect=execute):
            first = self.spawn(lambda: self.runtime.execute_journal(self.a, {"operation": "episode_report"}))
            try:
                self.assertTrue(entered.wait(2))

                def other_session():
                    self.assertEqual(self.runtime.authorize("b"), self.b)
                    self.runtime._audit(self.b, "gateway", 0, ["run"])
                    result = self.runtime.execute_journal(self.b, {"operation": "directions_list"})
                    self.assertEqual(result["exit_code"], 0)
                    self.runtime.revoke("b")

                other = self.spawn(other_session)
                self.assertTrue(other[0].wait(2), "another workspace blocked behind slow commit")
                self.assertFalse(other[2])
                self.assertFalse(first[0].is_set())
            finally:
                release.set()
                first[1].join(5)
            self.assertFalse(first[2])

    def test_tokens_for_one_scope_share_lock_and_revoke_cancels_queued_mutation(self):
        alias = replace(self.a, token="alias")
        self.runtime._capabilities[alias.token] = alias
        self.assertIs(self.runtime._scope_lock(alias), self.runtime._scope_lock(self.a))
        result = []
        with patch("supervisor.journal.SupervisorJournalService.execute") as execute:
            with self.runtime._scope_lock(self.a):
                queued = self.spawn(lambda: result.append(self.runtime.execute_journal(alias, {"operation": "direction_update"})))
                self.assertFalse(queued[0].wait(.1))
                self.runtime.revoke(alias.token)
            queued[1].join(3)
            self.assertTrue(queued[0].is_set())
            self.assertFalse(queued[2])
            execute.assert_not_called()
        self.assertEqual(json.loads(result[0]["stdout"])["error"]["code"], "invalid_capability")

    def test_slow_revoke_publication_releases_global_map_lock(self):
        entered, release = threading.Event(), threading.Event()

        def publish():
            entered.set()
            release.wait(5)

        self.runtime._workspace_views[self.a.token] = Mock(publish=publish)
        revoke = self.spawn(lambda: self.runtime.revoke(self.a.token))
        try:
            self.assertTrue(entered.wait(2))
            check = self.spawn(lambda: self.assertEqual(self.runtime.authorize("b"), self.b))
            self.assertTrue(check[0].wait(1))
            self.assertFalse(check[2])
        finally:
            release.set()
            revoke[1].join(3)
        self.assertFalse(revoke[2])
