"""Initial Git/source preparation does not require a GPU or measured baseline."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from aka.contracts.content import SourceContent, WorkspaceFile
from aka.experiment.measurement_store import MeasurementStore
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.task.source.formats import Artifact, from_git, pack
from aka.task.source.formats.source_tree import default_tree
from aka.workflow.light.initial_source import initialize_source, initial_kernel, prepare_campaign
from aka.workflow.light.support.paths import initialize_campaign_root, supervisor_measurement_root


class InitialSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve() / "campaign"
        policy, files = default_tree({"kernel.py": b"def kernel(): return None\n"})
        files["impl/example.cu"] = b"// initial source\n"
        lock = {"lock_version": 2, "profile": "test", "variables_digest": "test",
                "core_version": "test", "core_digest": "test", "tokens": [], "entries": []}
        initialize_campaign_root(root, name="test", launch_composition=lock,
                                 source={"file_system": "source_tree", "contract": policy})
        self.campaign = SimpleNamespace(
            workspace=root / "lineages/t0001/agent-workspace",
            candidate_workspace=GitCandidateWorkspace(),
            _selected_initial_source=pack(files, policy),
            _prepared_initial_content=SourceContent(tuple(
                WorkspaceFile(name, text=value.decode()) for name, value in files.items())),
            _link_runtime=Mock(),
        )
        self.root = root
        self.campaign.initialize_source = lambda: initialize_source(self.campaign)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.campaign.workspace, text=True).strip()

    def test_initial_source_archived_without_evaluate_or_fake_latency(self):
        prepare_campaign(self.campaign)
        store = MeasurementStore(supervisor_measurement_root(self.campaign.workspace))
        commit = self.git("rev-parse", "HEAD")
        source = Artifact(store.read_kernel(commit))
        self.assertEqual(source.snapshot, self.campaign._selected_initial_source)
        self.assertEqual(from_git(self.campaign.workspace, commit), source.snapshot)
        self.assertEqual(set(self.git("ls-files").splitlines()), {".gitignore", "kernel.py", "impl/example.cu"})
        self.assertEqual(store.record_ids(), [])
        manifest = json.loads((self.root / "campaign.json").read_text())
        self.assertEqual(manifest["v0"]["kernel_commit_id"], commit)
        self.assertNotIn("latency", json.dumps(manifest))
        self.assertFalse((self.root / ".git").exists())
        self.assertFalse(list(self.root.rglob(".atrex_source.json")))
        self.assertFalse(list(self.root.rglob("source-contract.json")))

    def test_resume_preserves_head_staged_and_uncommitted_source(self):
        prepare_campaign(self.campaign)
        self.git("checkout", "-b", "experiment")
        (self.campaign.workspace / "kernel.py").write_text("# new experiment\n")
        self.git("add", "kernel.py")
        self.git("commit", "-m", "Agent experiment")
        (self.campaign.workspace / "kernel.py").write_text("# staged experiment\n")
        self.git("add", "kernel.py")
        (self.campaign.workspace / "kernel.py").write_text("# unfinished draft\n")
        before = (self.git("rev-parse", "HEAD"), self.git("status", "--porcelain"),
                  self.git("diff", "--cached"), self.git("diff"))
        self.campaign.initialize_source = Mock(side_effect=AssertionError("must not reinitialize"))
        prepare_campaign(self.campaign)
        after = (self.git("rev-parse", "HEAD"), self.git("status", "--porcelain"),
                 self.git("diff", "--cached"), self.git("diff"))
        self.assertEqual(before, after)
        self.campaign._link_runtime.assert_called_once()
        self.assertEqual(self.git("branch", "--show-current"), "experiment")

    def test_public_task_reads_complete_tree_without_agent_contract_files(self):
        prepare_campaign(self.campaign)
        owner = self.campaign.candidate_workspace
        commit = self.git("rev-parse", "HEAD")
        initial = owner.read_source(self.campaign.workspace, revision=commit)
        self.assertEqual(set(initial.files), {"kernel.py", "impl/example.cu"})
        self.assertEqual(initial, owner.read_source(self.campaign.workspace))
        (self.campaign.workspace / "impl/example.cu").write_text("// new implementation\n")
        working = owner.read_source(self.campaign.workspace)
        self.assertNotEqual(owner.source_identity(initial), owner.source_identity(working))
        self.assertEqual(owner.read_source(self.campaign.workspace, revision=commit), initial)
        self.assertEqual(self.git("rev-parse", "HEAD"), commit)

    def test_pooled_seed_stays_at_initial_commit_without_performance(self):
        prepare_campaign(self.campaign)
        initial = initial_kernel(self.campaign)
        (self.campaign.workspace / "kernel.py").write_text("# Agent changes source\n")
        self.git("add", "kernel.py")
        self.git("commit", "-m", "New candidate")
        self.assertEqual(initial_kernel(self.campaign), initial)
        self.assertIsNone(initial["latency_us_geomean"])
        self.assertEqual(initial["record"], {})
        self.assertNotEqual(initial["commit"], self.git("rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
