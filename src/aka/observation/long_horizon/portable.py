"""Read declared portable Workflow snapshots, without native-layout discovery."""
import json
from pathlib import Path

from aka.contracts.workflow_export import CampaignRef, snapshot_from_document


class PortableWorkflowReader:
    def read(self, workspace):
        workspace = Path(workspace)
        path = workspace / 'workflow-export.json'
        if path.is_symlink():
            raise ValueError('public Workflow export must not be a symlink')
        return snapshot_from_document(json.loads(path.read_bytes()))

    def discover(self, directory):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            return ()
        candidates = (directory,) if (directory / 'workflow-export.json').is_file() else tuple(
            path for path in sorted(directory.iterdir()) if path.is_dir() and not path.is_symlink()
            and (path / 'workflow-export.json').is_file())
        return tuple(CampaignRef(facts.campaign.id, path, facts.campaign.name)
                     for path in candidates for facts in (self.read(path),))
