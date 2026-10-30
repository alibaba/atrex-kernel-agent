"""Workflow replay receipts around selected Workspace file transactions."""
import hashlib
from pathlib import Path
import tempfile

from aka.contracts.content import WorkspaceFile
from aka.light.orchestrator.durable_state import durable_write_json
from .search import read_state


def commit_files(workspace, transaction, files, message, *, workspace_provider):
    checksums = {name: hashlib.sha256(data).hexdigest() if data is not None else None
                 for name, data in files.items()}
    try:
        receipt = read_state(transaction)
    except FileNotFoundError:
        if workspace_provider.working_facts(workspace, include_untracked=False).changed_paths:
            raise RuntimeError("Search publication requires a clean controller workspace")
        receipt = {'schema_version':2, 'base':workspace_provider.current_revision(workspace),
                   'target':None, 'files':checksums, 'message':message}
        durable_write_json(transaction, receipt)
    if receipt.get('schema_version') != 2 or receipt['files'] != checksums or receipt['message'] != message:
        raise RuntimeError("Search publication replay differs from its pending transaction")
    current = workspace_provider.current_revision(workspace)
    if receipt['target'] is not None and current != receipt['target']:
        raise RuntimeError("Controller revision changed after search publication")
    if current != receipt['base']:
        facts = workspace_provider.revision_history(workspace, revision=current)[0]
        if (facts.parents != (receipt['base'],) or facts.message != message
                or not set(workspace_provider.revision_changes(workspace, receipt['base'], current)) <= files.keys()
                or dict(workspace_provider.read_revision(workspace,current,list(files)).files) != files):
            raise RuntimeError("Controller revision changed outside search publication")
        if receipt['target'] is not None and receipt['target'] != current:
            raise RuntimeError("Confirmed search revision differs from its receipt")
        receipt['target'] = current
        durable_write_json(transaction,receipt)
        return current
    if not set(workspace_provider.working_facts(workspace, include_untracked=False).changed_paths) <= files.keys():
        raise RuntimeError("Unrelated tracked changes block search publication recovery")
    prior = workspace_provider.read_revision(workspace, receipt['base'], list(files)).files
    working = workspace_provider.read_working(workspace, tuple(files)).files
    if any(working[name] not in (prior[name], data) for name,data in files.items()):
        raise RuntimeError("Unexpected working content blocks search publication recovery")
    with tempfile.TemporaryDirectory(prefix='aka-search-source-') as temporary:
        contributions,deleted=[],[]
        for number,(name,data) in enumerate(files.items()):
            if data is None:
                deleted.append(name)
            else:
                retained=Path(temporary)/str(number);retained.write_bytes(data)
                contributions.append(WorkspaceFile(name,source=retained))
        target=workspace_provider.record(workspace,expected_revision=receipt['base'],
            files=tuple(contributions),deleted_paths=tuple(deleted),message=message)
    receipt['target']=target
    durable_write_json(transaction,receipt)
    return target
