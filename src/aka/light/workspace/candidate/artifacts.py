"""Format-independent candidate operations for workflows, Runtime and evaluation.

Only this package interprets source declarations or chooses a file layout. Callers
use capabilities and opaque snapshots; they never infer a format from a filename.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import hashlib
import json
import subprocess

from . import storage
from .common import Layout, ExecutionSource, archive_bytes, SELECTION
from .registry import for_policy, resolve, select, task_selection





def layout(policy) -> Layout:
    return for_policy(policy).layout



def required_layout(framework: str) -> Layout:
    return task_selection(None, framework=framework).layout



def layout_at(workspace: Path, *, committed: bool = False) -> Layout:
    return layout(storage.policy_at(workspace, committed=committed))








@dataclass(frozen=True)
class Artifact:
    snapshot: bytes

    @cached_property
    def _parts(self):
        return storage.unpack(self.snapshot)

    @property
    def contract(self):
        return self._parts[0]

    @property
    def files(self) -> dict[str, bytes]:
        return dict(self._parts[1])

    @property
    def materialized_files(self) -> dict[str, bytes]:
        system = for_policy(self.contract)
        return {**self.files, **system.metadata(self.contract),
                SELECTION: (json.dumps({'name':system.name}, sort_keys=True)+'\n').encode()}


    @property
    def layout(self) -> Layout:
        return layout(self.contract)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.snapshot).hexdigest()

    def install(self, workspace: Path, *, replace: bool = True, bind: bool = False) -> None:
        storage.install(workspace, self.snapshot, replace=replace, bind=bind)

    def export(self, destination: Path) -> dict:
        """Export a historical candidate for inspection or baseline comparison."""
        from aka.light.workspace.files import publish
        if not self.layout.archived:
            publish(destination.parent, destination.name, self.snapshot)
            return {}
        if destination.exists():
            if read(destination).snapshot != self.snapshot:
                raise ValueError("Candidate output directory contains different source; choose a new directory")
        else:
            self.install(destination)
        return {"source_kind": "tree", "directory": str(destination), "file_count": len(self.files)}

    def allows_changes(self, paths: list[str]) -> bool:
        return (storage.allowed_changes(paths, self.contract) if self.layout.archived
                else paths == [self.layout.python_entrypoint])

    def execution_source(self, archive: str) -> ExecutionSource:
        return for_policy(self.contract).execution_source(self.files, self.contract, archive)


    def diagnostic_source(self) -> dict:
        return {'entrypoint': self.layout.python_entrypoint,
                'package_root': self.contract['package_root'],
                'runtime_requirements': self.contract['runtime_requirements']}

    def matches_measurement(self, workspace: Path, record: dict) -> bool:
        if isinstance(record.get('schema_version'), int) and record['schema_version'] >= 2:
            return record.get('kernel_sha256') == self.digest
        # Historical schema v1 only identified a single Python file by mtime.
        if self.layout.archived:
            return False
        from datetime import datetime
        try:
            measured = datetime.fromisoformat(str(record['timestamp']).replace('Z', '+00:00')).timestamp()
            return measured >= (workspace / self.layout.python_entrypoint).stat().st_mtime
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def verify_measurement_inputs(self, request: dict, side: str) -> bool:
        baseline = request['options'].get('baseline_path', '')
        filename = self.layout.python_entrypoint if side == 'candidate' else baseline
        if not self.layout.archived:
            return request['inputs'].get(filename) == self.digest
        prefix = '' if side == 'candidate' else baseline + '/'
        return (request['options'].get('source_snapshots', {}).get(side) == self.digest
                and all(request['inputs'].get(prefix + n) == hashlib.sha256(v).hexdigest()
                        for n, v in self.files.items()))


def read(workspace: Path, *, file_system: str | None = None) -> Artifact:
    if file_system is None:
        return Artifact(storage.snapshot(workspace))
    system, policy = resolve(workspace)
    if system.name != file_system:
        raise ValueError(f'Selected {file_system} file system does not match workspace {system.name}')
    return Artifact(storage.pack(select(file_system).collect(workspace, policy), policy))



def from_revision(workspace: Path, revision: str) -> Artifact:
    return Artifact(storage.from_git(workspace, revision))


def review_paths(workspace: Path) -> list[Path]:
    artifact = read(workspace)
    names = set(artifact.files)
    names.update(n for n in (storage.POLICY, 'solution.json', SELECTION) if (workspace / n).is_file())
    return [workspace / n for n in sorted(names)]


def commit_paths(workspace: Path, *, normalize: bool = False) -> list[str]:
    artifact = read(workspace)
    previous = from_revision(workspace, 'HEAD') if normalize else None
    if normalize:
        artifact.install(workspace)
    names = set(artifact.files)
    if previous:
        names.update(previous.files)
    names.update(n for n in (storage.POLICY, 'solution.json', SELECTION) if (workspace / n).is_file())
    return sorted(names)


def pinned_paths(workspace: Path) -> tuple[str, ...]:
    policy = storage.policy_at(workspace, committed=True)
    names = (storage.POLICY, *policy['immutable_files']) if policy and not layout(policy).replaceable_baseline else ()
    return (*names, SELECTION) if (workspace / SELECTION).is_file() else names



def freeze_for_execution(authority: Path, workspace: Path, sealed: bytes | None) -> bytes | None:
    policy = storage.policy_at(authority, committed=True)
    discovered = Artifact(sealed).contract if sealed is not None else storage.policy_at(workspace)
    if policy is None and not (authority / SELECTION).is_file() and layout(discovered).replaceable_baseline:
        policy = discovered
    if sealed is not None:
        if Artifact(sealed).contract != policy:
            raise ValueError('Sealed candidate has a different source contract')
        return sealed
    if (authority / SELECTION).is_file() and discovered != policy:
        raise ValueError('Candidate file system differs from the selected task file system')
    if policy is not None:
        return storage.snapshot(workspace, policy=policy)
    if (workspace / storage.POLICY).exists() or (workspace / storage.POLICY).is_symlink():
        raise ValueError('This single-file task cannot introduce a source-tree policy')
    return None


def stage_frozen(workspace: Path, source: bytes) -> None:
    (workspace / storage.POLICY).unlink(missing_ok=True)
    storage.install(workspace, source, replace=False)
    storage.install(workspace, source, bind=True)


def declared_paths(workspace: Path) -> set[str]:
    system, policy = resolve(workspace)
    return system.declared_paths(workspace, policy)



def evidence_paths(workspace: Path) -> set[str]:
    try:
        system, policy = resolve(workspace)
        names = system.evidence_paths(workspace, policy)
    except (OSError, ValueError):
        # Retain format selectors and malformed declarations for terminal diagnosis.
        names = {'solution.json', storage.POLICY, 'kernel.py'}
    names.add(SELECTION)
    return {name for name in names if (workspace / name).is_file()}



def typed_limitation(workspace: Path) -> str | None:
    system, policy = resolve(workspace)
    return system.typed_limitation(workspace, policy)


def capture_paths(workspace: Path) -> list[Path]:
    try:
        return review_paths(workspace)
    except (OSError, ValueError):
        return [workspace / name for name in sorted(evidence_paths(workspace))]



def restore_capture(workspace: Path, captured: Path) -> list[str]:
    artifact = read(captured)
    artifact.install(workspace)
    names = list(artifact.files)
    from aka.light.workspace.files import publish
    for name in artifact.layout.auxiliary_metadata:
        if (captured/name).is_file():
            publish(workspace, name, (captured/name).read_bytes())
            names.append(name)
    return names


def seed_baseline_metadata(workspace: Path, framework: str, name: str, *, native_eval: bool) -> None:
    task_selection(None, framework=framework).seed_metadata(workspace, framework, name, native_eval=native_eval)


def measurement_paths(workspace: Path, baseline_path: str | None) -> tuple[set[str], dict[str, str]]:
    policy = storage.policy_at(workspace)
    description = layout(policy)
    paths, snapshots = {SELECTION}, {}
    if description.archived:
        artifact = read(workspace)
        paths.update(artifact.files)
        paths.add(storage.POLICY)
        snapshots['candidate'] = artifact.digest
    else:
        paths.add(description.python_entrypoint)
    if baseline_path:
        baseline = workspace / baseline_path
        if baseline.is_dir():
            other = read(baseline)
            if not description.archived or other.contract != policy:
                raise ValueError('ABBA baseline must carry the same task-owned source contract')
            paths.update(baseline_path + '/' + name for name in other.files)
            paths.add(baseline_path + '/' + storage.POLICY)
            snapshots['baseline'] = other.digest
        else:
            if description.archived:
                raise ValueError('Candidate comparison requires a complete baseline directory')
            paths.add(baseline_path)
    return paths, snapshots


def measurement_artifacts(request: dict, inputs: dict[str, bytes]) -> dict[str, Artifact]:
    snapshots = request['options'].get('source_snapshots', {})
    artifacts = ({'candidate': Artifact(storage.from_inputs(inputs))}
                 if select('single_file').layout.python_entrypoint in inputs or snapshots.get('candidate') else {})
    baseline = request['options']['baseline_path']
    if baseline:
        source = (storage.from_inputs(inputs, prefix=baseline + '/')
                  if baseline + '/' + storage.POLICY in inputs or 'baseline' in snapshots else inputs[baseline])
        artifacts['baseline'] = Artifact(source)
    return artifacts


def language_hints(workspace: Path) -> list[str]:
    system, _ = resolve(workspace)
    return system.language_hints(workspace)


def read_path(path: Path) -> Artifact:
    """Read a full exported candidate, whether stored as one file or a directory."""
    if path.is_dir():
        return read(path)
    from aka.light.agent.session_tail import read_regular_bytes
    source = read_regular_bytes(path, limit=storage.MAX_FILE + 1)
    if len(source) > storage.MAX_FILE:
        raise ValueError("Candidate file exceeds 16 MiB")
    return Artifact(source)


def revision_identity(workspace: Path, revision: str = 'HEAD') -> str:
    artifact = from_revision(workspace, revision)
    if artifact.layout.archived:
        return 'sha256:' + artifact.digest
    from aka.legacy.task.git import git_path_blob
    return git_path_blob(workspace, revision, artifact.layout.python_entrypoint)


def revision_changed(workspace: Path, revision: str) -> bool:
    artifact = from_revision(workspace, revision)
    if not artifact.layout.archived:
        from aka.legacy.task.candidate_workspace.state import commit_changed_kernel
        return commit_changed_kernel(workspace, revision)
    parent = subprocess.run(['git', 'rev-parse', '--verify', f'{revision}^'],
                            cwd=workspace, capture_output=True, text=True)
    return parent.returncode != 0 or artifact.snapshot != from_revision(workspace, parent.stdout.strip()).snapshot


def workspace_identity(workspace: Path) -> str:
    artifact = read(workspace)
    if artifact.layout.archived:
        return 'sha256:' + artifact.digest
    from aka.legacy.task.git import git_worktree_blob
    return git_worktree_blob(workspace, artifact.layout.python_entrypoint)


def baseline_marker(workspace: Path, revision: str) -> dict:
    artifact = from_revision(workspace, revision)
    return ({'kernel_blob': '', 'source_sha256': artifact.digest} if artifact.layout.archived
            else {'kernel_blob': revision_identity(workspace, revision)})
