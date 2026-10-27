"""Format-independent candidate operations for workflows, Runtime and evaluation.

Only this package interprets source declarations or chooses a file layout. Callers
use capabilities and opaque snapshots; they never infer a format from a filename.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile

from . import native, storage


@dataclass(frozen=True)
class Layout:
    python_entrypoint: str | None
    framework: str | None = None
    typed_only: bool = False
    candidate_metadata: tuple[str, ...] = ()
    replaceable_baseline: bool = True
    archived: bool = False

    def owns(self, name: str) -> bool:
        return name in self.candidate_metadata

    def diagnostic_route(self, kind: str) -> str:
        if self.typed_only:
            if kind != "profile":
                raise ValueError("Compiled manifests support Evaluate and Profile; use Evaluate correctness_only for a build/correctness check")
            return "typed"
        return "dev" if self.archived else "single"


def layout(policy) -> Layout:
    if policy == native.POLICY:
        return Layout(None, "cuda", True, ("solution.json",), archived=True)
    if policy is not None:
        return Layout("kernel.py", replaceable_baseline=False, archived=True)
    return Layout("kernel.py")


def required_layout(framework: str) -> Layout:
    return layout(native.POLICY if framework.strip().lower() == "cuda" else None)


def layout_at(workspace: Path, *, committed: bool = False) -> Layout:
    return layout(storage.policy_at(workspace, committed=committed))


def archive_bytes(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, text in sorted(files.items()):
                storage.relative(name)
                data = text.encode()
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                tar.addfile(member, io.BytesIO(data))
    return output.getvalue()


@dataclass(frozen=True)
class ExecutionSource:
    candidate: str | dict
    archives: dict[str, dict[str, str]]
    requirements: tuple[str, ...] = ()


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
        files = self.files
        if self.contract is not None and self.contract != native.POLICY:
            files[storage.POLICY] = (json.dumps(self.contract, sort_keys=True) + "\n").encode()
        return files

    @property
    def layout(self) -> Layout:
        return layout(self.contract)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.snapshot).hexdigest()

    def install(self, workspace: Path, *, replace: bool = True) -> None:
        storage.install(workspace, self.snapshot, replace=replace)

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
        files = self.files
        if not self.layout.archived:
            return ExecutionSource(self.snapshot.decode(), {})
        if self.contract == native.POLICY:
            value = native.manifest(files['solution.json'])
            contents = {name: files[name].decode() for name in value['source']['files']}
            value['source'] = {'type': 'oss_archive', 'archive': archive,
                'sha256': hashlib.sha256(archive_bytes(contents)).hexdigest(),
                'files': {name: hashlib.sha256(files[name]).hexdigest() for name in contents}}
            return ExecutionSource(value, {archive: contents})
        if self.contract['package_root'] != '.':
            raise ValueError("Eval archives require package_root='.'; no Dev fallback")
        requirements = tuple(v['distribution'] + v.get('version', '') for v in self.contract['runtime_requirements'])
        return ExecutionSource({'archive': archive, 'entry_point': self.layout.python_entrypoint},
                               {archive: {k: v.decode() for k, v in files.items()}}, requirements)

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


def read(workspace: Path) -> Artifact:
    return Artifact(storage.snapshot(workspace))


def from_revision(workspace: Path, revision: str) -> Artifact:
    return Artifact(storage.from_git(workspace, revision))


def review_paths(workspace: Path) -> list[Path]:
    artifact = read(workspace)
    names = set(artifact.files)
    names.update(n for n in (storage.POLICY, 'solution.json') if (workspace / n).is_file())
    return [workspace / n for n in sorted(names)]


def commit_paths(workspace: Path, *, normalize: bool = False) -> list[str]:
    artifact = read(workspace)
    previous = from_revision(workspace, 'HEAD') if normalize else None
    if normalize:
        artifact.install(workspace)
    names = set(artifact.files)
    if previous:
        names.update(previous.files)
    names.update(n for n in (storage.POLICY, 'solution.json') if (workspace / n).is_file())
    return sorted(names)


def pinned_paths(workspace: Path) -> tuple[str, ...]:
    policy = storage.policy_at(workspace, committed=True)
    return (storage.POLICY, *policy['immutable_files']) if policy and not layout(policy).replaceable_baseline else ()


def freeze_for_execution(authority: Path, workspace: Path, sealed: bytes | None) -> bytes | None:
    policy = storage.policy_at(authority, committed=True)
    discovered = Artifact(sealed).contract if sealed is not None else storage.policy_at(workspace)
    if policy is None and layout(discovered).replaceable_baseline:
        policy = discovered
    if sealed is not None:
        if Artifact(sealed).contract != policy:
            raise ValueError('Sealed candidate has a different source contract')
        return sealed
    if policy is not None:
        return storage.snapshot(workspace, policy=policy)
    if (workspace / storage.POLICY).exists() or (workspace / storage.POLICY).is_symlink():
        raise ValueError('This single-file task cannot introduce a source-tree policy')
    return None


def stage_frozen(workspace: Path, source: bytes) -> None:
    (workspace / storage.POLICY).unlink(missing_ok=True)
    storage.install(workspace, source, replace=False)
    storage.install(workspace, source)


def declared_paths(workspace: Path) -> set[str]:
    policy = storage.policy_at(workspace)
    if policy is not None:
        return set(storage.collect(workspace, policy)) | {storage.POLICY}
    path = workspace / 'solution.json'
    if not path.is_file():
        return set()
    value = json.loads(path.read_text())
    entries = value.get('sources', []) if isinstance(value, dict) else []
    if not isinstance(entries, list):
        raise ValueError('Candidate declaration sources must be a list of paths')
    result = set()
    for entry in entries:
        name = entry if isinstance(entry, str) else entry.get('path') if isinstance(entry, dict) else None
        result.add(storage.relative(name))
    return result


def evidence_paths(workspace: Path) -> set[str]:
    """Known candidate evidence, including incomplete initialization attempts."""
    names = {'solution.json', storage.POLICY, layout(None).python_entrypoint}
    try:
        names.update(declared_paths(workspace))
    except (OSError, ValueError):
        # Terminal evidence must still retain a malformed declaration for diagnosis.
        pass
    return {name for name in names if (workspace / name).is_file()}


def typed_limitation(workspace: Path) -> str | None:
    description = layout_at(workspace)
    if description.typed_only:
        read(workspace)  # Validate the complete declared file closure.
        return None
    entry = description.python_entrypoint
    if not (workspace / entry).is_file():
        return 'missing ' + entry
    if description.archived:
        return None
    if declared_paths(workspace) - {entry}:
        return 'candidate declaration includes auxiliary sources'
    try:
        tree = ast.parse((workspace / entry).read_text())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(v.name.split('.')[0] for v in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    local = sorted(n for n in imports if n not in {'input', 'reference', Path(entry).stem}
                   and ((workspace / (n + '.py')).is_file() or (workspace / n / '__init__.py').is_file()))
    return 'candidate imports local auxiliary modules: ' + ', '.join(local) if local else None


def capture_paths(workspace: Path) -> list[Path]:
    try:
        return review_paths(workspace)
    except (OSError, ValueError):
        # Preserve incomplete edits on crash; only complete artifacts can be executed.
        return [p for p in workspace.rglob('*') if p.is_file()
                and (p.relative_to(workspace).as_posix() in {'kernel.py', 'solution.json', storage.POLICY}
                     or storage.source_name(p.relative_to(workspace).as_posix()) and p.suffix in native.SUFFIXES)]


def restore_capture(workspace: Path, captured: Path) -> list[str]:
    artifact = read(captured)
    artifact.install(workspace)
    names = list(artifact.files)
    # Python solution metadata is auxiliary, but still needed to resume a draft.
    if not artifact.layout.owns('solution.json') and (captured / 'solution.json').is_file():
        from aka.light.workspace.files import publish
        publish(workspace, 'solution.json', (captured / 'solution.json').read_bytes())
        names.append('solution.json')
    return names


def seed_baseline_metadata(workspace: Path, framework: str, name: str, *, native_eval: bool) -> None:
    """Seed only Python DSL metadata; compiled manifests are authored from their Skill."""
    path = workspace / 'solution.json'
    language = framework.strip().lower()
    if required_layout(language).typed_only or path.is_file() or not native_eval:
        return
    dependencies = {'triton': ['torch','triton'], 'gluon': ['torch','triton','gluon'],
                    'cutedsl': ['torch','nvidia-cutlass-dsl'], 'flydsl': ['torch','flydsl']}.get(language, ['torch'])
    reference = workspace / 'reference.py'
    symbol = 'Model' if reference.is_file() and 'class Model' in reference.read_text() else 'run'
    value = {'name': f'{name}_v1_{language}',
             'spec': {'languages': ['python', language], 'dependencies': dependencies, 'entry_point': f'kernel.py::{symbol}'},
             'sources': [{'path': 'kernel.py', 'role': 'Self-contained candidate implementation and framework loader.'}],
             'dependency_roles': {v: 'Candidate framework/runtime plumbing; document its non-compute role.' for v in dependencies}}
    path.write_text(json.dumps(value, indent=2) + '\n')


def measurement_paths(workspace: Path, baseline_path: str | None) -> tuple[set[str], dict[str, str]]:
    policy = storage.policy_at(workspace)
    description = layout(policy)
    paths, snapshots = set(), {}
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
                 if 'kernel.py' in inputs or snapshots.get('candidate') else {})
    baseline = request['options']['baseline_path']
    if baseline:
        source = (storage.from_inputs(inputs, prefix=baseline + '/')
                  if baseline + '/' + storage.POLICY in inputs or 'baseline' in snapshots else inputs[baseline])
        artifacts['baseline'] = Artifact(source)
    return artifacts


def language_hints(workspace: Path) -> list[str]:
    path = workspace / 'solution.json'
    value = json.loads(path.read_text()) if path.is_file() else {}
    hints = value.get('languages', [])
    return [str(v) for v in hints] if isinstance(hints, list) else []


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
