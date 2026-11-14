"""Stopped, same-location workspace checkpoints including Task-owned Git state."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from contextlib import ExitStack

from aka.contracts.workspace import CandidateHandle, CandidateCheckpointState, CandidateWorkspaceSnapshot

SCHEMA = 'aka.candidate-workspace.snapshot.v1'


def _absolute(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('checkpoint requires an original absolute path without symlink ancestors')
    return path


def _inventory(root):
    entries = {}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            entries[name] = {'kind': 'symlink', 'target': str(path.readlink()), 'mode': mode}
        elif path.is_file():
            entries[name] = {'kind': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'mode': mode}
        elif path.is_dir():
            entries[name] = {'kind': 'directory', 'mode': mode}
        else:
            raise ValueError(f'unsupported workspace checkpoint entry: {name}')
    return entries


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def _state(manifest):
    rows = manifest['candidates']
    handles = tuple(CandidateHandle(row['episode'], row['base_commit'], row['branch'], _absolute(row['path'])) for row in rows)
    return CandidateCheckpointState(_absolute(manifest['workspace']), manifest['revision'], handles,
                                    tuple(row['revision'] for row in rows))


def _inspect(snapshot):
    root = _absolute(snapshot.directory)
    if (root / 'manifest.json').is_symlink():
        raise ValueError('unsafe CandidateWorkspace checkpoint manifest')
    raw = (root / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot.identity:
        raise ValueError('CandidateWorkspace snapshot identity changed')
    manifest = json.loads(raw)
    if manifest['schema'] != SCHEMA:
        raise ValueError('unsupported CandidateWorkspace snapshot schema')
    state = _state(manifest)
    destinations = [state.workspace, *(c.path for c in state.candidates)]
    if len(set(destinations)) != len(destinations) or any(a in b.parents for a in destinations for b in destinations if a != b):
        raise ValueError('overlapping CandidateWorkspace checkpoint destinations')
    if len(manifest['trees']) != len(destinations):
        raise ValueError('incomplete CandidateWorkspace checkpoint')
    allowed = {'manifest.json', 'trees'}
    if {p.name for p in root.iterdir()} != allowed or (root / 'trees').is_symlink():
        raise ValueError('untracked CandidateWorkspace checkpoint entries')
    if {p.name for p in (root / 'trees').iterdir()} != {str(i) for i in range(len(destinations))}:
        raise ValueError('untracked CandidateWorkspace checkpoint trees')
    for index, tree in enumerate(manifest['trees']):
        retained = root / 'trees' / str(index)
        if retained.is_symlink() or not retained.is_dir() or _inventory(retained) != tree['entries']:
            raise ValueError('CandidateWorkspace checkpoint bytes, modes or links changed')
        if stat.S_IMODE(retained.stat().st_mode) != tree['mode']:
            raise ValueError('CandidateWorkspace checkpoint root mode changed')
    git_dir = root / 'trees/0/.git'
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise ValueError('checkpoint requires a standalone canonical workspace Git directory')
    if (git_dir / 'objects/info/alternates').exists():
        raise ValueError('checkpoint cannot depend on an external Git object store')
    if any(p.is_symlink() or p.name.endswith('.lock') for p in git_dir.rglob('*')):
        raise ValueError('Git checkpoint contains a symlink or active Git lock')
    # Resolve objects/refs solely in retained Git, never in the original workspace.
    retained_head = subprocess.check_output(['git', '--git-dir', str(git_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if retained_head != state.revision:
        raise ValueError('canonical workspace revision differs')
    for revision in (state.revision, *(c.base_commit for c in state.candidates), *state.candidate_revisions):
        result = subprocess.check_output(['git', '--git-dir', str(git_dir), 'rev-parse', '--verify', revision + '^{commit}'], text=True).strip()
        if result != revision:
            raise ValueError('checkpoint revision identity does not resolve')
    git_ids = set()
    for index, row in enumerate(manifest['candidates'], start=1):
        link = root / 'trees' / str(index) / '.git'
        expected_prefix = 'gitdir: ' + str(state.workspace / '.git/worktrees') + '/'
        if not link.is_file() or link.is_symlink() or not link.read_text().strip().startswith(expected_prefix):
            raise ValueError('candidate Git association differs from the canonical workspace')
        association = Path(link.read_text().strip()[len('gitdir: '):])
        if association.parent != state.workspace / '.git/worktrees' or '..' in association.parts:
            raise ValueError('unsafe candidate Git association')
        git_id = association.name
        git_ids.add(git_id)
        metadata = git_dir / 'worktrees' / git_id
        if (metadata / 'gitdir').read_text().strip() != str(state.candidates[index - 1].path / '.git'):
            raise ValueError('candidate worktree address differs')
        if (metadata / 'HEAD').read_text().strip() != 'ref: refs/heads/' + row['branch']:
            raise ValueError('candidate branch association differs')
        actual = subprocess.check_output(['git', '--git-dir', str(git_dir), 'rev-parse', '--verify', 'refs/heads/' + row['branch']], text=True).strip()
        if actual != row['revision']:
            raise ValueError('candidate branch revision differs')
    retained_ids = {p.name for p in (git_dir / 'worktrees').iterdir()} if (git_dir / 'worktrees').exists() else set()
    if git_ids != retained_ids:
        raise ValueError('checkpoint must include every registered candidate worktree')
    return manifest, state


def validate_checkpoint(snapshot, *, check_destinations=False):
    """Read-only validation, optionally including every existing target identity."""
    manifest, state = _inspect(snapshot)
    if check_destinations:
        _destinations(manifest, state)
    return state


def checkpoint_files(snapshot, paths):
    """Return validated canonical workspace file bytes without exposing storage layout."""
    _inspect(snapshot)
    root = snapshot.directory / 'trees/0'
    values = {}
    for name in paths:
        relative = Path(name)
        if (not name or relative.is_absolute() or '..' in relative.parts
                or relative.as_posix() != name or '.git' in relative.parts):
            raise ValueError('unsafe checkpoint file request')
        target = root / relative
        if target.resolve() != target.absolute() or not target.is_file():
            raise ValueError(f'checkpoint file is missing or is a symlink: {name}')
        values[name] = target.read_bytes()
    return values


def export_checkpoint(workspace, destination, *, candidates=()):
    """Caller holds the stopped Workflow lease throughout capture."""
    workspace = _absolute(workspace)
    destination = _absolute(destination)
    paths = [workspace, *(_absolute(c.path) for c in candidates)]
    if destination.exists() or any(p == destination or p in destination.parents for p in paths):
        raise ValueError('checkpoint destination must be new and outside captured workspaces')
    if not (workspace / '.git').is_dir() or (workspace / '.git').is_symlink():
        raise ValueError('checkpoint requires a standalone canonical workspace')
    rows = []
    for c in candidates:
        if _git(c.path, 'symbolic-ref', '--short', 'HEAD') != c.branch:
            raise ValueError('active candidate has changed its isolated branch')
        rows.append({**asdict(c), 'path': str(c.path), 'revision': _git(c.path, 'rev-parse', 'HEAD')})
    before = [_inventory(p) for p in paths]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.candidate-export-', dir=destination.parent) as raw:
        staging = Path(raw) / 'snapshot'; (staging / 'trees').mkdir(parents=True)
        trees = []
        for index, path in enumerate(paths):
            target = staging / 'trees' / str(index)
            shutil.copytree(path, target, symlinks=True)
            entries = _inventory(target)
            if entries != before[index] or entries != _inventory(path):
                raise ValueError('workspace changed during stopped checkpoint capture')
            trees.append({'entries': entries, 'mode': stat.S_IMODE(path.stat().st_mode)})
        manifest = {'schema': SCHEMA, 'workspace': str(workspace), 'revision': _git(workspace, 'rev-parse', 'HEAD'),
                    'candidates': rows, 'trees': trees}
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        (staging / 'manifest.json').write_bytes(payload)
        snapshot = CandidateWorkspaceSnapshot(staging, hashlib.sha256(payload).hexdigest())
        _inspect(snapshot)
        staging.rename(destination)
    return CandidateWorkspaceSnapshot(destination, snapshot.identity)


def _destinations(manifest, state):
    paths = [state.workspace, *(c.path for c in state.candidates)]
    missing = []
    for index, path in enumerate(paths):
        if path.exists():
            if not path.is_dir() or path.is_symlink():
                raise ValueError('checkpoint destination is not an ordinary directory')
            actual = _inventory(path)
            if actual:
                tree = manifest['trees'][index]
                if actual != tree['entries'] or stat.S_IMODE(path.stat().st_mode) != tree['mode']:
                    raise ValueError('existing workspace differs from the checkpoint identity')
                continue
        missing.append((index, path))
    return missing


def restore_checkpoint(snapshot):
    """Validate all bytes and existing destinations before restoring original paths."""
    manifest, state = _inspect(snapshot)
    missing = _destinations(manifest, state)
    installed = []
    with ExitStack() as stack:
        staged = []
        for index, path in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.candidate-restore-', dir=path.parent)))
            target = temporary / 'workspace'
            shutil.copytree(snapshot.directory / 'trees' / str(index), target, symlinks=True)
            if _inventory(target) != manifest['trees'][index]['entries']:
                raise ValueError('checkpoint changed during restoration')
            staged.append((target, path))
        try:
            for target, path in staged:
                if path.exists():
                    path.rmdir()  # previously validated empty destination; fail if a writer appeared
                target.rename(path)
                installed.append(path)
        except BaseException:
            for path in reversed(installed):
                shutil.rmtree(path)
            raise
    return state
