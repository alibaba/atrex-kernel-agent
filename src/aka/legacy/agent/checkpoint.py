"""Capture closed native sessions; restore only their exact original files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile

from aka.contracts.agent import AgentCheckpointSnapshot

SCHEMA = 'aka.agent-checkpoint.v2'


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def capture_native_session(output, workspace, runtime, session_id, environment):
    """Called by Agent after its invocation has stopped, never by a consumer."""
    if runtime not in {'claude', 'codex'} or not session_id:
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('invalid native session identity')
    target = Path(output) / '.aka-agent-sessions' / (runtime + '-' + _digest(session_id.encode()))
    target.parent.mkdir(parents=True, exist_ok=True)
    files = []
    if runtime == 'codex':
        from .codex_ledger import CodexSessionLedgerObserver, codex_home
        session_root = codex_home(environment).expanduser().resolve()
        path = CodexSessionLedgerObserver(session_root)._find_rollout_path(session_id)
        files.append(path)
    else:
        home = Path(environment.get('CLAUDE_CONFIG_DIR') or Path.home()/'.claude').expanduser().resolve()
        projects = home / 'projects'
        matches = sorted(projects.glob('*/' + session_id + '.jsonl')) if projects.is_dir() else []
        if len(matches) != 1 or not matches[0].is_file():
            raise ValueError('closed Claude session ledger is missing')
        path = matches[0]
        session_root = path.parent.resolve()
        files.append(path)
        family = session_root / session_id
        if family.is_dir():
            files.extend(p for p in family.rglob('*') if p.is_file())
    with tempfile.TemporaryDirectory(prefix='.agent-session-', dir=target.parent) as raw:
        staging = Path(raw)
        rows = []
        for index, path in enumerate(sorted(files)):
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError('native session path contains a symlink')
            if path == session_root or not path.is_relative_to(session_root):
                raise ValueError('native session path is outside its runtime root')
            content = path.read_bytes()
            name = str(index)
            (staging/name).write_bytes(content)
            rows.append({'file': name, 'target': str(path), 'root': str(session_root),
                         'sha256': _digest(content),
                         'mode': stat.S_IMODE(path.stat().st_mode)})
        (staging/'session.json').write_text(json.dumps({'runtime':runtime, 'session_id':session_id,
                                                      'files':rows}, sort_keys=True)+'\n')
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staging, target)


def export_checkpoint(workspace, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Agent checkpoint destination must be new')
    root = Path(workspace)/'.aka-agent-sessions'
    failures = Path(workspace)/'.aka-agent-session-errors'
    if failures.is_dir() and any(failures.iterdir()):
        raise ValueError('one or more native sessions lack resumable content')
    destination.mkdir(parents=True)
    sessions = []
    for directory in sorted(root.iterdir() if root.exists() else ()):
        value = json.loads((directory/'session.json').read_bytes())
        for row in value['files']:
            content = (directory/row['file']).read_bytes()
            if _digest(content) != row['sha256']:
                raise ValueError('retained Agent session content changed')
            name = str(len(sessions))
            (destination/name).write_bytes(content)
            sessions.append({**row, 'file':name, 'runtime':value['runtime'], 'session_id':value['session_id']})
    data = (json.dumps({'schema':SCHEMA,'files':sessions}, sort_keys=True)+'\n').encode()
    (destination/'manifest.json').write_bytes(data)
    snapshot = AgentCheckpointSnapshot(destination, _digest(data))
    validate_checkpoint(snapshot)
    return snapshot


def validate_checkpoint(snapshot, *, check_destinations=False):
    root = Path(snapshot.directory)
    if root.is_symlink() or (root/'manifest.json').is_symlink():
        raise ValueError('Agent checkpoint manifest must be a regular file')
    data = (root/'manifest.json').read_bytes()
    if _digest(data) != snapshot.identity:
        raise ValueError('Agent checkpoint manifest identity changed')
    value = json.loads(data)
    if value['schema'] != SCHEMA:
        raise ValueError('unsupported Agent checkpoint schema')
    expected = {'manifest.json'}
    destinations = set()
    session_roots = {}
    for index, row in enumerate(value['files']):
        if row['file'] != str(index):
            raise ValueError('invalid Agent checkpoint content name')
        expected.add(row['file'])
        path = root/row['file']
        target = Path(row['target'])
        session_root = Path(row['root'])
        session = (row['runtime'], row['session_id'])
        if (row['runtime'] not in {'claude', 'codex'}
                or not isinstance(row['session_id'], str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', row['session_id'])):
            raise ValueError('invalid native session identity')
        if (not session_root.is_absolute() or session_root.resolve() != session_root
                or (session_root.exists() and not session_root.is_dir())):
            raise ValueError('unsafe native session root')
        if session in session_roots and session_roots[session] != session_root:
            raise ValueError('native session declares multiple roots')
        session_roots[session] = session_root
        if (not target.is_absolute() or target.resolve() != target
                or target == session_root or not target.is_relative_to(session_root)):
            raise ValueError('native session destination is outside its runtime root')
        if any(target == other or target.is_relative_to(other) or other.is_relative_to(target)
               for other in destinations):
            raise ValueError('duplicate or overlapping native session destination')
        if type(row['mode']) is not int or row['mode'] < 0 or row['mode'] > 0o777:
            raise ValueError('invalid native session file mode')
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise ValueError('native session parent is not a directory')
        destinations.add(target)
        if path.is_symlink() or _digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Agent checkpoint content changed')
        if check_destinations and target.exists():
            if (not target.is_file() or _digest(target.read_bytes()) != row['sha256']
                    or stat.S_IMODE(target.stat().st_mode) != row['mode']):
                raise ValueError('existing native session differs from checkpoint')
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError('untracked Agent checkpoint content')
    return value


def restore_checkpoint(snapshot):
    value = validate_checkpoint(snapshot, check_destinations=True)
    for row in value['files']:
        target = Path(row['target'])
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((snapshot.directory/row['file']).read_bytes())
        target.chmod(row['mode'])
