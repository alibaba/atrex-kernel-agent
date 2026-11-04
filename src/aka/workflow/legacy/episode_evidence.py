"""Retain declared episode plans and compact profiles before workspace cleanup."""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath

FILENAME = 'public-evidence.json'
PATTERNS = (('plans', '*.md', 'plan'),
            ('profiles', '**/summary.txt', 'profiler-evidence'),
            ('profiles', '**/REPORT.md', 'profiler-evidence'),
            ('profiles', '**/evidence.md', 'profiler-evidence'))


def selected(workspace):
    workspace = Path(workspace).resolve()
    for directory, pattern, role in PATTERNS:
        for path in sorted((workspace / directory).glob(pattern)):
            if path.is_file() and not path.is_symlink() and path.resolve() == path.absolute():
                yield path.relative_to(workspace).as_posix(), role, path.read_bytes()


def capture(workspace, episode_dir, *, episode):
    rows = [{'path':name, 'role':role, 'sha256':hashlib.sha256(content).hexdigest(),
             'content_base64':base64.b64encode(content).decode('ascii')}
            for name, role, content in selected(workspace)]
    # Closed-session capture must not silently mix two observations.
    if rows != [{'path':name, 'role':role, 'sha256':hashlib.sha256(content).hexdigest(),
                 'content_base64':base64.b64encode(content).decode('ascii')}
                for name, role, content in selected(workspace)]:
        raise ValueError('episode evidence changed during capture')
    directory = Path(episode_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'schema':1, 'episode':episode, 'files':rows}, sort_keys=True)+'\n')
    temporary.replace(path)


def read(episode_dir, *, episode):
    from .export import document, _artifact
    path = Path(episode_dir) / FILENAME
    if path.is_symlink():
        raise ValueError('episode evidence must not be a symlink')
    value = document(path)
    if value is None:
        return ()  # Historical runs cannot recover declarations from private archives.
    try:
        if (type(value['schema']) is not int or value['schema'] != 1
                or type(value['episode']) is not int or value['episode'] != episode
                or not isinstance(value['files'], list)):
            raise ValueError('episode evidence identity is invalid')
        artifacts, names = [], set()
        for row in value['files']:
            name = row['path']
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
                    or name in names or len(relative.parts) < 2):
                raise ValueError('episode evidence path is invalid')
            role = row['role']
            allowed = ((role == 'plan' and relative.parts[0] == 'plans'
                        and len(relative.parts) == 2 and relative.suffix == '.md')
                       or (role == 'profiler-evidence' and relative.parts[0] == 'profiles'
                           and relative.name in {'summary.txt','REPORT.md','evidence.md'}))
            if not allowed:
                raise ValueError('episode evidence is not a declared output')
            content = base64.b64decode(row['content_base64'], validate=True)
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('episode evidence digest changed')
            names.add(name)
            filename = f'episodes/e{episode:04d}/{name}'
            artifacts.append(_artifact('output_'+hashlib.sha256(filename.encode()).hexdigest(),
                                       filename, role, content, 'text/plain'))
        return tuple(artifacts)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('malformed captured episode evidence') from error
