"""Pin explicitly selected revisions and import only their local object closure."""
from pathlib import Path
import re

from .history import resolve_revision
from .records import _git, current_revision


def _reference(retention_key):
    if (not isinstance(retention_key, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", retention_key)
            or ".." in retention_key or retention_key.endswith(".")):
        raise ValueError("retention key must be a bounded literal identifier")
    return "refs/aka-retained/" + retention_key


def retain_revision(workspace, revision, *, retention_key):
    reference = _reference(retention_key)
    revision = resolve_revision(workspace, revision)
    previous = _git(workspace, "rev-parse", "--verify", reference, check=False)
    if previous.returncode == 0 and previous.stdout.decode().strip() != revision:
        raise ValueError("retention key is already bound to another revision")
    _git(workspace, "update-ref", reference, revision)


def import_revision(workspace, source_workspace, revision, *, retention_key):
    reference = _reference(retention_key)
    current_revision(workspace)
    source = Path(source_workspace).resolve(strict=True)
    revision = resolve_revision(source, revision)
    previous = _git(workspace, "rev-parse", "--verify", reference, check=False)
    if previous.returncode == 0 and previous.stdout.decode().strip() != revision:
        raise ValueError("retention key is already bound to another revision")
    if Path(workspace).resolve(strict=True) != source:
        _git(workspace, "fetch", "--quiet", "--no-tags", "--no-write-fetch-head", str(source), revision)
    retain_revision(workspace, revision, retention_key=retention_key)


def fork_revision(workspace, revision, destination):
    """Create a new standalone controller from a committed revision and its context."""
    import os
    import tempfile
    from .context import (_decode, _retained, _retain, _revision_manifest,
                          restore_context_projection, _sync_directory)
    source = Path(workspace).resolve(strict=True)
    revision = resolve_revision(source, revision)
    destination = Path(destination)
    if not destination.is_absolute() or destination.resolve() != destination:
        raise ValueError("fork destination must be an absolute path without links or aliases")
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("fork destination must be new and outside its source")
    raw = _revision_manifest(source, revision)
    context = _decode(raw) if raw is not None else None
    retained = {row['sha256']: _retained(source, row['sha256'])
                for row in context['context'].values()} if context is not None else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + '.aka-fork.lock')
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        if destination.exists():
            raise ValueError("fork destination was created by another owner")
        with tempfile.TemporaryDirectory(prefix='.aka-fork-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'repository'
            _git(destination.parent, 'clone', '--quiet', '--no-hardlinks', '--no-checkout', str(source), str(staged))
            _git(staged, 'checkout', '--quiet', '--detach', revision)
            _git(staged, 'remote', 'remove', 'origin')
            _git(staged, 'config', 'user.name', 'GPU Kernel Optimizer')
            _git(staged, 'config', 'user.email', 'gpu-kernel-optimizer@local')
            if context is not None:
                for digest, payload in retained.items():
                    _retain(staged, digest, payload)
                restore_context_projection(staged, revision)
            if current_revision(staged) != revision:
                raise RuntimeError('forked Workspace revision identity changed')
            if destination.exists():
                raise ValueError("fork destination was created during preparation")
            os.rename(staged, destination)
            _sync_directory(destination.parent)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
