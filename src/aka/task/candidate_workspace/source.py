"""Read or materialize complete source trees without editing Agent Git."""
from pathlib import Path
import hashlib
from aka.contracts.workspace import RevisionContent
from aka.task.source.formats.registry import resolve
from .history import resolve_revision


def read_source(workspace, *, revision=None):
    workspace = Path(workspace).resolve()
    system, policy = resolve(workspace)
    if revision is None:
        files = system.collect(workspace, policy)
    else:
        files = system.from_git(workspace, resolve_revision(workspace, revision), policy)
    return RevisionContent(files)


def source_identity(source):
    # Canonical, length-delimited full-tree identity; no single-file fallback.
    digest = hashlib.sha256()
    for name, content in sorted(source.files.items()):
        if content is None:
            continue
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def install_source(workspace, source):
    """Materialize a validated complete artifact; never stage, commit or reset."""
    workspace = Path(workspace).resolve()
    system, policy = resolve(workspace)
    files = {name: value for name, value in source.files.items() if value is not None}
    system.validate(files, policy)
    previous = system.collect(workspace, policy, validate=False)
    system.install(workspace, files, policy)
    for name in previous.keys() - files.keys():
        if system.editable(policy, name):
            (workspace / name).unlink()
