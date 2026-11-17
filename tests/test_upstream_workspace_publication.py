"""Upstream publication fixes through the current persistent Light interfaces."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import PackagePath
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from aka.contracts.content import SourceContent, WorkspaceFile
from aka.core.publication import publish_immutable, replace_bytes
from aka.resources import workspace as resources
from aka.resources.snapshot_store import _write_immutable
from aka.task.candidate_workspace.provider import GitCandidateWorkspace


def test_resource_publication_recovers_after_actual_process_death(tmp_path):
    destination = tmp_path.resolve() / 'object'
    script = '''
import os, signal, sys
from pathlib import Path
from aka.core import publication
publication._unlink = lambda *args: os.kill(os.getpid(), signal.SIGKILL)
publication.publish_immutable(Path(sys.argv[1]), b'retained resource')
'''
    child = subprocess.run([sys.executable, '-c', script, str(destination)], capture_output=True)
    assert child.returncode == -signal.SIGKILL, child.stderr
    stage = destination.with_name('.object.aka-publish')
    assert destination.stat().st_nlink == 2
    assert os.path.samefile(destination, stage)
    _write_immutable(destination, b'retained resource')
    assert destination.read_bytes() == b'retained resource'
    assert destination.stat().st_nlink == 1
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert not stage.exists()
    lock = destination.parent / '.aka-publication.lock'
    assert lock.stat().st_size == 0 and stat.S_IMODE(lock.stat().st_mode) == 0o600


@pytest.mark.parametrize('staged,mode', [(b'pay', 0o600), (b'payload', 0o444)])
def test_incomplete_resource_staging_is_recovered(tmp_path, staged, mode):
    root = tmp_path.resolve()
    stage = root / '.object.aka-publish'
    stage.write_bytes(staged)
    stage.chmod(mode)
    _write_immutable(root / 'object', b'payload')
    assert (root / 'object').read_bytes() == b'payload'
    assert not stage.exists()


def test_concurrent_publication_and_unknown_alias_rejection(tmp_path):
    destination = tmp_path.resolve() / 'object'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: publish_immutable(destination, b'payload'), range(12)))
    assert destination.stat().st_nlink == 1
    with pytest.raises(ValueError, match='collision'):
        publish_immutable(destination, b'other')
    alias = destination.with_name('unknown-alias')
    os.link(destination, alias)
    with pytest.raises(ValueError, match='aliases'):
        publish_immutable(destination, b'payload')
    assert alias.exists() and destination.read_bytes() == b'payload'


def test_replace_recovers_only_matching_staging(tmp_path):
    destination = tmp_path.resolve() / 'manifest.json'
    stage = destination.with_name('.manifest.json.aka-replace')
    stage.write_bytes(b'{"state":')
    stage.chmod(0o600)
    replace_bytes(destination, b'{"state":"ready"}')
    assert destination.read_bytes() == b'{"state":"ready"}'
    stage.write_bytes(b'different transaction')
    stage.chmod(0o600)
    with pytest.raises(ValueError, match='intent differs'):
        replace_bytes(destination, b'{"state":"changed"}')
    assert stage.read_bytes() == b'different transaction'
    assert destination.read_bytes() == b'{"state":"ready"}'


@pytest.fixture
def wheel_asset(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    installation = root / 'site-packages'
    entry = PackagePath('aka/resources/reference/fixture.txt')
    asset = installation / entry
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'package resource')
    asset.chmod(0o644)
    os.link(asset, root / 'installer-cache')
    entry.size = asset.stat().st_size
    entry.hash = SimpleNamespace(mode='sha256', value=base64.urlsafe_b64encode(
        hashlib.sha256(asset.read_bytes()).digest()).rstrip(b'=').decode())
    owner = SimpleNamespace(files=[entry], locate_file=lambda item: installation / item)
    monkeypatch.setattr(resources, 'distribution', lambda name: owner)
    return asset


def test_public_install_detaches_record_verified_hardlinked_asset(tmp_path, wheel_asset):
    root = tmp_path.resolve()
    alias = root / 'installation-alias'
    alias.symlink_to(wheel_asset.parent, target_is_directory=True)
    selected = alias / wheel_asset.name
    workspace = root / 'agent'
    owner = GitCandidateWorkspace()
    owner.install_files(workspace, (WorkspaceFile('tools/fixture.txt', source=selected),), immutable=True)
    installed = workspace / 'tools/fixture.txt'
    assert installed.read_bytes() == wheel_asset.read_bytes()
    assert not os.path.samefile(installed, wheel_asset)
    assert installed.stat().st_nlink == 1 and wheel_asset.stat().st_nlink == 2
    assert stat.S_IMODE(installed.stat().st_mode) == 0o644
    assert not list(root.glob('aka-workspace-resources-*'))


def test_modified_wheel_asset_is_rejected_before_installation(tmp_path, wheel_asset):
    wheel_asset.write_bytes(b'altered resource')
    destination = tmp_path.resolve() / 'agent'
    with pytest.raises(ValueError, match='wheel RECORD'):
        GitCandidateWorkspace().install_files(destination,
            (WorkspaceFile('tools/fixture.txt', source=wheel_asset),), immutable=True)
    assert not destination.exists()


def test_persistent_agent_initialization_accepts_parent_alias_and_preserves_git(tmp_path):
    root = tmp_path.resolve()
    real = root / 'real'
    real.mkdir()
    alias = root / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    workspace = alias / 'agent'
    content = SourceContent((WorkspaceFile('kernel.py', text='initial = 1\n'),))
    owner = GitCandidateWorkspace()
    commit = owner.initialize_agent(workspace, content)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=workspace)
    git('branch', 'agent-experiment')
    (workspace / 'kernel.py').write_text('staged = 2\n')
    git('add', 'kernel.py')
    (workspace / 'kernel.py').write_text('uncommitted = 3\n')
    before = (git('show-ref'), git('show', ':kernel.py'), git('status', '--porcelain'))
    assert owner.initialize_agent(workspace, content) == commit
    assert before == (git('show-ref'), git('show', ':kernel.py'), git('status', '--porcelain'))
    assert (workspace / 'kernel.py').read_text() == 'uncommitted = 3\n'
    assert git('ls-files').decode().splitlines() == ['.gitignore', 'kernel.py']
    assert not (workspace / '.atrex_context.json').exists()
    assert not hasattr(owner, 'initialize_kernel') and not hasattr(owner, 'promote')


@pytest.mark.parametrize('link', ['workspace', 'selected-child'])
def test_agent_initialization_rejects_leaf_and_selected_child_symlinks(tmp_path, link):
    root = tmp_path.resolve()
    target = root / 'target'
    target.mkdir()
    workspace = root / 'agent'
    if link == 'workspace':
        workspace.symlink_to(target, target_is_directory=True)
    else:
        workspace.mkdir()
        (target / 'kernel.py').write_text('original\n')
        (workspace / 'kernel.py').symlink_to(target / 'kernel.py')
    with pytest.raises(ValueError, match='symlink'):
        GitCandidateWorkspace().initialize_agent(workspace,
            SourceContent((WorkspaceFile('kernel.py', text='replacement\n'),)))
    assert not (workspace / '.git').exists()
