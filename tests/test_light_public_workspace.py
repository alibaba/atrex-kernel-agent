"""Every persistent lineage receives public inputs and configured knowledge."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from aka.contracts.content import SourceContent, WorkspaceFile
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.contracts.filesystem_access import require_path_access
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.runtime_resources import (
    install_runtime_resources, install_reference_knowledge, reference_knowledge_access,
)


def test_refresh_restores_public_inputs_without_changing_agent_git(tmp_path):
    root = tmp_path.resolve()
    workspace = root / 'agent'
    owner = GitCandidateWorkspace()
    owner.initialize_agent(workspace, SourceContent((WorkspaceFile('kernel.py', text='initial = 1\n'),)))
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=workspace)
    (workspace / 'kernel.py').write_text('staged = 2\n')
    git('add', 'kernel.py')
    (workspace / 'kernel.py').write_text('draft = 3\n')
    before = (git('rev-parse', 'HEAD'), git('diff'), git('diff', '--cached'), git('ls-files'))
    # This models a hidden-shape Task: only the supplied public contract is allowed.
    campaign = SimpleNamespace(candidate_workspace=owner,
        problem=SimpleNamespace(agent=SimpleNamespace(files=(
            WorkspaceFile('metadata.json', text='{}'),
            WorkspaceFile('agent_problem.json', text='{"domain":"public"}'),))),
        problem_provider=SimpleNamespace(runtime_files=lambda problem: ()),
        _module_resources=SimpleNamespace(install=Mock()), agent_reference_paths=())
    install_runtime_resources(campaign, workspace)
    (workspace / 'metadata.json').unlink()
    install_runtime_resources(campaign, workspace)
    assert (workspace / 'metadata.json').read_text() == '{}'
    assert not (workspace / 'shapes.json').exists()
    assert not (workspace / 'input.py').exists()
    assert before == (git('rev-parse', 'HEAD'), git('diff'), git('diff', '--cached'), git('ls-files'))


def test_reference_links_are_readable_and_have_valid_readonly_sandbox_grants(tmp_path):
    from aka.agent.execution.bubblewrap import validate_access
    root = tmp_path.resolve()
    workspace = root / 'agent'
    workspace.mkdir()
    source = root / 'knowledge'
    source.mkdir()
    (source / 'guide.md').write_text('GPU knowledge')
    references = (('nvidia', str(source)),)
    install_reference_knowledge(workspace, references)
    install_reference_knowledge(workspace, references)  # Persistent Session refresh.
    assert (workspace / 'references/nvidia/guide.md').read_text() == 'GPU knowledge'
    assert (workspace / 'gpu-wiki/references/nvidia/guide.md').read_text() == 'GPU knowledge'
    grants = reference_knowledge_access(workspace, references)
    access = FilesystemAccess((FilesystemMount(workspace, workspace, True), *grants.mounts),
                             read_only_paths=grants.read_only_paths)
    home = root / 'sessions/private/home'
    home.mkdir(parents=True)
    _, _, protected, mounts = validate_access(workspace, home, root / 'operator-home', access)
    assert workspace / 'references' in protected
    assert workspace / 'gpu-wiki' in protected
    assert any(m.source == source and not m.writable for m in mounts)
    assert require_path_access(access, source / 'guide.md') == source / 'guide.md'
    with pytest.raises(PermissionError):
        require_path_access(access, source / 'guide.md', write=True)
    with pytest.raises(PermissionError):
        require_path_access(access, workspace / 'references/nvidia', write=True)
    (workspace / 'references/nvidia').unlink()
    (workspace / 'references/nvidia').symlink_to(root)
    with pytest.raises(ValueError, match='link changed'):
        reference_knowledge_access(workspace, references)
    with pytest.raises(ValueError, match='link changed'):
        install_reference_knowledge(workspace, references)


def test_reference_install_does_not_follow_replaced_parent(tmp_path):
    root = tmp_path.resolve()
    workspace = root / 'agent'
    workspace.mkdir()
    outside = root / 'outside'
    outside.mkdir()
    (workspace / 'references').symlink_to(outside)
    with pytest.raises(OSError):
        install_reference_knowledge(workspace, (('nvidia', str(outside)),))
    assert not list(outside.iterdir())


def test_evaluator_source_has_readonly_agent_grant_and_bubblewrap_overlay(tmp_path):
    from aka.agent.execution.bubblewrap import validate_access, mount_path
    from aka.workflow.light.campaign import ManagedCampaign
    root = tmp_path.resolve()
    workspace = root / 'agent'
    bench = workspace / 'atrex-bench'
    bench.mkdir(parents=True)
    (bench / 'timing.py').write_text('sample_count = 20\n')
    (workspace / 'kernel.py').write_text('candidate = 1\n')
    campaign = SimpleNamespace(
        _module_resources=SimpleNamespace(filesystem_access=lambda _: FilesystemAccess()),
        agent_reference_paths=(), agent_read_only_paths=(), private_reference_dir=None,
        problem=SimpleNamespace(agent=SimpleNamespace(files=())),
    )
    access = ManagedCampaign.agent_access(campaign, workspace)
    assert bench in access.read_only_paths
    assert require_path_access(access, bench / 'timing.py') == bench / 'timing.py'
    with pytest.raises(PermissionError):
        require_path_access(access, bench / 'timing.py', write=True)
    assert require_path_access(access, workspace / 'kernel.py', write=True) == workspace / 'kernel.py'
    home = root / 'sessions/private/home'
    home.mkdir(parents=True)
    _, _, protected, _ = validate_access(workspace, home, root / 'operator-home', access)
    argv = []
    for path in protected:
        mount_path(argv, path, path)
    index = argv.index('--ro-bind')
    assert argv[index:index + 3] == ['--ro-bind', str(bench), str(bench)]


def test_plugin_wiki_mount_and_campaign_references_coexist_on_refresh(tmp_path):
    import json
    from aka.resources.plugins import PluginRegistry
    root = tmp_path.resolve()
    plugin = root / 'plugin'
    wiki = plugin / 'wiki'
    wiki.mkdir(parents=True)
    (wiki / 'README.md').write_text('shared knowledge')
    skill = plugin / 'skill'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('knowledge')
    (plugin / 'plugin.json').write_text(json.dumps({
        'id': 'gpu-wiki', 'version': '1.0.0', 'api_version': 1,
        'resources': {'gpu-wiki': 'wiki'}, 'skills': {'wiki': {'path': 'skill'}},
    }))
    registry = PluginRegistry(plugins=(plugin,), resources=(), agent_skills=())
    workspace = root / 'workspace'
    workspace.mkdir()
    source = root / 'nvidia'
    source.mkdir()
    (source / 'kernel.py').write_text('implementation')
    references = (('nvidia', str(source)),)
    for _ in range(2):
        registry.install(workspace)
        install_reference_knowledge(workspace, references)
        registry.filesystem_access(workspace)
        reference_knowledge_access(workspace, references)
    assert not (workspace / 'gpu-wiki').is_symlink()
    assert (workspace / 'gpu-wiki/README.md').read_text() == 'shared knowledge'
    assert (workspace / 'gpu-wiki/references/nvidia/kernel.py').read_text() == 'implementation'
    assert not (wiki / 'references').exists()
