"""Candidate lifecycle and wire-contract regressions; no network or GPU required."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from aka.light.workspace.candidate import native
from aka.light.workspace import candidate as sources
from aka.light.evaluation import source_transport


def test_cuda_baseline_does_not_seed_python_manifest(tmp_path):
    from aka.light.workflow.bootstrap import Campaign
    campaign = SimpleNamespace(workspace=tmp_path, framework="cuda", atrex_bench_root="bench",
                               campaign_name="native-rmsnorm")
    Campaign._ensure_framework_baseline_solution_manifest(campaign)
    assert not (tmp_path / "solution.json").exists()
    campaign.framework = "triton"
    Campaign._ensure_framework_baseline_solution_manifest(campaign)
    value = json.loads((tmp_path / "solution.json").read_text())
    assert value["spec"]["languages"] == ["python", "triton"]


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path.resolve()
    value = {"schema_version": 1, "name": "add", "version": "1",
             "source": {"type": "local", "files": ["kernel.cu", "include/offset.h"]},
             "entry_point": {"type": "cuda", "symbol": "launch_add_one",
                 "build": {"toolchain": "torch_extension", "sources": ["kernel.cu"]},
                 "inputs": [{"kind": "tensor", "name": "x", "dtype": "float32"}],
                 "outputs": [{"name": "out", "like": "x"}],
                 "arguments": [{"ref": "x"}, {"ref": "out"}, {"numel": "x"}]}}
    (root / "solution.json").write_text(json.dumps(value))
    (root / "kernel.cu").write_text('#include "include/offset.h"\n')
    (root / "include").mkdir()
    (root / "include/offset.h").write_text('#define OFFSET 1\n')
    (root / "input.py").write_text('def _make_inputs(n=4): return {"x": n}\n')
    (root / "reference.py").write_text('class Model: pass\n')
    (root / "shapes.json").write_text('{"0":{"input_kwargs":{"n":4}}}')
    return root


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def init_git(root):
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    git(root, "add", "solution.json", "kernel.cu", "include/offset.h", "input.py", "reference.py", "shapes.json")
    git(root, "commit", "-m", "baseline")
    return git(root, "rev-parse", "HEAD")


def test_snapshot_restores_all_files_and_native_to_python(workspace):
    sealed = sources.snapshot(workspace)
    (workspace / "include/offset.h").write_text('broken')
    (workspace / "kernel.py").write_text('stale')
    sources.install(workspace, sealed)
    assert sources.snapshot(workspace) == sealed
    assert not (workspace / "kernel.py").exists()
    sources.install(workspace, b'class Model: pass\n')
    assert sources.snapshot(workspace) == b'class Model: pass\n'
    assert not (workspace / 'kernel.cu').exists()
    assert not (workspace / 'include/offset.h').exists()
    assert not (workspace / 'solution.json').exists()
    sources.install(workspace, sealed)
    assert sources.snapshot(workspace) == sealed


def test_restore_replaces_broken_manifest(workspace):
    sealed = sources.snapshot(workspace)
    (workspace / "solution.json").write_text('{')
    sources.install(workspace, sealed)
    assert sources.snapshot(workspace) == sealed


def test_every_header_and_build_change_changes_identity(workspace):
    first = sources.snapshot(workspace)
    (workspace / "include/offset.h").write_text('#define OFFSET 2\n')
    second = sources.snapshot(workspace)
    assert first != second
    value = json.loads((workspace / "solution.json").read_text())
    value['entry_point']['build']['cuda_cflags'] = ['-O3']
    (workspace / "solution.json").write_text(json.dumps(value))
    assert sources.snapshot(workspace) != second


def test_native_git_commit_and_metadata_only_stall(workspace):
    from aka.light.workspace.state import commit_changed_kernel
    base = init_git(workspace)
    (workspace / "include/offset.h").write_text('#define OFFSET 2\n')
    declaration = json.loads((workspace / "solution.json").read_text())
    declaration["entry_point"]["build"]["cuda_cflags"] = ["-O3"]
    (workspace / "solution.json").write_text(json.dumps(declaration))
    sealed = sources.snapshot(workspace)
    handle = SimpleNamespace(path=workspace, base_commit=base, branch='main', episode=1)
    commit = sources.commit_candidate(handle, sealed)
    assert sources.from_git(workspace, commit) == sealed
    assert commit_changed_kernel(workspace, commit)
    (workspace / 'note.txt').write_text('metadata')
    git(workspace, 'add', 'note.txt'); git(workspace, 'commit', '-m', 'metadata')
    assert not commit_changed_kernel(workspace, git(workspace,'rev-parse','HEAD'))


def test_deleted_header_is_removed_from_restored_candidate(workspace):
    initial = sources.snapshot(workspace)
    value = json.loads((workspace/'solution.json').read_text())
    value['source']['files'].remove('include/offset.h')
    (workspace/'solution.json').write_text(json.dumps(value))
    reduced = sources.snapshot(workspace)
    sources.install(workspace, initial)
    sources.install(workspace, reduced)
    assert not (workspace/'include/offset.h').exists()


@pytest.mark.parametrize('path', ['../escape.cu', '/tmp/escape.cu', 'reference.py', 'skills/x.cu', '.agate_dependencies/x.cu'])
def test_manifest_rejects_unsafe_candidate_paths(workspace, path):
    value = json.loads((workspace/'solution.json').read_text())
    value['source']['files'].append(path)
    with pytest.raises(ValueError):
        native.manifest(json.dumps(value))


def test_symlinks_rejected_on_disk_and_in_git(workspace):
    (workspace/'include/offset.h').unlink()
    (workspace/'include/offset.h').symlink_to('../kernel.cu')
    with pytest.raises((OSError, ValueError)):
        sources.snapshot(workspace)
    init_git(workspace)
    with pytest.raises(ValueError, match='link'):
        sources.from_git(workspace, 'HEAD')


def test_eval_abba_profile_transport_preserves_manifest(workspace):
    request = {'spec': {'target_hardware': ['GPU']}, 'abba': {}}
    source_transport.attach(request, workspace)
    source_transport.attach(request, workspace, side='baseline')
    for side in ('candidate', 'baseline'):
        value = request['candidate'] if side=='candidate' else request['abba']['baseline']
        assert value['source']['type'] == 'oss_archive'
        assert value['source']['files']['include/offset.h'] == hashlib.sha256((workspace/'include/offset.h').read_bytes()).hexdigest()
    profile = source_transport.diagnostic_request({'spec': request['spec']}, workspace, 'profile')
    assert source_transport.ARCHIVES in profile and source_transport.DIAGNOSTIC not in profile
    with pytest.raises(ValueError, match='Evaluate and Profile'):
        source_transport.diagnostic_request({}, workspace, 'check')
    uploaded = []
    def reserve(method, path, body, timeout):
        assert body['kind'] == 'profile'
        return {'job_id': 'p1', 'uploads': [{'path': f['path'], 'put_url': 'https://example.com/upload', 'upload_ref': 'ref'} for f in body['files']]}
    wire = source_transport.uploaded_request(profile, 'key', json_call=reserve, upload=lambda u,d: uploaded.append(d), kind='profile')
    assert wire['candidate']['source']['sha256'] == hashlib.sha256(uploaded[0]).hexdigest()
    assert wire['oss_files'][0]['upload_ref'] == 'ref'


def test_non_cuda_python_keeps_original_identity(tmp_path):
    root=tmp_path.resolve()
    (root/'kernel.py').write_bytes(b'class Model: pass\n')
    (root/'solution.json').write_text('{"languages":["triton"]}')
    assert sources.snapshot(root) == b'class Model: pass\n'


def test_agate_skill_is_installed_and_discoverable(tmp_path):
    from aka.light.workspace.assets import link_runtime
    from aka.light.workspace.skill_manifest import SKILL_MANIFEST
    root=tmp_path.resolve()
    link_runtime(root)
    manifest = SKILL_MANIFEST['atrex-gpu-gateway']
    assert all((root/'skills/atrex-gpu-gateway'/f).is_file() for f in manifest.files)
    assert (root/'.agents/skills/atrex-gpu-gateway').resolve() == root/'skills/atrex-gpu-gateway'
    assert not (root/'skills/native-cuda').exists()


def test_native_probe_runs_eval_and_preserves_l2_tolerance(workspace):
    from aka.light.acceptance.typed_numerical import input_source
    from aka.light.evaluation.gateway import build_typed_request
    case={'id':'positive','fields':{'x':{'generator':'constant','value':1}}}
    data=input_source((workspace/'input.py').read_bytes(),case,123)
    compile(data, 'input.py', 'exec')
    assert len(data)<16384
    request=build_typed_request(workspace,'GPU',600,[],['python3','test_kernel.py','--correctness-only','--correctness-max-rel-l2','0.001','--multi-seed','0'],'run')
    assert request['options']['correctness_max_rel_l2']==0.001
    assert request['options']['num_correctness_cases']==1
    assert request['mode']=='correctness_only'
    assert request['candidate']['entry_point']['type']=='cuda'


def test_measurement_record_restore_and_duplicate_isolation(workspace, tmp_path):
    from aka.light.experiment.store import MeasurementStore, DuplicateTask
    store = MeasurementStore(tmp_path.resolve()/'private')
    source = sources.snapshot(workspace)
    identity = store.kernel(source)
    files = sources.unpack(source)[1]
    request = {'operation':'evaluate', 'options': {'source_snapshots': {'candidate': hashlib.sha256(source).hexdigest()}},
               'inputs': {k: hashlib.sha256(v).hexdigest() for k,v in files.items()}}
    with store.reserve(request, {'candidate': identity}) as task:
        record=task.finish({'exit_code':0,'stdout':'ok','stderr':''},cacheable=True)
    assert store.read(record['gateway_record_id'])['kernels']['candidate']==identity
    assert store.read_kernel(identity['kernel_id'])==source
    with pytest.raises(DuplicateTask):
        with store.reserve(request, {'candidate':identity}): pass
    (workspace/'include/offset.h').write_text('#define OFFSET 2\n')
    assert store.kernel(sources.snapshot(workspace))['kernel_id']!=identity['kernel_id']


def test_episode_performance_uses_native_snapshot_without_python(workspace):
    from aka.light.workflow.episode import _latest_complete_episode_performance, EPISODE_EVALUATIONS_PATH
    path = workspace / EPISODE_EVALUATIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 2, 'kernel_sha256': hashlib.sha256(sources.snapshot(workspace)).hexdigest(),
               'result': {'all_pass': True, 'latency_us_geomean': 2.0, 'latency_us_by_shape': {'0': 2.0}}}
    path.write_text(json.dumps(payload) + '\n')
    assert _latest_complete_episode_performance(workspace, expected_shape_ids={'0'})['latency_us_geomean'] == 2.0
    (workspace/'include/offset.h').write_text('#define OFFSET 2\n')
    assert _latest_complete_episode_performance(workspace, expected_shape_ids={'0'}) is None


def test_crash_snapshot_restores_cuda_header_and_manifest(workspace):
    from aka.light.workflow.baseline_progress import _snapshot_after_exit, _atomic_write_json, progress_path, restore_latest_candidate
    original=sources.snapshot(workspace)
    capture=_snapshot_after_exit(workspace,exit_number=1,agent_cli='codex')
    _atomic_write_json(progress_path(workspace),{'latest_snapshot':capture})
    (workspace/'kernel.cu').write_text('// interrupted edit')
    (workspace/'include/offset.h').unlink()
    restored=restore_latest_candidate(workspace)
    assert set(restored)=={'kernel.cu','include/offset.h','solution.json'}
    assert sources.snapshot(workspace)==original


@pytest.mark.parametrize('exit_code,result,status', [
    (0,{'all_pass': True},'passed'),
    (1,{'all_pass':False,'diagnostics':[{'stage':'compile','message':'compiler failed'}]},'needs_validation'),
    (1,None,'needs_validation'),
    (1,{'all_pass':False,'diagnostics':[{'stage':'correctness','check':'outputs','relative_l2':0.1,'message':'mismatch'}]},'needs_repair'),
    (1,{'all_pass':False,'diagnostics':[{'stage':'correctness','message':'input generation failed'}]},'needs_validation'),
])
def test_typed_numerical_probe_failures_are_not_passes(workspace,tmp_path,exit_code,result,status):
    from aka.light.acceptance.typed_numerical import run
    class Campaign:
        def measure_for_acceptance(self,root,argv,**kwargs):
            assert '--kind' in argv and argv[argv.index('--kind')+1]=='run'
            assert not (root/'kernel.py').exists()
            assert sources.snapshot(root)==sources.snapshot(workspace)
            assert b'123' in (root/'input.py').read_bytes()
            return {'exit_code':exit_code,'stdout':'[test_kernel] RESULT_JSON='+json.dumps(result) if result else '', 'stderr':'test'}
    cancel=SimpleNamespace(check=lambda:None,parent=None)
    suite={'world_size':1,'cases':[{'id':'risk','fields':{'x':{'generator':'constant','value':1}}}]}
    schedule=[{'case_id':'risk','shape_ids':['0'],'seeds':[123],'selection_digest':'exact'}]
    output=run(Campaign(),workspace,suite,{'0':{'input_kwargs':{'n':4}}},schedule,tmp_path.resolve(),cancel=cancel,max_rel_l2=0.001)
    assert output['status']==status


def test_file_system_owns_format_knowledge():
    """Prevent another format-specific branch spreading into workflow consumers."""
    import ast
    root = Path(__file__).resolve().parents[1] / 'src/aka/light'
    violations = []
    for path in root.rglob('*.py'):
        if path.is_relative_to(root / 'workspace/candidate'):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in {'aka.light.workspace.candidate.native', 'aka.light.workspace.candidate.storage'}:
                    violations.append(str(path))
                if node.module == 'aka.light.workspace.candidate' and any(n.name in {'native', 'storage'} for n in node.names):
                    violations.append(str(path))
        assert 'agate_cuda' not in path.read_text(), path
    assert violations == []


def test_artifact_roundtrip_and_execution_are_format_independent(workspace, tmp_path):
    from aka.light.workspace.candidate import Artifact, read
    compiled = read(workspace)
    plain = Artifact(b'class Model: pass\n')
    for index, artifact in enumerate([plain, compiled]):
        target = tmp_path / f'copy{index}'
        target.mkdir()
        artifact.install(target)
        assert read(target).snapshot == artifact.snapshot
        assert read(target).digest == artifact.digest
        packaged = artifact.execution_source('source.tar.gz')
        if packaged.archives:
            assert packaged.candidate['source']['type'] == 'oss_archive'
        else:
            assert packaged.candidate == artifact.snapshot.decode()
    assert plain.layout.python_entrypoint == 'kernel.py'
    assert compiled.layout.python_entrypoint is None
    assert not (tmp_path / 'copy1/kernel.py').exists()


def test_staging_preserves_pinned_files_and_blocks_format_downgrade(workspace, tmp_path):
    from aka.light.workspace.candidate import freeze_for_execution, stage_frozen, read
    init_git(workspace)
    original = read(workspace).snapshot
    draft = tmp_path/'draft'; draft.mkdir()
    sources.install(draft, original)
    (draft/'kernel.py').write_text('stale wrapper')
    frozen = freeze_for_execution(workspace, draft, None)
    staged = tmp_path/'staged'; staged.mkdir()
    (staged/'kernel.py').write_text('stale')
    stage_frozen(staged, frozen)
    assert not (staged/'kernel.py').exists()
    assert read(staged).snapshot == original
    (draft/'solution.json').write_text('{')
    # A sealed request uses captured bytes even if the mutable draft changes later.
    assert freeze_for_execution(workspace, draft, original) == original
    with pytest.raises(ValueError, match='different source contract'):
        freeze_for_execution(workspace, draft, b'class Model: pass\n')


def test_baseline_commit_paths_remove_obsolete_wrapper(workspace):
    from aka.light.workspace.candidate import commit_paths, read, from_revision
    original = read(workspace).snapshot
    sources.install(workspace, b'class Model: pass\n')
    git(workspace, 'init', '-b', 'main')
    git(workspace, 'config', 'user.name', 'Test')
    git(workspace, 'config', 'user.email', 'test@example.com')
    git(workspace, 'add', 'kernel.py')
    git(workspace, 'commit', '-m', 'V0')
    for n,v in sources.unpack(original)[1].items():
        (workspace/n).parent.mkdir(parents=True,exist_ok=True);(workspace/n).write_bytes(v)
    staged = commit_paths(workspace, normalize=True)
    assert 'kernel.py' in staged
    assert not (workspace/'kernel.py').exists()
    git(workspace, 'add', '--', *staged)
    git(workspace, 'commit', '-m', 'V1')
    assert from_revision(workspace, 'HEAD').snapshot == original


def test_python_legacy_measurement_and_export_remain_supported(tmp_path):
    from datetime import datetime, timezone
    from aka.light.workspace.candidate import Artifact
    artifact = Artifact(b'class Model: pass\n')
    artifact.install(tmp_path)
    assert artifact.matches_measurement(tmp_path, {'schema_version':1,'timestamp':datetime.now(timezone.utc).isoformat()})
    assert not artifact.matches_measurement(tmp_path, {'schema_version':1,'timestamp':'2000-01-01T00:00:00Z'})
    output=tmp_path/'baseline.py'; assert artifact.export(output)=={}
    assert output.read_bytes()==artifact.snapshot


@pytest.mark.parametrize('compiled', [False, True])
def test_episode_commit_uses_one_file_system_for_all_formats(workspace, compiled):
    from aka.light.workspace.git import EpisodeWorktree
    from aka.light.workspace.candidate import read, from_revision
    if compiled:
        base = init_git(workspace)
        (workspace/'include/offset.h').write_text('#define OFFSET 3\n')
    else:
        sources.install(workspace, b'class Model: pass\n')
        git(workspace, 'init', '-b', 'main')
        git(workspace, 'config', 'user.name', 'Test')
        git(workspace, 'config', 'user.email', 'test@example.com')
        git(workspace, 'add', 'kernel.py', 'input.py', 'reference.py', 'shapes.json')
        git(workspace, 'commit', '-m', 'baseline')
        base = git(workspace, 'rev-parse', 'HEAD')
        (workspace/'kernel.py').write_text('class Model: value=2\n')
    expected = read(workspace).snapshot
    handle = EpisodeWorktree(1, base, 'main', workspace)
    commit = handle.commit_candidate(expected)
    assert handle.validate_candidate(commit)[0] == ''
    assert from_revision(workspace, commit).snapshot == expected
    with pytest.raises(ValueError, match='no source change'):
        EpisodeWorktree(2, commit, 'main', workspace).commit_candidate(expected)


def test_imported_python_tree_keeps_pinned_adapter_and_full_identity(tmp_path):
    from aka.light.workspace.candidate import Artifact, pack, read, freeze_for_execution
    adapter=b'from impl import Model\n'
    contract={'schema_version':1,'source_revision':None,'seed_digest':'sha256:'+'0'*64,
              'package_root':'.','editable_roots':['impl'],
              'immutable_files':{'kernel.py':hashlib.sha256(adapter).hexdigest()},
              'runtime_requirements':[]}
    artifact=Artifact(pack({'kernel.py':adapter,'impl/__init__.py':b'class Model: pass\n'},contract))
    artifact.install(tmp_path)
    git(tmp_path,'init','-b','main');git(tmp_path,'config','user.name','Test');git(tmp_path,'config','user.email','test@example.com')
    git(tmp_path,'add','kernel.py','impl/__init__.py','.atrex_source.json');git(tmp_path,'commit','-m','import')
    assert read(tmp_path).snapshot==artifact.snapshot
    assert artifact.execution_source('tree.tar.gz').candidate['entry_point']=='kernel.py'
    assert not artifact.allows_changes(['kernel.py'])
    assert artifact.allows_changes(['impl/new.py'])
    (tmp_path/'kernel.py').write_text('tampered')
    with pytest.raises(ValueError,match='Fixed source'):
        freeze_for_execution(tmp_path,tmp_path,None)


def test_installed_numerical_assets_resolve_venv_aliases(workspace):
    from aka.light.acceptance.numerical import _sources, _digest
    files, _ = _sources(SimpleNamespace(), workspace)
    assert files['typed_transport.py'] == files['typed_transport.py'].resolve()
    assert _digest(files)


def test_baseline_marker_and_recovery_compare_git_identity(workspace, monkeypatch):
    from aka.light.workflow import bootstrap
    from aka.light.workspace.candidate import read, baseline_marker, workspace_identity
    compiled=read(workspace).snapshot
    sources.install(workspace,b'class Model: pass\n')
    git(workspace,'init','-b','main');git(workspace,'config','user.name','Test');git(workspace,'config','user.email','test@example.com')
    git(workspace,'add','kernel.py');git(workspace,'commit','-m','V0')
    blob=git(workspace,'rev-parse','HEAD:kernel.py')
    assert baseline_marker(workspace,'HEAD')=={'kernel_blob':blob}
    assert workspace_identity(workspace)==blob
    campaign=SimpleNamespace(workspace=workspace)
    monkeypatch.setattr(bootstrap,'load_framework_baseline_progress',lambda _: {'latest_snapshot':{}})
    def restore(root):
        sources.install(root,compiled)
        return ['kernel.cu','solution.json','include/offset.h']
    monkeypatch.setattr(bootstrap,'restore_latest_framework_baseline_candidate',restore)
    bootstrap.Campaign._restore_framework_baseline_candidate(campaign,blob)
    assert read(workspace).snapshot==compiled
