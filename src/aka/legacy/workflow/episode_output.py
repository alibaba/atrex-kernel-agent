"""Capture terminal attempted bytes before their worktree is removed.

This Workflow-owned output never reads Task's private archive representation.
Only hash-bound evaluator results for the captured source become public facts.
"""
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from aka.contracts.workflow_export import EpisodeCandidateFact, EvaluationFact, ResultFact, _result_from_document

FILENAME = 'public-candidate.json'


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _result(result, expected):
    from .export import _number
    if not isinstance(result, dict) or type(result.get('all_pass')) is not bool:
        raise ValueError('episode evaluation requires an explicit all_pass boolean')
    by_case = result.get('latency_us_by_shape') or {}
    if not isinstance(by_case, dict):
        raise ValueError('episode evaluation case latencies must be an object')
    latencies = {}
    for key, value in by_case.items():
        number = _number(value)
        if number is not None and number < 0:
            raise ValueError('episode evaluation latency must not be negative')
        if number is not None and number > 0:
            latencies[str(key)] = number
    latency = _number(result.get('latency_us_geomean', result.get('latency_us')))
    if latency is not None and latency < 0:
        raise ValueError('episode evaluation latency must not be negative')
    latency = latency if latency else None
    complete = bool(expected) and set(latencies) == set(expected) and latency is not None
    passed = result['all_pass']
    metrics = {name: value for name in ('latency_us_arith_mean', 'speedup_vs_ref_mean',
        'speedup_vs_ref_geomean', 'max_abs_err', 'max_rel_err')
        if (value := _number(result.get(name))) is not None}
    return ResultFact('pass' if passed and complete else ('unmeasured' if passed else 'correctness_failure'),
        'PASS' if passed else 'FAIL', 'PASS' if passed else 'FAIL', complete, latency, latencies,
        tuple(expected), 'episode_candidate', result.get('performance_objective'),
        _number(result.get('performance_score')), metrics, True)


def capture(workspace, episode_dir, *, episode, version, base_revision, expected_case_ids,
            source_files=None, source_digest=None):
    from .episode_evidence import capture as capture_evidence
    capture_evidence(workspace, episode_dir, episode=episode)
    workspace, episode_dir = Path(workspace), Path(episode_dir)
    files = {}
    unavailable = None
    for name in (source_files if source_files is not None else ('kernel.py', 'solution.json')):
        path = workspace/name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            unavailable = 'candidate source is not a regular workspace file'
            break
        if path.is_file():
            files[name] = path.read_bytes()
            if source_files is not None and files[name] != source_files[name]:
                raise ValueError('episode source changed before output capture')
        elif source_files is not None:
            raise ValueError('episode source file disappeared before output capture')
    if 'kernel.py' not in files:
        unavailable = unavailable or 'candidate kernel is missing'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip()
    revision = head
    for name, content in files.items():
        committed = subprocess.run(['git', 'show', f'{head}:{name}'], cwd=workspace, capture_output=True)
        if committed.returncode or committed.stdout != content:
            revision = None
    # A deleted manifest also means the final source differs from HEAD.
    if 'solution.json' not in files and subprocess.run(
            ['git','cat-file','-e',f'{head}:solution.json'],cwd=workspace,capture_output=True).returncode == 0:
        revision = None
    digests = {name:_digest(content) for name,content in files.items()}
    evaluations = []
    ledger = workspace/'.atrex_long_horizon/evaluations.jsonl'
    if not unavailable and ledger.exists():
        if ledger.is_symlink() or not ledger.is_file():
            raise ValueError('episode evaluation ledger is not a regular file')
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row,dict) or type(row.get('schema_version')) is not int:
                raise ValueError('episode evaluation receipt is corrupt')
            if row['schema_version'] < 3:
                continue  # Historical receipts cannot certify both candidate files.
            if (not isinstance(row.get('kernel_sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', row['kernel_sha256'])
                    or 'solution_sha256' not in row
                    or (row['solution_sha256'] is not None and (not isinstance(row['solution_sha256'],str)
                        or not re.fullmatch('[0-9a-f]{64}',row['solution_sha256'])))
                    or not isinstance(row.get('result'),dict)):
                raise ValueError('episode evaluation receipt has invalid content identity')
            if (row.get('kernel_sha256') != (source_digest or digests['kernel.py']) or
                    row.get('solution_sha256') != digests.get('solution.json')):
                continue
            evaluation = EvaluationFact(str(row.get('gateway_kind') or 'run'), row.get('timestamp'),
                digests, _result(row.get('result'),expected_case_ids))
            evaluations.append(asdict(evaluation))
    # The terminal coding session has stopped. Refuse a concurrent source change.
    if not unavailable:
        for name, content in files.items():
            path = workspace/name
            if path.is_symlink() or not path.is_file() or path.read_bytes()!=content:
                raise ValueError('episode source changed during output capture')
        if ('solution.json' not in files and (workspace/'solution.json').exists()) or subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()!=head:
            raise ValueError('episode source identity changed during output capture')
    value = {'schema':1, 'episode':episode, 'version':version, 'base_revision':base_revision,
        'head_revision':head, 'revision':revision, 'unavailable':unavailable,
        'files':{name:{'sha256':digests[name],'content_base64':base64.b64encode(content).decode('ascii')}
                 for name,content in files.items()} if not unavailable else {},
        'evaluations':evaluations}
    episode_dir.mkdir(parents=True,exist_ok=True)
    path = episode_dir/FILENAME
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def read(episode_dir, *, episode, version, expected_case_ids, attempt):
    try:
        return _read(episode_dir, episode=episode, version=version,
                     expected_case_ids=expected_case_ids, attempt=attempt)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('malformed captured episode output') from error


def _read(episode_dir, *, episode, version, expected_case_ids, attempt):
    from .export import document, _artifact
    path = Path(episode_dir)/FILENAME
    if path.is_symlink():
        raise ValueError('episode output must not be a symlink')
    value = document(path)
    if value is None:
        return None, ()  # Older runs did not capture terminal public output.
    if (type(value.get('schema')) is not int or value['schema'] != 1
            or type(value.get('episode')) is not int or value['episode'] != episode
            or type(value.get('version')) is not int or value['version'] != version
            or (attempt.get('base_commit') and value.get('base_revision') != attempt['base_commit'])
            or (attempt.get('episode_head') and value.get('head_revision') != attempt['episode_head'])):
        raise ValueError('episode output differs from recorded attempt identity')
    if value.get('unavailable') is not None:
        if not isinstance(value['unavailable'],str) or not value['unavailable']:
            raise ValueError('invalid unavailable episode reason')
        if value.get('files') or value.get('evaluations'):
            raise ValueError('unavailable episode output contains candidate evidence')
        return None, ()
    for key in ('head_revision','revision','base_revision'):
        if value[key] is not None and (not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{40}',value[key])):
            raise ValueError('invalid captured episode revision')
    if value['revision'] is not None and value['revision'] != value['head_revision']:
        raise ValueError('captured source revision differs from final HEAD')
    if 'kernel.py' not in value['files'] or any(
            not isinstance(name, str) or not name or '\\' in name
            or Path(name).is_absolute() or any(part in {'.', '..'} for part in name.split('/'))
            or any(not part for part in name.split('/')) for name in value['files']):
        raise ValueError('invalid captured episode source files')
    artifacts = []; handles = {}; digests = {}
    for name, row in value['files'].items():
        content = base64.b64decode(row['content_base64'],validate=True)
        if _digest(content) != row['sha256']:
            raise ValueError('captured episode source digest changed')
        identifier = f'e{episode}_final_{name.replace(".","_")}'
        role = 'episode-candidate-manifest' if name=='solution.json' else 'episode-candidate-source'
        artifacts.append(_artifact(identifier,f'episodes/e{episode:04d}/final/{name}',role,content,
                                   'text/x-python' if name=='kernel.py' else 'application/json'))
        handles[name] = identifier; digests[name] = row['sha256']
    evaluations = []
    if not isinstance(value['evaluations'],list):
        raise ValueError('captured episode evaluations must be an array')
    for row in value['evaluations']:
        if row['source_sha256'] != digests:
            raise ValueError('captured episode evaluation differs from final source')
        if not isinstance(row['kind'],str) or not row['kind']:
            raise ValueError('captured episode evaluation has no kind')
        if row['timestamp'] is not None and not isinstance(row['timestamp'],str):
            raise ValueError('invalid captured evaluation timestamp')
        result = dict(row['result']); result['expected_case_ids'] = tuple(result['expected_case_ids'])
        if result['expected_case_ids'] != tuple(expected_case_ids):
            raise ValueError('captured episode evaluation differs from campaign cases')
        evaluations.append(EvaluationFact(**{**row,'result':_result_from_document(result)}))
    return EpisodeCandidateFact(value['head_revision'],value['revision'],value['base_revision'],
                                handles,tuple(evaluations)), tuple(artifacts)
