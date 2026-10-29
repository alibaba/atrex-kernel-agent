"""Capture terminal attempted bytes before their worktree is removed.

This Workflow-owned output never reads Task's private archive representation.
Only hash-bound evaluator results for the captured source become public facts.
"""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

from aka.contracts.workflow_export import EpisodeCandidateFact, EvaluationFact, ResultFact

FILENAME = 'public-candidate.json'


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _evaluation_document(evaluation):
    """Encode Workflow's partial output from validated public facts."""
    result = evaluation.result
    return {
        'kind': evaluation.kind,
        'timestamp': evaluation.timestamp,
        'source_sha256': dict(evaluation.source_sha256),
        'result': {
            'state': result.state, 'correctness': result.correctness,
            'gate': result.gate, 'complete': result.complete,
            'latency_us': result.latency_us,
            'case_latencies_us': dict(result.case_latencies_us),
            'expected_case_ids': list(result.expected_case_ids),
            'subject': result.subject, 'objective': result.objective,
            'score': result.score, 'metrics': dict(result.metrics),
            'case_ids_opaque': result.case_ids_opaque,
        },
    }


def _evaluation_fact(measurement):
    state = ('correctness_failure' if measurement.correctness == 'FAIL' or measurement.gate == 'FAIL'
             else 'pass' if measurement.gate == measurement.correctness == 'PASS' and measurement.complete
             else 'unmeasured')
    result = ResultFact(state, measurement.correctness, measurement.gate,
        measurement.complete, measurement.latency_us, measurement.case_latencies_us,
        measurement.expected_case_ids, 'episode_candidate', measurement.objective,
        measurement.score, measurement.metrics, True)
    return EvaluationFact(measurement.operation, measurement.timestamp,
                          measurement.source_sha256, result)


def capture(workspace, episode_dir, *, episode, version, base_revision, expected_case_ids,
            episode_evidence_reader, episode_evidence_interpreter):
    from .episode_evidence import capture as capture_evidence
    capture_evidence(workspace, episode_dir, episode=episode)
    workspace, episode_dir = Path(workspace), Path(episode_dir)
    files = {}
    unavailable = None
    for name in ('kernel.py', 'solution.json'):
        path = workspace/name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            unavailable = 'candidate source is not a regular workspace file'
            break
        if path.is_file():
            files[name] = path.read_bytes()
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
    if not unavailable:
        if not callable(getattr(episode_evidence_reader, 'read_episode_receipts', None)):
            raise TypeError('selected Experiment does not provide episode evidence reading')
        if not callable(getattr(episode_evidence_interpreter, 'interpret_episode_receipts', None)):
            raise TypeError('selected Evaluation does not provide episode evidence interpretation')
        receipts = episode_evidence_reader.read_episode_receipts(workspace, source_sha256=digests)
        for receipt in receipts:
            # Project each row before reading the next, retaining original error ordering.
            measurements = episode_evidence_interpreter.interpret_episode_receipts(
                (receipt,), expected_case_ids=tuple(expected_case_ids))
            for measurement in measurements:
                evaluations.append(_evaluation_document(_evaluation_fact(measurement)))
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
    if 'kernel.py' not in value['files'] or set(value['files'])-{'kernel.py','solution.json'}:
        raise ValueError('invalid captured episode source files')
    artifacts = []; handles = {}; digests = {}
    for name, row in value['files'].items():
        content = base64.b64decode(row['content_base64'],validate=True)
        if _digest(content) != row['sha256']:
            raise ValueError('captured episode source digest changed')
        identifier = f'e{episode}_final_{name.replace(".","_")}'
        role = 'episode-candidate-source' if name=='kernel.py' else 'episode-candidate-manifest'
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
        evaluations.append(EvaluationFact(**{**row,'result':ResultFact(**result)}))
    return EpisodeCandidateFact(value['head_revision'],value['revision'],value['base_revision'],
                                handles,tuple(evaluations)), tuple(artifacts)
