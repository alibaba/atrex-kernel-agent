"""Detached Workflow facts and content-bound public outputs (schema 3).

Missing measurements remain None. Episode outcome and measurement completeness
are independent. No private memory, journal or ledger dictionaries cross here.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from aka.core.keys import ServiceKey

EXPORT_VERSION = 3
WORKFLOW_EXPORT = ServiceKey('workflow_export', 'WorkflowExportReader', module='aka.contracts.workflow_export')
ResultState = Literal['unmeasured', 'running', 'pass', 'correctness_failure',
                      'infrastructure_error', 'rejected', 'interrupted']


def _freeze(value):
    """Copy public JSON data recursively before freezing it."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('public mapping keys must be strings')
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('public numbers must be finite')
        return value
    raise ValueError('public facts must contain JSON data')


def _number(value, *, positive=False, optional=False):
    if optional and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('invalid public measurement')


def _counter(value, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not int or value < 0:
        raise ValueError('invalid public counter')


def _flag(value):
    if type(value) is not bool:
        raise ValueError('invalid public boolean')


def _frozen_fields(instance, *names):
    for name in names:
        object.__setattr__(instance, name, _freeze(getattr(instance, name)))


def _usage_summary(value):
    measurement = value.get('measurement', 'unavailable')
    if measurement not in ('exact', 'partial', 'unavailable'):
        raise ValueError('invalid public usage measurement')
    total = value.get('total')
    _counter(total, optional=True)
    if (measurement == 'unavailable') != (total is None):
        raise ValueError('public usage total contradicts its measurement')


def _document(value):
    if is_dataclass(value):
        return {item.name: _document(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_document(item) for item in value]
    return value


@dataclass(frozen=True)
class ResultFact:
    state: ResultState
    correctness: str | None
    gate: str | None
    complete: bool
    latency_us: float | None
    case_latencies_us: Mapping[str, float]
    expected_case_ids: tuple[str, ...]
    subject: str | None
    objective: str | None
    score: float | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    case_ids_opaque: bool = True

    def __post_init__(self):
        if (not isinstance(self.case_latencies_us, Mapping) or not isinstance(self.metrics, Mapping)
                or not isinstance(self.expected_case_ids, (tuple, list))):
            raise ValueError('invalid public measurement containers')
        _frozen_fields(self, 'case_latencies_us', 'expected_case_ids', 'metrics')
        expected = self.expected_case_ids
        if len(set(expected)) != len(expected) or any(not isinstance(key, str) or not key for key in expected):
            raise ValueError('invalid public case identities')
        if self.state not in ('unmeasured', 'running', 'pass', 'correctness_failure',
                              'infrastructure_error', 'rejected', 'interrupted'):
            raise ValueError('unsupported public result state')
        _number(self.latency_us, positive=True, optional=True)
        _number(self.score, optional=True)
        for key, value in self.case_latencies_us.items():
            if not key:
                raise ValueError('invalid public case identity')
            _number(value, positive=True)
        for value in self.metrics.values():
            _number(value)
        _flag(self.complete)
        _flag(self.case_ids_opaque)
        if self.complete and (not expected or set(self.case_latencies_us) != set(expected) or self.latency_us is None):
            raise ValueError('complete result lacks required measurements')
        if self.state == 'pass' and (not self.complete or self.correctness != 'PASS' or self.gate != 'PASS'):
            raise ValueError('passing result lacks complete verification')


@dataclass(frozen=True)
class CandidateFact:
    version: int
    episode: int | None
    revision: str | None
    record_revision: str | None
    promoted: bool
    timestamp: str | None
    result: ResultFact
    summary: str | None = None
    comparability: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _counter(self.version)
        _counter(self.episode, optional=True)
        _flag(self.promoted)
        _frozen_fields(self, 'comparability')
        if not isinstance(self.result, ResultFact):
            raise ValueError('candidate lacks a public result')


@dataclass(frozen=True)
class EvaluationFact:
    kind: str
    timestamp: str | None
    source_sha256: Mapping[str, str]
    result: ResultFact

    def __post_init__(self):
        _frozen_fields(self, 'source_sha256')
        if not isinstance(self.result, ResultFact):
            raise ValueError('evaluation lacks a public result')
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError('episode evaluation has no kind')
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValueError('invalid episode evaluation timestamp')


@dataclass(frozen=True)
class EpisodeCandidateFact:
    head_revision: str | None
    revision: str | None
    base_revision: str | None
    artifacts: Mapping[str, str]
    evaluations: tuple[EvaluationFact, ...] = ()

    def __post_init__(self):
        _frozen_fields(self, 'artifacts')
        object.__setattr__(self, 'evaluations', tuple(self.evaluations))
        for revision in (self.head_revision, self.revision, self.base_revision):
            if revision is not None and (not isinstance(revision, str) or not revision):
                raise ValueError('invalid episode candidate revision')
        if self.revision is not None and self.revision != self.head_revision:
            raise ValueError('episode source revision differs from captured HEAD')
        if not {'kernel.py', 'solution.json'}.intersection(self.artifacts) or any(
                not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/")) for name in self.artifacts):
            raise ValueError('invalid final episode source handles')
        if any(not isinstance(item, EvaluationFact) for item in self.evaluations):
            raise ValueError('invalid public episode evaluation')


@dataclass(frozen=True)
class EpisodeFact:
    episode: int
    version: int | None
    status: str
    accepted: bool
    candidate_revision: str | None
    session_id: str | None
    summary: str | None
    gate: str | None
    experiments: tuple[dict, ...] = ()
    phases: tuple[dict, ...] = ()
    mode: str | None = None
    violation: str | None = None
    final_candidate: EpisodeCandidateFact | None = None
    usage: Mapping = field(default_factory=dict)
    verification: Mapping = field(default_factory=dict)
    detail_available: bool = False

    def __post_init__(self):
        _counter(self.episode)
        _counter(self.version, optional=True)
        _flag(self.accepted)
        _frozen_fields(self, 'experiments', 'phases', 'usage', 'verification')
        _flag(self.detail_available)
        for key in ('total', 'control_tokens', 'raw_control_tokens'):
            _counter(self.usage.get(key), optional=True)
        _usage_summary(self.usage)
        for value in (self.mode, self.violation):
            if value is not None and not isinstance(value, str):
                raise ValueError('invalid episode mode or violation')


@dataclass(frozen=True)
class SessionFact:
    session_id: str
    runtime: str
    episode: int | None
    status: str
    updated_at: str | None
    usage: Mapping[str, int | str | None]
    events: tuple[dict, ...] = ()
    resume_supported: bool = False

    def __post_init__(self):
        _counter(self.episode, optional=True)
        _flag(self.resume_supported)
        if not isinstance(self.usage, Mapping):
            raise ValueError('invalid public usage')
        _frozen_fields(self, 'usage', 'events')
        for key, value in self.usage.items():
            if key.endswith('_tokens'):
                _counter(value, optional=True)


@dataclass(frozen=True)
class ProgressFact:
    phase: str
    episode: int | None
    version: int | None
    completed_episodes: int | None
    observed_tokens: int | None
    updated_at: str | None
    writer_active: bool
    mode: str | None = None
    experiment_count: int | None = None
    latest_experiment: Mapping = field(default_factory=dict)
    record_updating: bool = False
    pending_tokens: bool = False
    last_terminal: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for value in (self.episode, self.version, self.completed_episodes, self.observed_tokens):
            _counter(value, optional=True)
        _flag(self.writer_active)
        _counter(self.experiment_count, optional=True)
        _flag(self.record_updating)
        _flag(self.pending_tokens)
        _frozen_fields(self, 'latest_experiment', 'last_terminal')


@dataclass(frozen=True)
class CampaignRef:
    id: str
    workspace: Path
    name: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('public campaign has no identity')
        object.__setattr__(self, 'workspace', Path(self.workspace))


@dataclass(frozen=True)
class ExportArtifact:
    id: str
    filename: str
    content_type: str
    role: str
    sha256: str
    visibility: Literal['public']
    content: bytes = field(repr=False)

    def __post_init__(self):
        from pathlib import PurePosixPath
        filename = PurePosixPath(self.filename)
        if filename.is_absolute() or '..' in filename.parts or not filename.parts or str(filename) != self.filename:
            raise ValueError('invalid public artifact filename')
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('invalid public artifact identity')
        if self.visibility != 'public':
            raise ValueError('non-public artifact in public snapshot')
        if not isinstance(self.content, bytes):
            raise ValueError('public artifact content must be bytes')
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise ValueError('public artifact content identity changed')

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class WorkflowExportSnapshot:
    schema_version: int
    campaign: CampaignRef
    framework: str
    candidates: tuple[CandidateFact, ...]
    baseline: CandidateFact | None
    incumbent: CandidateFact | None
    episodes: tuple[EpisodeFact, ...]
    sessions: tuple[SessionFact, ...]
    progress: ProgressFact
    artifacts: tuple[ExportArtifact, ...]
    warnings: tuple[dict[str, str | bool | None], ...] = ()
    hardware: Mapping[str, str] = field(default_factory=dict)
    token_summary: Mapping = field(default_factory=dict)
    counts: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != EXPORT_VERSION:
            raise ValueError('unsupported public Workflow snapshot')
        if not isinstance(self.campaign, CampaignRef) or not isinstance(self.progress, ProgressFact):
            raise ValueError('invalid public campaign or progress')
        for name in ('candidates', 'episodes', 'sessions', 'artifacts'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name, kind in (('episodes', EpisodeFact), ('sessions', SessionFact), ('artifacts', ExportArtifact)):
            if any(not isinstance(item, kind) for item in getattr(self, name)):
                raise ValueError('invalid public ' + name)
        _frozen_fields(self, 'warnings', 'hardware', 'token_summary', 'counts')
        for key in ('episodes', 'accepted', 'rejected'):
            _counter(self.counts.get(key), optional=True)
        for key in ('total', 'control_total', 'receipt_count'):
            _counter(self.token_summary.get(key), optional=True)
        _usage_summary(self.token_summary)
        if len({a.id for a in self.artifacts}) != len(self.artifacts):
            raise ValueError('duplicate public artifact identity')
        if any(not isinstance(c, CandidateFact) for c in self.candidates) or len({c.version for c in self.candidates}) != len(self.candidates):
            raise ValueError('duplicate or absent public candidate identity')
        if self.baseline is not None and (self.baseline not in self.candidates or self.baseline.version != 0):
            raise ValueError('baseline differs from declared candidates')
        if self.incumbent is not None and (self.incumbent not in self.candidates or not self.incumbent.promoted or self.incumbent.result.state != 'pass'):
            raise ValueError('incumbent differs from verified declared candidates')
        if len({e.episode for e in self.episodes}) != len(self.episodes):
            raise ValueError('duplicate public episode identity')
        by_id = {artifact.id: artifact for artifact in self.artifacts}
        for episode in self.episodes:
            final = episode.final_candidate
            if final is None:
                continue
            digests = {}
            for name, handle in final.artifacts.items():
                artifact = by_id.get(handle)
                role = 'episode-candidate-manifest' if name == 'solution.json' else 'episode-candidate-source'
                if artifact is None or artifact.role != role or artifact.filename != f'episodes/e{episode.episode:04d}/final/{name}':
                    raise ValueError('final episode artifact differs from its source handle')
                digests[name] = artifact.sha256
            for evaluation in final.evaluations:
                if evaluation.source_sha256 != digests:
                    raise ValueError('episode evaluation does not bind final source bytes')
                if self.baseline is not None and evaluation.result.expected_case_ids != self.baseline.result.expected_case_ids:
                    raise ValueError('episode evaluation differs from campaign cases')
        for row in self.warnings:
            if not isinstance(row.get('code'), str):
                raise ValueError('public warning has no code')
            if 'gate_failed' in row:
                _flag(row['gate_failed'])

    def artifact(self, artifact_id: str) -> bytes:
        """Return declared content even after the producer changes."""
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                if hashlib.sha256(artifact.content).hexdigest() != artifact.sha256:
                    raise ValueError('public artifact integrity mismatch')
                return artifact.content
        raise KeyError(artifact_id)


@dataclass(frozen=True)
class CheckpointInfo:
    campaign_id: str
    workspace: Path
    sha256: str
    observed_tokens: int | None
    completed_episodes: int | None
    incumbent_version: int
    selection_identity: str


class WorkflowSnapshotReader(Protocol):
    """Read-only facts and declared content, without checkpoint/control authority."""
    def discover(self, directory: Path) -> tuple[CampaignRef, ...]: ...
    def read(self, workspace: Path) -> WorkflowExportSnapshot: ...


class WorkflowExportReader(WorkflowSnapshotReader, Protocol):
    def request_pause(self, workspace: Path) -> None: ...
    def resume_checkpoint(self, workspace: Path, *, campaign_id: str) -> int: ...
    def generate_checkpoint(self, workspace: Path) -> bytes: ...
    def validate_checkpoint(self, content: bytes, *, workspace: Path, campaign_id: str) -> CheckpointInfo: ...
    def restore_checkpoint(self, content: bytes, *, workspace: Path, campaign_id: str) -> CheckpointInfo: ...


def snapshot_document(snapshot: WorkflowExportSnapshot, *, include_content: bool = True) -> dict:
    """Portable public facts; no native runtime paths or evaluator-only content."""
    import base64
    value = _document(snapshot)
    value['campaign']['workspace'] = str(snapshot.campaign.workspace)
    for artifact in value['artifacts']:
        content = artifact.pop('content')
        artifact['size_bytes'] = len(content)
        if include_content:
            artifact['content_base64'] = base64.b64encode(content).decode('ascii')
    return value



def _result_from_document(row):
    """Use the same validation and ownership boundary as direct construction."""
    return ResultFact(**row)


def snapshot_from_document(value: dict) -> WorkflowExportSnapshot:
    """Decode and validate a detached public export, including missing measurements."""
    import base64
    from pathlib import PurePosixPath

    def counter(value, *, optional=False):
        if optional and value is None:
            return
        if type(value) is not int or value < 0:
            raise ValueError('invalid public counter')

    def flag(value):
        if type(value) is not bool:
            raise ValueError('invalid public boolean')

    def candidate(row):
        if row is None:
            return None
        flag(row['promoted'])
        counter(row['version']);counter(row['episode'],optional=True)
        return CandidateFact(**{**row, 'result':_result_from_document(row['result'])})

    try:
        if type(value.get('schema_version')) is not int or value['schema_version'] != EXPORT_VERSION:
            raise ValueError('unsupported public Workflow snapshot')
        artifacts = []
        for row in value['artifacts']:
            row = dict(row)
            filename=PurePosixPath(row['filename'])
            if filename.is_absolute() or '..' in filename.parts or not filename.parts or str(filename)!=row['filename']:
                raise ValueError('invalid public artifact filename')
            if not isinstance(row['id'],str) or not row['id']:
                raise ValueError('invalid public artifact identity')
            content=base64.b64decode(row.pop('content_base64'),validate=True)
            size=row.pop('size_bytes');counter(size)
            if len(content)!=size or hashlib.sha256(content).hexdigest()!=row['sha256']:
                raise ValueError('public artifact content identity changed')
            if row['visibility']!='public':
                raise ValueError('non-public artifact in public snapshot')
            artifacts.append(ExportArtifact(**row,content=content))
        if len({a.id for a in artifacts}) != len(artifacts):
            raise ValueError('duplicate public artifact identity')
        candidates=tuple(candidate(row) for row in value['candidates'])
        if any(c is None for c in candidates) or len({c.version for c in candidates}) != len(candidates):
            raise ValueError('duplicate or absent public candidate identity')
        baseline,incumbent=candidate(value['baseline']),candidate(value['incumbent'])
        if baseline is not None and (baseline not in candidates or baseline.version!=0):
            raise ValueError('baseline differs from declared candidates')
        if incumbent is not None and (incumbent not in candidates or not incumbent.promoted or incumbent.result.state!='pass'):
            raise ValueError('incumbent differs from verified declared candidates')
        progress=ProgressFact(**value['progress'])
        for number_value in (progress.episode,progress.version,progress.completed_episodes,progress.observed_tokens):
            counter(number_value,optional=True)
        flag(progress.writer_active)
        sessions=[]
        for row in value['sessions']:
            counter(row['episode'],optional=True);flag(row['resume_supported'])
            for key,number_value in row['usage'].items():
                if key.endswith('_tokens'):
                    counter(number_value,optional=True)
            sessions.append(SessionFact(**{**row,'events':tuple(row['events'])}))
        episodes=[]
        for row in value['episodes']:
            counter(row['episode']);counter(row['version'],optional=True);flag(row['accepted'])
            for key in ('mode','violation'):
                if row.get(key) is not None and not isinstance(row[key],str):
                    raise ValueError('invalid episode mode or violation')
            final=row.get('final_candidate')
            if final is not None:
                final=dict(final)
                for key in ('head_revision','revision','base_revision'):
                    revision=final[key]
                    if revision is not None and (not isinstance(revision,str) or not revision):
                        raise ValueError('invalid episode candidate revision')
                if final['revision'] is not None and final['revision']!=final['head_revision']:
                    raise ValueError('episode source revision differs from captured HEAD')
                handles=final['artifacts']
                if (not isinstance(handles,dict) or not {'kernel.py', 'solution.json'}.intersection(handles) or any(
                        not isinstance(name,str) or not name or '\\' in name or Path(name).is_absolute()
                        or any(part in {'', '.', '..'} for part in name.split('/')) for name in handles)):
                    raise ValueError('invalid final episode source handles')
                by_id={artifact.id:artifact for artifact in artifacts}
                digests={}
                for name,handle in handles.items():
                    artifact=by_id.get(handle)
                    role='episode-candidate-manifest' if name=='solution.json' else 'episode-candidate-source'
                    if (artifact is None or artifact.role!=role or
                            artifact.filename!=f"episodes/e{row['episode']:04d}/final/{name}"):
                        raise ValueError('final episode artifact differs from its source handle')
                    digests[name]=artifact.sha256
                evaluations=[]
                for evaluation in final['evaluations']:
                    if evaluation['source_sha256']!=digests:
                        raise ValueError('episode evaluation does not bind final source bytes')
                    if (baseline is not None and tuple(evaluation['result']['expected_case_ids'])
                            != baseline.result.expected_case_ids):
                        raise ValueError('episode evaluation differs from campaign cases')
                    if not isinstance(evaluation['kind'],str) or not evaluation['kind']:
                        raise ValueError('episode evaluation has no kind')
                    if evaluation['timestamp'] is not None and not isinstance(evaluation['timestamp'],str):
                        raise ValueError('invalid episode evaluation timestamp')
                    evaluations.append(EvaluationFact(**{**evaluation,'result':_result_from_document(evaluation['result'])}))
                final=EpisodeCandidateFact(**{**final,'evaluations':tuple(evaluations)})
            episodes.append(EpisodeFact(**{**row,'experiments':tuple(row['experiments']),
                'phases':tuple(row.get('phases',())),'final_candidate':final}))
        if len({episode.episode for episode in episodes})!=len(episodes):
            raise ValueError('duplicate public episode identity')
        warnings=tuple(value['warnings'])
        for row in warnings:
            if not isinstance(row.get('code'),str):
                raise ValueError('public warning has no code')
            if 'gate_failed' in row:
                flag(row['gate_failed'])
        ref=CampaignRef(**{**value['campaign'],'workspace':Path(value['campaign']['workspace'])})
        if not isinstance(ref.id,str) or not ref.id:
            raise ValueError('public campaign has no identity')
        return WorkflowExportSnapshot(EXPORT_VERSION,ref,value['framework'],candidates,baseline,incumbent,
            tuple(episodes),tuple(sessions),progress,tuple(artifacts),warnings,dict(value.get('hardware',{})),
            value.get('token_summary', {}), value.get('counts', {}))
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('malformed public Workflow snapshot') from error
