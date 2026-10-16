"""Detached Workflow facts and content-bound public outputs (schema 3).

Missing measurements remain None. Episode outcome and measurement completeness
are independent. No private memory, journal or ledger dictionaries cross here.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Literal, Protocol

EXPORT_VERSION = 3
ResultState = Literal['unmeasured', 'running', 'pass', 'correctness_failure',
                      'infrastructure_error', 'rejected', 'interrupted']


@dataclass(frozen=True)
class ResultFact:
    state: ResultState
    correctness: str | None
    gate: str | None
    complete: bool
    latency_us: float | None
    case_latencies_us: dict[str, float]
    expected_case_ids: tuple[str, ...]
    subject: str | None
    objective: str | None
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    case_ids_opaque: bool = True


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


@dataclass(frozen=True)
class EvaluationFact:
    kind: str
    timestamp: str | None
    source_sha256: dict[str, str]
    result: ResultFact


@dataclass(frozen=True)
class EpisodeCandidateFact:
    head_revision: str | None
    revision: str | None
    base_revision: str | None
    artifacts: dict[str, str]
    evaluations: tuple[EvaluationFact, ...] = ()


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


@dataclass(frozen=True)
class SessionFact:
    session_id: str
    runtime: str
    episode: int | None
    status: str
    updated_at: str | None
    usage: dict[str, int | str | None]
    events: tuple[dict, ...] = ()
    resume_supported: bool = False


@dataclass(frozen=True)
class ProgressFact:
    phase: str
    episode: int | None
    version: int | None
    completed_episodes: int | None
    observed_tokens: int | None
    updated_at: str | None
    writer_active: bool


@dataclass(frozen=True)
class CampaignRef:
    id: str
    workspace: Path
    name: str


@dataclass(frozen=True)
class ExportArtifact:
    id: str
    filename: str
    content_type: str
    role: str
    sha256: str
    visibility: Literal['public']
    content: bytes = field(repr=False)

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
    hardware: dict[str, str] = field(default_factory=dict)

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


class WorkflowExportReader(Protocol):
    def discover(self, directory: Path) -> tuple[CampaignRef, ...]: ...
    def read(self, workspace: Path) -> WorkflowExportSnapshot: ...

    def request_pause(self, workspace: Path) -> None: ...
    def resume_checkpoint(self, workspace: Path, *, campaign_id: str) -> int: ...
    def generate_checkpoint(self, workspace: Path) -> bytes: ...
    def validate_checkpoint(self, content: bytes, *, workspace: Path, campaign_id: str) -> CheckpointInfo: ...
    def restore_checkpoint(self, content: bytes, *, workspace: Path, campaign_id: str) -> CheckpointInfo: ...


def snapshot_document(snapshot: WorkflowExportSnapshot, *, include_content: bool = True) -> dict:
    """Portable public facts; no native runtime paths or evaluator-only content."""
    from dataclasses import asdict
    import base64
    value = asdict(snapshot)
    value['campaign']['workspace'] = str(snapshot.campaign.workspace)
    for artifact in value['artifacts']:
        content = artifact.pop('content')
        artifact['size_bytes'] = len(content)
        if include_content:
            artifact['content_base64'] = base64.b64encode(content).decode('ascii')
    return value



def _result_from_document(row):
    """Shared semantic check for durable producer output and portable snapshots."""
    import math
    def number(value, *, positive=False, optional=False):
        if optional and value is None:
            return
        if type(value) not in (int,float) or not math.isfinite(value) or (positive and value <= 0):
            raise ValueError('invalid public measurement')
    def flag(value):
        if type(value) is not bool:
            raise ValueError('invalid public boolean')
    result = dict(row)
    result['expected_case_ids'] = tuple(result['expected_case_ids'])
    expected = result['expected_case_ids']
    if len(set(expected)) != len(expected) or any(not isinstance(key,str) or not key for key in expected):
        raise ValueError('invalid public case identities')
    if result['state'] not in ('unmeasured','running','pass','correctness_failure','infrastructure_error','rejected','interrupted'):
        raise ValueError('unsupported public result state')
    number(result['latency_us'],positive=True,optional=True)
    number(result['score'],optional=True)
    for key, value in result['case_latencies_us'].items():
        if not isinstance(key,str):
            raise ValueError('invalid public case identity')
        number(value,positive=True)
    for value in result.get('metrics',{}).values():
        number(value)
    flag(result['complete']); flag(result.get('case_ids_opaque',True))
    if result['complete'] and (not expected or set(result['case_latencies_us']) != set(expected) or result['latency_us'] is None):
        raise ValueError('complete result lacks required measurements')
    if result['state']=='pass' and (not result['complete'] or result['correctness']!='PASS' or result['gate']!='PASS'):
        raise ValueError('passing result lacks complete verification')
    return ResultFact(**result)


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
                import re
                final=dict(final)
                for key in ('head_revision','revision','base_revision'):
                    revision=final[key]
                    if revision is not None and (not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision)):
                        raise ValueError('invalid episode candidate revision')
                if final['revision'] is not None and final['revision']!=final['head_revision']:
                    raise ValueError('episode source revision differs from captured HEAD')
                handles=final['artifacts']
                if not isinstance(handles,dict) or 'kernel.py' not in handles or set(handles)-{'kernel.py','solution.json'}:
                    raise ValueError('invalid final episode source handles')
                by_id={artifact.id:artifact for artifact in artifacts}
                digests={}
                for name,handle in handles.items():
                    artifact=by_id.get(handle)
                    role='episode-candidate-source' if name=='kernel.py' else 'episode-candidate-manifest'
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
            tuple(episodes),tuple(sessions),progress,tuple(artifacts),warnings,dict(value.get('hardware',{})))
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('malformed public Workflow snapshot') from error
