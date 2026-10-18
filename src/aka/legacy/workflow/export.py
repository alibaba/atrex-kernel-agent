"""Workflow owns interpretation of durable state and public evidence selection."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

from aka.contracts.workflow_export import (
    EXPORT_VERSION, CampaignRef, CandidateFact, EpisodeFact, ExportArtifact,
    ProgressFact, ResultFact, WorkflowExportSnapshot, snapshot_from_document,
)
from aka.core.keys import ServiceKey

WORKFLOW_EXPORT = ServiceKey('workflow_export', 'WorkflowExportReader', module='aka.contracts.workflow_export')
name = 'export'
provide = ('workflow_export',)
identity_packages = ('aka.legacy.workflow', 'aka.legacy.agent')
Config = {'type': 'object', 'properties': {}, 'additionalProperties': False}
Defaults = {}
RUNTIME = '.atrex_long_horizon'
MAX_WARNING_LEDGER_BYTES = 1024 * 1024
MAX_WARNING_LEDGER_LINES = 10_000
MAX_WARNING_LINE_BYTES = 8 * 1024
MAX_EXPORTED_WARNINGS = 100
WARNING_FIELDS = ('timestamp', 'severity', 'code', 'stop_reason', 'gate_failed')


def document(path):
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f'Workflow record is corrupt: {path.name}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Workflow record must be an object: {path.name}')
    return value


def _number(value):
    if value is None:
        return None
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Workflow measurement must be finite numeric or absent')
    return float(value)


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def verified_memory(workspace, memory):
    """Keep Workflow's existing promotion eligibility check private."""
    if not isinstance(memory, dict):
        return False
    performance = memory.get('performance') or {}
    latency = _number(performance.get('latency_us_geomean', performance.get('latency_us')))
    if (memory.get('correctness', {}).get('status') != 'PASS'
            or memory.get('quality_gate', {}).get('result') != 'PASS'
            or latency is None or latency <= 0
            or performance.get('measurement_status') != 'complete'):
        return False
    revision = memory.get('git_commit_hash')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        return False
    return subprocess.run(['git', 'cat-file', '-e', f'{revision}:kernel.py'],
                          cwd=workspace, capture_output=True).returncode == 0


def _record_revision(workspace, path, memory):
    result = subprocess.run(['git', 'log', '-1', '--format=%H', 'HEAD', '--',
                             path.relative_to(workspace).as_posix()],
                            cwd=workspace, capture_output=True, text=True)
    revision = result.stdout.strip() if result.returncode == 0 else ''
    if revision:
        blob = subprocess.run(['git', 'show', f'{revision}:{path.relative_to(workspace).as_posix()}'],
                              cwd=workspace, capture_output=True)
        if blob.returncode == 0 and json.loads(blob.stdout) == memory:
            return revision
    return None


def _result(memory, attempt, expected):
    perf = memory.get('performance') or {}
    correctness = (memory.get('correctness') or {}).get('status')
    gate = (memory.get('quality_gate') or {}).get('result')
    latencies = {str(k): _number(v) for k, v in (perf.get('latency_us_by_shape') or {}).items()}
    if any(v is None or v <= 0 for v in latencies.values()):
        raise ValueError('Workflow case latency is invalid')
    latency = _number(perf.get('latency_us_geomean', perf.get('latency_us')))
    complete = (perf.get('measurement_status') == 'complete' and bool(expected)
                and set(latencies) == set(expected) and latency is not None and latency > 0)
    verification = attempt.get('verification') or {}
    if attempt.get('verification_infrastructure_error') or verification.get('gate') == 'ERROR':
        state = 'infrastructure_error'
    elif str(correctness).upper() == 'FAIL':
        state = 'correctness_failure'
    elif attempt.get('status') == 'interrupted':
        state = 'interrupted'
    elif correctness == 'PASS' and gate == 'PASS' and complete:
        state = 'pass'
    elif gate in ('FAIL', 'REJECT'):
        state = 'rejected'
    else:
        state = 'unmeasured'
    metrics = {key: number for key in ('speedup_vs_ref_geomean', 'speedup_vs_ref_mean',
               'tflops', 'bandwidth_gbps', 'tflops_peak_utilization_pct',
               'bandwidth_peak_utilization_pct', 'authoritative_improvement_pct')
               if (number := _number(perf.get(key))) is not None}
    return ResultFact(state, correctness, gate, complete, latency, latencies,
                      tuple(expected), perf.get('measurement_subject'),
                      perf.get('performance_objective'), _number(perf.get('performance_score')),
                      metrics, perf.get('shape_ids_are_opaque') is True)


def _artifact(identifier, filename, role, content, content_type='application/json'):
    return ExportArtifact(identifier, filename, content_type, role,
                          hashlib.sha256(content).hexdigest(), 'public', content)


def _warning_records(path):
    if not path.is_file():
        return ()
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_WARNING_LEDGER_BYTES + 1)
    except FileNotFoundError:
        return ()
    if len(raw) > MAX_WARNING_LEDGER_BYTES:
        raise ValueError('Workflow warning ledger exceeds the byte limit')
    lines = raw.splitlines()
    if len(lines) > MAX_WARNING_LEDGER_LINES:
        raise ValueError('Workflow warning ledger exceeds the line limit')
    warnings = []
    corrupt_lines = []
    string_fields = ('timestamp', 'severity', 'code', 'message', 'stop_reason')
    for line_number, line in enumerate(lines, 1):
        try:
            if len(line) > MAX_WARNING_LINE_BYTES:
                raise ValueError('warning record exceeds the byte limit')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('warning record must be an object')
            if not isinstance(row.get('code'), str) or not row['code']:
                raise ValueError('warning record has no code')
            if any(key in row and row[key] is not None and not isinstance(row[key], str)
                   for key in string_fields):
                raise ValueError('warning record has an invalid string field')
            if ('schema_version' in row
                    and (type(row['schema_version']) is not int or row['schema_version'] < 1)):
                raise ValueError('warning record has an invalid schema version')
            if 'gate_failed' in row and type(row['gate_failed']) is not bool:
                raise ValueError('warning record has an invalid gate flag')
        except (UnicodeError, ValueError, TypeError):
            corrupt_lines.append(line_number)
            continue
        warnings.append({key: row[key] for key in WARNING_FIELDS if key in row})
    deduplicated = {}
    for warning in warnings:
        key = tuple(warning.items())
        deduplicated.pop(key, None)
        deduplicated[key] = warning
    warnings = list(deduplicated.values())
    if corrupt_lines:
        warnings.append({
            'severity': 'warning',
            'code': 'warning_ledger_corrupt',
            'gate_failed': False,
            'first_corrupt_line': str(corrupt_lines[0]),
            'corrupt_count': str(len(corrupt_lines)),
            'ledger_sha256': hashlib.sha256(raw).hexdigest(),
        })
    return tuple(warnings[-MAX_EXPORTED_WARNINGS:])


class RetainedWorkflowExport:
    def request_pause(self, workspace):
        workspace = Path(workspace).resolve()
        if not (workspace / RUNTIME / 'public-identity.json').is_file():
            raise ValueError('cannot pause an unregistered campaign')
        (workspace / RUNTIME / 'pause-requested').touch()

    def resume_checkpoint(self, workspace, *, campaign_id):
        from .checkpoint import resume
        return resume(workspace, campaign_id=campaign_id)

    def generate_checkpoint(self, workspace):
        from .checkpoint import generate
        return generate(workspace)

    def validate_checkpoint(self, content, *, workspace, campaign_id):
        from .checkpoint import validate
        return validate(content, workspace=workspace, campaign_id=campaign_id)

    def restore_checkpoint(self, content, *, workspace, campaign_id):
        from .checkpoint import restore
        return restore(content, workspace=workspace, campaign_id=campaign_id)

    def discover(self, directory):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            return ()
        # Layout recognition stays in the producer. Consumers never infer names.
        found = []
        for path in sorted(directory.iterdir()):
            if path.is_dir() and not path.is_symlink() and ((path / RUNTIME).is_dir() or (path/'workflow-export.json').is_file()):
                found.append(self._reference(path))
        return tuple(found)

    def _reference(self, workspace):
        if not (workspace/RUNTIME).exists() and (workspace/'workflow-export.json').is_file():
            snapshot = snapshot_from_document(document(workspace/'workflow-export.json'))
            return CampaignRef(snapshot.campaign.id, workspace, snapshot.campaign.name)
        identity = document(workspace / RUNTIME / 'public-identity.json')
        identifier = identity['campaign_id'] if identity else hashlib.sha256(str(workspace).encode()).hexdigest()
        return CampaignRef(identifier, workspace, workspace.name)

    def read(self, workspace):
        from .public_state import writer_active
        from aka.legacy.agent.public_output import read_sessions
        workspace = Path(workspace).resolve()
        if not (workspace/RUNTIME).exists() and (workspace/'workflow-export.json').is_file():
            return snapshot_from_document(document(workspace/'workflow-export.json'))
        runtime = workspace / RUNTIME
        state = document(runtime / 'state.json') or {}
        active = document(runtime / 'active_episode.json') or {}
        attempts = {}
        rows = state.get('attempts') or []
        if not isinstance(rows, list):
            raise ValueError('Workflow attempts must be an array')
        for row in [*rows, *(document(p) for p in sorted(runtime.glob('episodes/e*/attempt.json')))]:
            if not isinstance(row, dict) or _integer(row.get('episode')) is None:
                raise ValueError('Workflow attempt has no episode identity')
            attempts[row['episode']] = row
        by_version = {row.get('version'): row for row in attempts.values()}
        memories = []
        for path in sorted((workspace / 'memory').glob('v*.json')):
            if re.fullmatch(r'v\d+', path.stem):
                memories.append((int(path.stem[1:]), document(path), path))
        baseline_memory = next((m for n,m,_ in memories if n == 0), {}) or {}
        expected = tuple(sorted(str(k) for k in (baseline_memory.get('performance') or {}).get('latency_us_by_shape', {})))
        candidates = []
        for version, memory, path in sorted(memories):
            attempt = by_version.get(version, {})
            revision = memory.get('git_commit_hash')
            record_revision = _record_revision(workspace, path, memory)
            # Carry-forward records describe the incumbent, never a rejected source.
            promoted = version == 0 or attempt.get('accepted') is True or (version == 1 and not attempt and verified_memory(workspace,memory))
            candidates.append(CandidateFact(version, _integer(attempt.get('episode')),
                revision if isinstance(revision, str) else None, record_revision, promoted,
                memory.get('timestamp'), _result(memory, attempt, expected),
                (memory.get('optimization') or {}).get('action_description')))
        known = {candidate.version for candidate in candidates}
        for version, attempt in by_version.items():
            if type(version) is int and version >= 0 and version not in known:
                candidates.append(CandidateFact(version,_integer(attempt.get('episode')),
                    attempt.get('candidate_commit'),None,False,attempt.get('timestamp'),
                    _result({},attempt,expected),attempt.get('summary')))
                known.add(version)
        active_version = _integer(active.get('memory_version'))
        if active_version is not None and active_version not in known:
            candidates.append(CandidateFact(active_version,_integer(active.get('episode')),None,None,False,None,
                ResultFact('running',None,None,False,None,{},expected,None,None,None)))
        candidates.sort(key=lambda candidate:candidate.version)
        artifacts = []
        # Bind exported sources to the exact measured candidate revision.
        for candidate in candidates:
            if not candidate.revision or not candidate.record_revision:
                continue
            for filename, role, mime in (('kernel.py','candidate-source','text/x-python'),
                                          ('solution.json','candidate-manifest','application/json')):
                blob = subprocess.run(['git','show',f'{candidate.revision}:{filename}'],cwd=workspace,capture_output=True)
                if blob.returncode:
                    if filename == 'kernel.py':
                        raise ValueError('recorded candidate source is missing')
                    continue
                artifacts.append(_artifact(f'v{candidate.version}_' + filename.replace('.','_'),
                    f'candidates/v{candidate.version}/{filename}',role,blob.stdout,mime))
        # Public authored plans and compact profiler evidence are declared by
        # Workflow; consumers do not follow paths embedded in private records.
        from .episode_evidence import selected, read as read_episode_evidence
        for relative, role, content in selected(workspace):
            artifacts.append(_artifact('output_' + hashlib.sha256(relative.encode()).hexdigest(),
                                       relative, role, content, 'text/plain'))
        episodes = []
        from .episode_output import read as read_episode_output
        for number, attempt in sorted(attempts.items()):
            artifacts.extend(read_episode_evidence(runtime / f'episodes/e{number:04d}', episode=number))
            final_candidate, final_artifacts = read_episode_output(
                runtime / f'episodes/e{number:04d}', episode=number,
                version=_integer(attempt.get('version')), expected_case_ids=expected, attempt=attempt)
            artifacts.extend(final_artifacts)
            journal = document(runtime / f'episodes/e{number:04d}/episode_runtime/journal.json') or {}
            experiments = journal.get('experiments') or []
            # Only declared experiment facts cross the public boundary.
            public_experiments = tuple({k: row[k] for k in ('name', 'hypothesis', 'change', 'result', 'decision', 'timestamp')
                                        if k in row} for row in experiments if isinstance(row, dict))
            telemetry = document(runtime / f'episodes/e{number:04d}/telemetry.summary.json') or {}
            phases = tuple({'phase':phase, 'measurement':row.get('measurement','unavailable'),
                            'usage':row.get('usage'), 'interval_count':row.get('interval_count',0)}
                           for phase,row in (telemetry.get('phase_tokens',{}).get('phases') or {}).items()
                           if isinstance(row,dict))
            episodes.append(EpisodeFact(number, _integer(attempt.get('version')),
                str(attempt.get('status') or 'unmeasured'), attempt.get('accepted') is True,
                attempt.get('candidate_commit'), attempt.get('session_id'), attempt.get('summary'),
                (attempt.get('verification') or {}).get('gate'), public_experiments, phases,
                attempt.get('mode'), attempt.get('violation'), final_candidate))
        for path in sorted((workspace/'.gpu_wiki_profile/raw/query_events').glob('*/*.json')):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(workspace):
                relative = path.relative_to(workspace).as_posix()
                artifacts.append(_artifact('wiki_'+hashlib.sha256(relative.encode()).hexdigest(),
                    'wiki/'+path.parent.name+'/'+path.name,'wiki-query-event',path.read_bytes()))
        sessions = list(read_sessions(workspace))
        for directory in sorted(runtime.glob('episodes/e*/agent_observations')):
            sessions.extend(read_sessions(directory, direct=True))
        sessions = tuple(replace(s, episode=next((e.episode for e in episodes if e.session_id==s.session_id),s.episode)) for s in sessions)
        observed = [p.stat().st_mtime for p in (runtime/'state.json', runtime/'active_episode.json', workspace/'memory/live.json') if p.is_file()]
        observed.extend(datetime.fromisoformat(s.updated_at).timestamp() for s in sessions if s.updated_at)
        updated = datetime.fromtimestamp(max(observed), timezone.utc).isoformat() if observed else None
        running = writer_active(workspace)
        terminal = document(workspace / 'trace-retention-manifest.json') or {}
        phase = active.get('phase') if running else ('paused' if (runtime/'pause-requested').exists() else terminal.get('status', 'stopped'))
        progress = ProgressFact(str(phase or 'preparing'), _integer(active.get('episode')),
            _integer(active.get('memory_version')), _integer(state.get('episodes')),
            _integer(state.get('tokens')), updated, running)
        warnings = _warning_records(runtime / 'warnings.jsonl')
        framework = str((document(workspace / '.orchestrator_mode.json') or {}).get('framework')
                        or (document(workspace / 'framework_baseline.json') or {}).get('framework') or '')
        baseline = next((c for c in candidates if c.version == 0), None)
        incumbent = next((c for c in reversed(candidates) if c.promoted and c.record_revision
                          and c.result.state == 'pass'), None)
        return WorkflowExportSnapshot(EXPORT_VERSION, self._reference(workspace), framework,
            tuple(candidates), baseline, incumbent, tuple(episodes), sessions, progress,
            tuple(artifacts), warnings,
            {key:str(value) for key,value in (terminal.get('hardware') or {}).items()
             if key in ('platform','arch','sandbox_hardware')})


def apply(ctx, config):
    ctx.provide(WORKFLOW_EXPORT, RetainedWorkflowExport())
