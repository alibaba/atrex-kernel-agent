"""Workflow-owned projection of progress and settled usage for public readers."""
from datetime import datetime, timezone
import math
from pathlib import Path
import re

TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'total_tokens')
SCOPE = ('Settled Episode Agent usage. Excludes setup, V1, independent reviewers and auxiliary CLIs. '
         'Current invocation usage settles when the call returns.')


def obj(value):
    return value if isinstance(value, dict) else {}


def sequence(value):
    return value if isinstance(value, list) else []


def integer(value):
    return value if type(value) is int and value >= 0 else None


def version(value):
    if integer(value) is not None:
        return value
    match = re.fullmatch(r'v([0-9]+)', value) if isinstance(value, str) else None
    return int(match[1]) if match else None


def stamp(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, OverflowError):
        return 0.0


def mtime(path):
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value else None


def usage(value):
    value = obj(value)
    measurement = value.get('measurement', 'unavailable')
    if measurement not in {'exact', 'partial', 'unavailable'}:
        measurement = 'unavailable'
    return {**{key: integer(value.get(key)) if measurement != 'unavailable' else None
               for key in TOKEN_FIELDS}, 'measurement': measurement}


def episode_usage(attempt, telemetry):
    """Retain the former settled-ledger rules without exposing the ledger."""
    phases = obj(telemetry.get('phase_tokens'))
    terminal = usage(phases.get('terminal_usage'))
    reasons = sorted({str(r) for r in sequence(telemetry.get('reason_codes'))
                      + sequence(phases.get('reason_codes'))
                      + sequence(obj(attempt.get('telemetry')).get('reason_codes'))})
    control = integer(telemetry.get('control_tokens'))
    if control is None:
        control = integer(attempt.get('tokens'))
    total = terminal['total_tokens']
    measurement = terminal['measurement'] if total is not None else 'unavailable'
    if total is not None and control is not None and total != control:
        reasons = sorted(set(reasons) | {'control_token_total_mismatch'})
    if {'same_session_resume_usage_semantics_unqualified', 'control_token_total_mismatch'} & set(reasons) and measurement == 'exact':
        measurement = 'partial'
    if total == 0 and measurement != 'exact' and 'structured_usage_unavailable' in reasons:
        total, measurement = None, 'unavailable'
    if total is None and control:
        total, measurement = control, 'partial'
        reasons = sorted(set(reasons) | {'structured_usage_unavailable'})
    return {'total': total, 'control_tokens': control if control or total is not None else None,
            'raw_control_tokens': control, 'measurement': measurement, 'terminal_usage': terminal,
            'attribution_measurement': phases.get('measurement', 'unavailable'), 'reason_codes': reasons}


def campaign_usage(state, settled_usage, *, pending):
    raw = integer(state.get('tokens'))
    unavailable = any(row['measurement'] == 'unavailable' for row in settled_usage)
    observed = sum(row['raw_control_tokens'] or 0 for row in settled_usage)
    reasons = sorted({reason for row in settled_usage for reason in row['reason_codes']})
    if unavailable:
        reasons = sorted(set(reasons) | {'episode_usage_unavailable'})
    if raw is not None and observed != raw:
        reasons = sorted(set(reasons) | {'control_usage_not_fully_archived'})
    if raw is None:
        measurement = 'unavailable'
    elif unavailable or any(row['measurement'] != 'exact' for row in settled_usage) or observed != raw:
        measurement = 'partial' if raw else 'unavailable'
    else:
        measurement = 'exact'
    return {'total': None if measurement == 'unavailable' else raw, 'control_total': raw,
            'measurement': measurement, 'receipt_count': len(obj(state.get('usage_receipts'))),
            'reason_codes': reasons, 'pending': pending, 'scope': SCOPE}


def experiment(value):
    value = obj(value)
    result = {key: value[key] for key in ('name', 'timestamp', 'hypothesis', 'change', 'result',
              'conclusion', 'summary', 'decision') if isinstance(value.get(key), str)}
    evaluation = obj(value.get('evaluation'))
    result['evaluation'] = {key: evaluation[key] for key in ('correctness', 'performance', 'latency_us', 'kernel_hash')
        if isinstance(evaluation.get(key), str) or type(evaluation.get(key)) in (int, float)
        and math.isfinite(evaluation[key])}
    return result


def telemetry(workspace, number, attempt, document, warnings):
    path = workspace / '.atrex_long_horizon' / f'episodes/e{number:04d}/telemetry.summary.json'
    value = document(path) or {}
    if value and value.get('schema_version', 'atrex_long_horizon_episode_telemetry_v1') != 'atrex_long_horizon_episode_telemetry_v1':
        warnings.append({'code': 'unsupported_telemetry_schema'})
        return {}
    if value and (value.get('episode') != number or version(value.get('version')) != version(attempt.get('version'))):
        warnings.append({'code': 'telemetry_identity_mismatch'})
        return {}
    return value


def journal(workspace, number, attempt, active, document, warnings):
    path = workspace / '.atrex_long_horizon' / f'episodes/e{number:04d}/episode_runtime/journal.json'
    value = document(path) or {}
    worktree = active.get('worktree')
    if not value and active.get('episode') == number and isinstance(worktree, str):
        value = document(Path(worktree) / '.atrex_long_horizon/journal.json') or {}
    if value and (value.get('schema_version', 1) != 1 or value.get('episode') != number
            or value.get('memory_version') is not None and version(value['memory_version']) != version(attempt.get('version'))
            or active.get('episode') == number and any(active.get(key) and value.get(key) != active[key]
                for key in ('base_commit', 'episode_branch'))):
        warnings.append({'code': 'journal_identity_mismatch'})
        return {}
    return value


def progress(workspace, state, active, attempts, terminal, document, warnings):
    live = document(workspace / 'memory/live.json') or {}
    associated = not active or not live or (active.get('episode') == live.get('episode')
        and version(active.get('memory_version')) == version(live.get('version')))
    if not associated:
        warnings.append({'code': 'active_live_identity_mismatch'})
    matched = live if associated else {}
    phase = active.get('phase') or matched.get('phase') or 'waiting'
    number = active.get('episode', matched.get('episode'))
    settled = number in attempts or matched.get('phase') == 'recorded'
    observed = max(stamp(live.get('updated_at')), mtime(workspace / '.atrex_long_horizon/active_episode.json'),
                   mtime(workspace / '.atrex_long_horizon/state.json'))
    stale_terminal = bool(terminal and observed > stamp(terminal.get('generated_at')))
    if stale_terminal:
        warnings.append({'code': 'older_terminal_record'})
    if not active and not live:
        crash = document(workspace / '.atrex_long_horizon/framework_baseline/progress.json') or {}
        if crash.get('status') == 'interrupted':
            phase = 'v1_interrupted'
    if not active and not stale_terminal and terminal.get('status') in {'completed', 'failed', 'interrupted'}:
        phase = terminal['status']
    if (workspace / '.atrex_long_horizon/pause-requested').exists():
        phase = 'paused'
    return {'phase': 'recorded' if settled else phase, 'episode': integer(number),
            'version': version(active.get('memory_version', matched.get('version'))),
            'mode': active.get('mode') or obj(attempts.get(number)).get('mode'),
            'experiment_count': integer(matched.get('experiment_count')),
            'latest_experiment': experiment(matched.get('latest_experiment')),
            'updated_at': iso(max(stamp(matched.get('updated_at')), mtime(workspace / '.atrex_long_horizon/active_episode.json'))),
            'record_updating': not associated,
            'pending_tokens': bool(number and not settled and (phase in {'preparing', 'exploring'}
                or phase not in {'awaiting_verification', 'checking_fast_evaluator', 'verifying', 'promoting',
                                 'promoted', 'recording', 'recorded', 'completed', 'failed', 'interrupted'})),
            'last_terminal': {'status': terminal.get('status'), 'updated_at': terminal.get('generated_at'), 'stale': stale_terminal}}
