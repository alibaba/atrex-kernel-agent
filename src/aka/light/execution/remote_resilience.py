"""Host-side Remote admission, classified retries and private recovery telemetry.

Only explicit admission rejection or a caller-certified idempotent operation
may be retried. A successful transport never implies a successful GPU command.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from aka.light.execution.contract import ExecutionUnavailable


@dataclass(frozen=True)
class Policy:
    attempts: int = 16
    budget_s: float = 600
    base_delay_s: float = 1
    max_delay_s: float = 30
    admission_interval_s: float = 6.5
    replay_on_restart: bool = False
    max_replays: int = 2

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError('Invalid Remote resilience policy')
        result = cls(**value)
        for key in ['attempts', 'max_replays']:
            if type(getattr(result, key)) is not int or not 0 <= getattr(result, key) <= 1000:
                raise ValueError('Invalid Remote resilience ' + key)
        if result.attempts < 1:
            raise ValueError('Remote resilience attempts must be positive')
        for key in ['budget_s', 'base_delay_s', 'max_delay_s', 'admission_interval_s']:
            number = getattr(result, key)
            if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 86400:
                raise ValueError('Invalid Remote resilience ' + key)
        if result.budget_s <= 0 or type(result.replay_on_restart) is not bool:
            raise ValueError('Invalid Remote resilience deadline or replay permission')
        return result


class Failure(ExecutionUnavailable):
    def __init__(self, code, *, transient=False, admission_rejected=False, retry_after=0):
        self.code = code
        self.transient = transient
        self.admission_rejected = admission_rejected
        self.retry_after = retry_after
        self.exhausted = False
        super().__init__('Remote infrastructure: ' + code + '; original execution identity retained')


def classify(text):
    """Classify in memory; never persist CLI text, signed URLs or credentials."""
    lower = text.lower()
    if '429' in lower and any(s in lower for s in ['server returned', 'status', 'rate', '限流']):
        match = re.search(r'(?:retry[- ]after[\s"\:=]+|至少等待\s*)(\d+(?:\.\d+)?)', lower)
        wait = min(float(match.group(1)), 86400) if match else 0
        return Failure('rate_limited', transient=True, admission_rejected=True, retry_after=wait)
    if any(s in lower for s in ['unauthorized', 'authentication required', 'login expired', 'token expired',
                                'server returned 401', 'server returned 403', '登录已过期']):
        return Failure('authentication_required')
    if any(s in lower for s in ['different inputs', 'different runner version', 'source archive hash mismatch',
                                'escapes job directory', 'permission denied', 'invalid argument', 'unknown flag', 'pod uid mismatch']):
        return Failure('configuration_or_protocol_rejected')
    if any(s in lower for s in ['no such file or directory', 'filenotfounderror']):
        return Failure('remote_state_missing')
    if 'no space left on device' in lower:
        return Failure('remote_storage_full')
    if any(s in lower for s in ['server returned 500', 'server returned 502', 'server returned 503',
                                'server returned 504', 'connection reset', 'connection refused',
                                'timed out', 'timeout', 'temporary failure', 'unexpected eof',
                                'tls handshake', 'connection aborted', 'network is unreachable']):
        return Failure('transport_unavailable', transient=True)
    # The operation's idempotence, not an optimistic guess about exit 1, is the
    # authority for retrying an unclassified transport failure.
    return Failure('unclassified_cli_failure', transient=True)


GATE = '''import fcntl,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);interval=float(sys.argv[2])
with p.open('a+') as stream:
 fcntl.flock(stream,fcntl.LOCK_EX)
 stream.seek(0)
 try:previous=float(stream.read() or '0')
 except ValueError:previous=0
 delay=interval-(time.monotonic()-previous)
 if 0<delay<=interval:time.sleep(delay)
 admitted=time.monotonic();stream.seek(0);stream.truncate();stream.write(str(admitted));stream.flush()
print(admitted)
'''


def admit(config, interval, *, deadline=None):
    """One source-wide lock across Pods, processes and status/upload/download."""
    if interval == 0:
        return time.monotonic()
    namespace = os.environ.get('AKA_REMOTE_ADMISSION_DIR')
    base = Path(namespace) if namespace else Path.home() / '.cache/aka/remote-admission'
    base.mkdir(parents=True, exist_ok=True)
    # Rate limits are source-wide: do not split the gate by Pod or execution ID.
    key = hashlib.sha256(config.get('context', 'default').encode()).hexdigest()[:16]
    supplied = os.environ.get('AKA_REMOTE_ADMISSION_LOCK')
    target = Path(supplied) if supplied else base / (key + '.lock')
    target.parent.mkdir(parents=True, exist_ok=True)
    remaining = max(.001, deadline-time.monotonic()) if deadline is not None else 86400
    # Blocking flock joins existing waiters fairly, including old experiment
    # shims. A short-lived helper permits a bounded wait from HTTP worker
    # threads: on timeout subprocess.run kills it and releases its file lock.
    try:
        result = subprocess.run([sys.executable,'-c',GATE,str(target),str(interval)],
                                capture_output=True,text=True,timeout=remaining)
    except subprocess.TimeoutExpired:
        raise Failure('admission_wait_exhausted', transient=True) from None
    if result.returncode:
        raise Failure('local_admission_unavailable')
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise Failure('invalid_admission_response') from None


def event(directory, operation, failure, attempt, **extra):
    record = dict(time=time.time(), operation=operation, code=failure.code,
                  attempt=attempt, transient=failure.transient, **extra)
    try:
        with (Path(directory) / 'infra-events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    except OSError:
        # Telemetry must not replace the original outage with another exception.
        pass


def trace(directory,operation,phase,**extra):
    # Counts/times only: command arguments, URLs and credentials are never logged.
    try:
        with (Path(directory)/'transport-events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':time.time(),'operation':operation,'phase':phase,**extra},separators=(',',':'))+'\n')
    except OSError:pass


def call(argv, *, config, directory, policy, operation, parser, retry_safe=False):
    started = time.monotonic()
    for attempt in range(1, policy.attempts + 1):
        try:
            trace(directory,operation,'waiting',attempt=attempt,command_bytes=sum(len(str(x).encode()) for x in argv))
            admit(config, policy.admission_interval_s, deadline=started + policy.budget_s)
            trace(directory,operation,'admitted',attempt=attempt,elapsed_s=time.monotonic()-started)
            # Prevent double admission if the optional experiment shim remains
            # on PATH. Older shims may ignore this and merely add extra waiting.
            environment = {**os.environ, 'AKA_REMOTE_NATIVE_ADMISSION': '1'}
            remaining = policy.budget_s - (time.monotonic() - started)
            if remaining <= 0:
                raise Failure('retry_deadline_exhausted', transient=True)
            result = subprocess.run(argv, capture_output=True, text=True, timeout=min(60, remaining), env=environment)
            if result.returncode:
                raise classify((result.stderr or '') + '\n' + (result.stdout or ''))
            value=parser(result.stdout)
            trace(directory,operation,'complete',attempt=attempt,elapsed_s=time.monotonic()-started)
            return value
        except subprocess.TimeoutExpired:
            failure = Failure('transport_timeout', transient=True)
        except OSError as error:
            if error.errno == 28:
                failure = Failure('local_storage_full')
            elif isinstance(error, FileNotFoundError):
                failure = Failure('local_cli_unavailable')
            else:
                failure = Failure('local_transport_error', transient=True)
        except Failure as error:
            failure = error
        except (ValueError, TypeError):
            failure = Failure('invalid_cli_response', transient=True)
        retry = (failure.admission_rejected or retry_safe) and failure.transient
        event(directory, operation, failure, attempt, retry_allowed=retry)
        elapsed = time.monotonic() - started
        if not retry or attempt == policy.attempts or elapsed >= policy.budget_s:
            failure.exhausted = retry
            raise failure
        ceiling = min(policy.max_delay_s, policy.base_delay_s * 2 ** min(attempt - 1, 20))
        delay = max(failure.retry_after, random.uniform(ceiling / 2, ceiling))
        if elapsed + delay >= policy.budget_s:
            failure.exhausted = True
            raise failure
        time.sleep(delay)


def decode_runner(stdout):
    rows = [line[len('AKA_REMOTE='):] for line in stdout.splitlines() if line.startswith('AKA_REMOTE=')]
    if len(rows) != 1:
        raise Failure('invalid_runner_response', transient=True)
    try:
        result = json.loads(rows[0])
    except (ValueError, TypeError):
        raise Failure('invalid_runner_response', transient=True) from None
    if not isinstance(result, dict):
        raise Failure('invalid_runner_response', transient=True)
    if 'error' in result:
        raise classify(str(result['error']))
    return result
