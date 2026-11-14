"""Count DSH native child attempts once across persisted JSONL generations."""
from dataclasses import replace
from pathlib import Path
import json
from aka.core.filesystem import read_regular_bytes
from .dsh import native_session_files, native_usage
from .model import TokenUsage, sum_token_usages


def attempts(files):
    rows = {}
    errors = []
    total_bytes = 0
    for path in files:
        cap = 16 * 1024 * 1024
        raw = read_regular_bytes(path, limit=cap + 1)
        if len(raw) > cap:
            raise ValueError("DSH observation exceeds its file byte cap")
        total_bytes += len(raw)
        if total_bytes > 64 * 1024 * 1024:
            raise ValueError("DSH observation exceeds its byte cap")
        session, seeded = None, False
        for line in raw.splitlines():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("DSH observation must contain object events")
            kind = event.get("type")
            if kind == "session":
                session, seeded = event.get("id"), event.get("isSeeded") is True
            elif kind == "session/end-seed":
                seeded = False
            elif not seeded and kind in {"assistant/message", "assistant/attempt"}:
                sequence = event.get("seq")
                if not isinstance(session, str) or type(sequence) is not int or sequence < 0:
                    errors.append("dsh_missing_sequence")
                    continue
                key = (session, sequence)
                usage = native_usage(event)
                if usage.total_tokens is None:
                    errors.append("dsh_usage_unavailable")
                if key in rows and rows[key] != usage:
                    errors.append("dsh_usage_changed")
                rows[key] = usage
    return rows, tuple(dict.fromkeys(errors))


class DshSessionLedgerObserver:
    def __init__(self, environment, session_id=""):
        self.environment = dict(environment)
        self.previous = {}
        if session_id:
            try:
                _, files = native_session_files(self.environment, session_id)
            except ValueError:
                files = ()
            try:
                self.previous, self.baseline_errors = attempts(files)
            except (OSError, ValueError):
                self.baseline_errors = ("dsh_baseline_unavailable",)
        else:
            self.baseline_errors = ()

    def observe(self, session_id, stream_usage):
        _, files = native_session_files(self.environment, session_id)
        current, errors = attempts(files)
        changed = any(key in self.previous and self.previous[key] != usage for key, usage in current.items())
        if changed:
            errors += ("dsh_usage_changed",)
        children = [usage for key, usage in current.items() if key not in self.previous and key[0] != session_id]
        total = sum_token_usages([stream_usage, *children]) if children else stream_usage
        errors = tuple(dict.fromkeys((*self.baseline_errors, *errors)))
        if errors and total.total_tokens is not None:
            total = replace(total, measurement="partial")
        self.previous.update(current)
        return total, errors
