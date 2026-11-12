"""DSH headless protocol, independent of its Cordis plugin implementation."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import tempfile

from .markers import phase_marker_receipts
from .model import NormalizedAgentEvent, TokenUsage, sum_token_usages

CONNECTION_ENV = "ATREX_AKA_RUNTIME_CONNECTION_FILE"


def prepare_runtime_connection(environment):
    """Hand off only the owned Session capability across DSH's child-env scrub."""
    environment = dict(environment)
    environment.pop(CONNECTION_ENV, None)
    url, token = environment.get("ATREX_AKA_RUNTIME_URL"), environment.get("ATREX_AKA_RUNTIME_TOKEN")
    if url and token:
        root = session_root(environment).parent.resolve() / "aka-connections"
        if root.is_symlink():
            raise ValueError("DSH Session connection directory must not be a symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchmod(directory, 0o700)
        finally:
            os.close(directory)
        descriptor, filename = tempfile.mkstemp(prefix="session-", suffix=".json", dir=root)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump({"url": url, "token": token}, output)
        except BaseException:
            Path(filename).unlink(missing_ok=True)
            raise
        environment[CONNECTION_ENV] = filename
    return environment


def release_runtime_connection(environment):
    filename = environment.pop(CONNECTION_ENV, None)
    if filename:
        Path(filename).unlink(missing_ok=True)


@contextmanager
def runtime_connection_environment(environment):
    """Bind the transport after the managed Runtime mints its capability."""
    prepared = prepare_runtime_connection(environment)
    try:
        yield prepared
    finally:
        release_runtime_connection(prepared)


def settings_values(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw) if raw else {}
    except (ValueError, TypeError) as error:
        raise ValueError("ATREX_DSH_SESSION_SETTINGS must be a JSON object") from error
    if not isinstance(value, dict) or set(value) - {"provider", "model", "profile", "reasoning_effort"}:
        raise ValueError("DSH settings accept only provider, model, profile and reasoning_effort")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("DSH settings must be non-empty strings")
    if "profile" in value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value["profile"].strip()):
        raise ValueError("DSH profile must be a profile name, not a path")
    return {key: item.strip() for key, item in value.items()}


def session_root(environment: Mapping[str, str]) -> Path:
    home = Path(environment.get("HOME") or str(Path.home())).expanduser()
    return Path(environment.get("DSH_HOME") or home / ".dsh").expanduser() / "aka-sessions"


def json_events(stdout: str):
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            yield value


def session_id_from_stream(stdout: str) -> str:
    for event in json_events(stdout):
        if event.get("type") == "session":
            value = event.get("sessionId")
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value):
                return value
    return ""


def terminal_failure(stdout: str, session_id: str = "") -> tuple[str | None, str | None]:
    events = list(json_events(stdout))
    final = events[-1] if events else {}
    identity = session_id_from_stream(stdout)
    if final.get("type") == "error":
        return "provider_error", str(final.get("code") or "dsh_startup_error")
    if (final.get("type") != "final" or not identity
            or final.get("sessionId") != identity
            or (session_id and session_id != identity)):
        return "provider_error", "dsh_incomplete_stream"
    reason = final.get("reason")
    if not isinstance(reason, dict):
        return "provider_error", "dsh_missing_terminal_reason"
    kind = reason.get("kind")
    if kind == "completed":
        return None, None
    if kind == "max-tokens":
        return "output_limit", "max_output_tokens"
    error = reason.get("error") if isinstance(reason.get("error"), dict) else reason
    code = str(error.get("code") or kind or "dsh_provider_error")
    if code.casefold() in {"rate_limit", "rate_limit_error", "rate_limited", "429"}:
        return "rate_limit", code
    return "provider_error", code


def native_usage(event: dict) -> TokenUsage:
    """Read settled native attempts, including retries omitted from messages."""
    data = event.get("data")
    if not isinstance(data, dict):
        return TokenUsage.unavailable()
    if isinstance(data.get("usage"), dict):
        return token_usage(data["usage"])
    chunks = data.get("stream")
    for row in reversed(chunks if isinstance(chunks, list) else []):
        chunk = row.get("chunk") if isinstance(row, dict) and row.get("type") == "chunk" else None
        if isinstance(chunk, dict) and chunk.get("type") == "usage":
            return token_usage(chunk.get("usage"))
    return TokenUsage.unavailable()


def token_usage(raw: object) -> TokenUsage:
    """DSH's shared LLM vocabulary has disjoint input/cache/output buckets."""
    if not isinstance(raw, Mapping):
        return TokenUsage.unavailable()

    for field in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens"):
        value = raw.get(field)
        if value is not None and (type(value) is not int or value < 0):
            return TokenUsage.unavailable()

    def count(name):
        value = raw.get(name)
        return value if type(value) is int and value >= 0 else None

    incoming, outgoing = count("inputTokens"), count("outputTokens")
    if incoming is None or outgoing is None:
        return TokenUsage.unavailable()
    cached, written, total = count("cacheReadTokens"), count("cacheWriteTokens"), count("totalTokens")
    components = incoming + outgoing + (cached or 0) + (written or 0)
    if total is not None and total < components:
        return TokenUsage.unavailable()
    return TokenUsage(incoming, outgoing, cached, written, total if total is not None else components,
                      "exact" if total is not None else "partial")


class UsageLedger:
    """Invocation-local step counters; resumed history is never charged again."""
    def __init__(self):
        self.steps: dict[tuple[int, int], TokenUsage] = {}
        self.pending: set[tuple[int, int]] = set()
        self.missing: set[tuple[int, int]] = set()
        self.settled = False

    def feed(self, event: dict) -> tuple[tuple[int, int] | None, TokenUsage | None]:
        if event.get("type") == "final":
            self.settled = isinstance(event.get("reason"), dict)
        if event.get("type") != "status":
            return None, None
        key = (event.get("turn"), event.get("step"))
        if any(type(value) is not int or value < 0 for value in key):
            return None, None
        if event.get("phase") == "step_start":
            self.settled = False
            self.pending.add(key)
        if event.get("phase") != "step_end":
            return None, None
        self.pending.discard(key)
        usage = token_usage(event.get("usage"))
        if usage.total_tokens is None:
            self.missing.add(key)
            return key, None
        if key in self.steps and self.steps[key] != usage:
            self.missing.add(key)
        self.steps[key] = usage
        return key, usage

    def total(self) -> TokenUsage:
        result = sum_token_usages(tuple(self.steps.values()))
        if result.total_tokens is not None and (not self.settled or self.pending or self.missing):
            result = replace(result, measurement="partial")
        return result


def normalize_stream(stdout: str):
    ledger = UsageLedger()
    normalized = []
    positions = {}
    for event in json_events(stdout):
        key, usage = ledger.feed(event)
        if usage is not None:
            position = positions.setdefault(key, len(normalized))
            row = NormalizedAgentEvent(position, "usage_delta", usage)
            if position == len(normalized):
                normalized.append(row)
            else:
                normalized[position] = row
        if event.get("type") == "tool_result" and event.get("status") == "completed":
            for action, phase, marker_id in phase_marker_receipts(event.get("result")):
                normalized.append(NormalizedAgentEvent(len(normalized), "phase_marker",
                                                       phase=phase, action=action, marker_id=marker_id))
    terminal = ledger.total()
    if terminal.total_tokens is not None:
        normalized.append(NormalizedAgentEvent(len(normalized), "terminal_usage", terminal))
    return tuple(normalized), terminal


def native_session_files(environment: Mapping[str, str], session_id: str) -> tuple[Path, list[Path]]:
    """Select this session family in AKA's uncompressed DSH persistence root."""
    root = session_root(environment).resolve()
    candidates = {}
    for index, path in enumerate(root.glob("*/*/session*.jsonl")):
        if index >= 4096:
            raise ValueError("DSH native session discovery exceeded its file limit")
        match = re.fullmatch(r"session(?:\.v([1-9][0-9]*))?\.jsonl", path.name)
        if match is None or not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
            continue
        try:
            with path.open("rb") as source:
                header = json.loads(source.readline(65536))
        except (OSError, ValueError):
            continue
        if (not isinstance(header, dict) or header.get("type") != "session"
                or not isinstance(header.get("id"), str)
                or not isinstance(header.get("parentSession", ""), str)):
            continue
        version = int(match[1] or 0)
        previous = candidates.get(path.parent)
        if previous is None or version > previous[0]:
            candidates[path.parent] = (version, path, header)
    family = {session_id}
    while True:
        children = {row[2].get("id") for row in candidates.values()
                    if row[2].get("parentSession") in family and isinstance(row[2].get("id"), str)}
        if children <= family:
            break
        family.update(children)
    files = [row[1] for row in candidates.values() if row[2].get("id") in family]
    if not any(row[2].get("id") == session_id for row in candidates.values()):
        raise ValueError("closed DSH session ledger is missing")
    return root, sorted(files)
