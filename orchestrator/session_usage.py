"""Provider counters, not tokenizer estimates; native/stream copies are one bill.

Claude reconciliation follows the main Runtime's claude_ledger implementation.
An unreconciled total is explicitly partial, never main + possibly inclusive total.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

from .agent_runtime.adapter import token_usage_from_mapping, token_usage_from_model_usage
from .agent_runtime.model import TokenUsage, subtract_token_usage, sum_token_usages

COMPONENTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def json_events(text: str):
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            yield value


def _codex_usage(raw: object) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage.unavailable()
    # Codex's input includes cached input. Our buckets are disjoint.
    value = token_usage_from_mapping(raw)
    cached = raw.get("cached_input_tokens", 0)
    if not isinstance(cached, int) or isinstance(cached, bool) or cached < 0:
        return TokenUsage.unavailable()
    if value.input_tokens is None or value.input_tokens < cached:
        return TokenUsage.unavailable()
    return replace(
        value,
        input_tokens=value.input_tokens - cached,
        cache_read_tokens=cached,
        cache_write_tokens=raw.get("cache_write_input_tokens", 0),
    )


def _same(left: TokenUsage, right: TokenUsage) -> bool:
    return left.total_tokens is not None and all(
        getattr(left, key) == getattr(right, key) for key in (*COMPONENTS, "total_tokens")
    )


def _credits(stdout: str, finished: bool) -> dict:
    """Qoder credits are not tokens; a final counter may already include children."""
    terminal = None

    def read(value):
        if not isinstance(value, dict):
            return None
        result = value.get("credits", value.get("total_credits", value.get("totalCredits")))
        return (
            result
            if isinstance(result, (int, float))
            and not isinstance(result, bool)
            and math.isfinite(result)
            and result >= 0
            else None
        )

    for event in json_events(stdout):
        if event.get("type") != "result":
            continue
        terminal = read(event)
        if terminal is None:
            values = [read(row) for row in (event.get("modelUsage") or {}).values()]
            if values and all(value is not None for value in values):
                terminal = sum(values)
    return {
        "credits": terminal,
        "measurement": "exact"
        if terminal is not None and finished
        else "unavailable"
        if terminal is None
        else "partial",
        "scope": "provider_terminal; child credits are not added again",
    }


def _upper_bound(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    values = {}
    for key in COMPONENTS:
        known = [getattr(row, key) for row in (left, right) if getattr(row, key) is not None]
        values[key] = max(known) if known else None
    total = max(
        left.total_tokens or 0, right.total_tokens or 0, sum(v or 0 for v in values.values())
    )
    if left.total_tokens is None and right.total_tokens is None:
        return TokenUsage.unavailable()
    return TokenUsage(**values, total_tokens=total, measurement="partial")


class UsageAccumulator:
    """Fold newly captured records once; retain counters, never transcript text."""

    def __init__(self, backend: str, *, max_entries: int = 100_000):
        self.backend = backend
        self.max_entries = max_entries
        self.old_ids: set[str] = set()
        self.responses: dict[str, dict] = {}
        self.missing: set[str] = set()
        self.terminal = TokenUsage.unavailable()
        self.native_seen = False
        self.codex_totals: dict[str, TokenUsage] = {}
        self.prior_codex: dict[str, TokenUsage] = {}
        self.indices: dict[str, int] = {}
        self.credit_event = ""

    def feed(self, path: str, text: str, *, previous: bool = False) -> None:
        is_native = path != "provider/stdout.stream-json"
        for line in text.splitlines():
            # A runaway stream cannot make the counter index grow indefinitely.
            if len(self.responses) + len(self.old_ids) + len(self.missing) >= self.max_entries:
                self.missing.add("usage_index_limit_exceeded")
                return
            index = self.indices.get(path, 0)
            self.indices[path] = index + 1
            try:
                event = json.loads(line)
            except ValueError:
                if is_native:
                    self.missing.add(path + ":malformed-json")
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            body = event.get("payload")
            if previous:
                if isinstance(message, dict) and message.get("id"):
                    self.old_ids.add(message["id"])
                    self.responses.pop(message["id"], None)
                    self.missing.discard(message["id"])
                if isinstance(body, dict) and body.get("type") == "token_count":
                    info = body.get("info") or {}
                    self.prior_codex[path] = _codex_usage(info.get("total_token_usage"))
                continue
            if is_native:
                self.native_seen = True
            if not is_native and event.get("type") in {"result", "turn.completed"}:
                self.credit_event = line
                self.terminal = (
                    _codex_usage(event.get("usage"))
                    if self.backend == "codex"
                    else token_usage_from_model_usage(event.get("modelUsage"))
                )
                if self.terminal.total_tokens is None:
                    self.terminal = token_usage_from_mapping(event.get("usage"))
                continue
            if is_native and isinstance(body, dict) and body.get("type") == "token_count":
                info = body.get("info") or {}
                current = _codex_usage(info.get("total_token_usage"))
                prior = self.codex_totals.get(path, self.prior_codex.get(path, TokenUsage.zero()))
                if current.total_tokens is not None:
                    try:
                        delta = subtract_token_usage(current, prior)
                    except ValueError:
                        self.missing.add(path + ":counter-regression")
                        continue
                    self.codex_totals[path] = current
                    if delta.total_tokens:
                        last = _codex_usage(info.get("last_token_usage"))
                        if not _same(delta, last):
                            self.missing.add(path + ":unreconciled-token-delta")
                        key = f"{path}:{current.total_tokens}"
                        self.responses[key] = {
                            "message_id": key,
                            "path": path,
                            "usage": delta,
                            "agent": path,
                            "native": True,
                        }
                continue
            if not isinstance(message, dict) or (
                event.get("type") not in {"assistant", "message", "message_end"}
                or message.get("role", "assistant") != "assistant"
            ):
                continue
            message_id = message.get("id") or event.get("id")
            if not message_id:
                message_id = f"{path}:line-{index}"
                self.missing.add("response_identity_unavailable")
            if message_id in self.old_ids:
                continue
            usage = token_usage_from_mapping(message.get("usage"))
            if usage.total_tokens is None:
                if message_id not in self.responses:
                    self.missing.add(message_id)
                continue
            self.missing.discard(message_id)
            agent = (
                event.get("agentId")
                or event.get("parent_tool_use_id")
                or (path if "subagents/" in path else "main")
            )
            earlier = self.responses.get(message_id)
            # Native counters win regardless of pipe/monitor interleaving.
            # A child's copied main context is not a second bill.
            if (
                earlier
                and earlier["native"]
                and (not is_native or (earlier["agent"] == "main" and agent != "main"))
            ):
                continue
            self.responses[message_id] = {
                "message_id": message_id,
                "path": path,
                "agent": agent,
                "usage": usage,
                "native": is_native,
            }

    def report(self, *, finished: bool = False) -> dict:
        observed = sum_token_usages([r["usage"] for r in self.responses.values()])
        main = sum_token_usages(
            [r["usage"] for r in self.responses.values() if r["agent"] == "main"]
        )
        warnings = []
        exact = finished and not self.missing and self.native_seen and bool(self.responses)
        if self.backend == "codex" and self.codex_totals:
            # Native counters are cumulative per rollout; subtract the start snapshot.
            # Root's stdout total may be session-cumulative or turn-only. Do not add it again.
            total = observed
            exact = exact and (
                _same(observed, self.terminal)
                or any(_same(total, self.terminal) for total in self.codex_totals.values())
            )
            basis = "native_rollout_deltas"
            if not exact:
                warnings.append("codex_native_usage_incomplete_or_unreconciled")
        elif exact and (_same(observed, self.terminal) or _same(main, self.terminal)):
            total = observed
            basis = "reconciled_unique_responses"
            if not _same(observed, self.terminal):
                warnings.append("terminal_excludes_subagents; native child usage included")
        else:
            total = _upper_bound(observed, self.terminal)
            exact = False
            basis = "unreconciled_provider_counters"
            warnings.append(
                "native response usage and terminal total could not be fully reconciled"
            )
        if not exact and total.total_tokens is not None:
            total = replace(total, measurement="partial")
        agent_rows = {}
        for response in self.responses.values():
            agent_rows.setdefault(str(response["agent"]), []).append(response["usage"])
        groups = {
            agent: asdict(sum_token_usages(rows)) for agent, rows in sorted(agent_rows.items())
        }
        return {
            "backend": self.backend,
            "total": asdict(total),
            "accounting_basis": basis,
            "provider_credits": _credits(self.credit_event, finished)
            if self.backend == "qodercli"
            else None,
            "terminal": asdict(self.terminal),
            "observed_responses": asdict(observed),
            "by_agent": groups,
            "response_count": len(self.responses),
            "responses": [
                {
                    **{k: v for k, v in r.items() if k not in {"usage", "native"}},
                    "usage": asdict(r["usage"]),
                }
                for r in self.responses.values()
            ],
            "warnings": warnings,
            "missing_usage": sorted(self.missing),
            "subagent_coverage": "native_transcripts_and_stream; unexported calls cannot be counted",
        }


def summarize_usage(
    backend: str,
    stdout: str,
    native: dict[str, str],
    *,
    previous: dict[str, str] | None = None,
    finished: bool = False,
) -> dict:
    """One-shot counterpart of the live incremental accounting."""
    accumulator = UsageAccumulator(backend)
    for path, text in (previous or {}).items():
        accumulator.feed(path, text, previous=True)
    accumulator.feed("provider/stdout.stream-json", stdout)
    for path, text in sorted(native.items()):
        accumulator.feed(path, text)
    return accumulator.report(finished=finished)
