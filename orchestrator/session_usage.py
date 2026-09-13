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


def summarize_usage(
    backend: str,
    stdout: str,
    native: dict[str, str],
    *,
    previous: dict[str, str] | None = None,
    finished: bool = False,
) -> dict:
    """Attribute unique responses; previous native content is excluded on resume."""
    previous = previous or {}
    old_ids = set()
    for text in previous.values():
        for event in json_events(text):
            message = event.get("message", {})
            if isinstance(message, dict) and message.get("id"):
                old_ids.add(message["id"])
    responses: dict[str, dict] = {}
    missing = set()
    terminal = TokenUsage.unavailable()
    native_seen = False
    codex_totals: dict[str, TokenUsage] = {}
    prior_codex: dict[str, TokenUsage] = {}
    for path, text in previous.items():
        for event in json_events(text):
            body = event.get("payload", {})
            if isinstance(body, dict) and body.get("type") == "token_count":
                info = body.get("info") or {}
                prior_codex[path] = _codex_usage(info.get("total_token_usage"))

    # Stream is provisional; the last native counters supersede it by message ID.
    for path, text in [("provider/stdout.stream-json", stdout), *sorted(native.items())]:
        is_native = path != "provider/stdout.stream-json"
        if is_native:
            for line in text.splitlines():
                try:
                    json.loads(line)
                except ValueError:
                    missing.add(path + ":malformed-json")
        for index, event in enumerate(json_events(text)):
            if is_native:
                native_seen = True
            if not is_native and event.get("type") in {"result", "turn.completed"}:
                terminal = (
                    _codex_usage(event.get("usage"))
                    if backend == "codex"
                    else token_usage_from_model_usage(event.get("modelUsage"))
                )
                if terminal.total_tokens is None:
                    terminal = token_usage_from_mapping(event.get("usage"))
                continue
            body = event.get("payload", {})
            if is_native and isinstance(body, dict) and body.get("type") == "token_count":
                info = body.get("info") or {}
                current = _codex_usage(info.get("total_token_usage"))
                prior = codex_totals.get(path, prior_codex.get(path, TokenUsage.zero()))
                if current.total_tokens is not None:
                    try:
                        delta = subtract_token_usage(current, prior)
                    except ValueError:
                        missing.add(path + ":counter-regression")
                        continue
                    codex_totals[path] = current
                    if delta.total_tokens:
                        last = _codex_usage(info.get("last_token_usage"))
                        if not _same(delta, last):
                            missing.add(path + ":unreconciled-token-delta")
                        key = f"{path}:{current.total_tokens}"
                        responses[key] = {
                            "message_id": key,
                            "path": path,
                            "usage": delta,
                            "agent": path,
                            "native": True,
                        }
                continue
            message = event.get("message")
            if not isinstance(message, dict) or (
                event.get("type") not in {"assistant", "message", "message_end"}
                or message.get("role", "assistant") != "assistant"
            ):
                continue
            message_id = message.get("id") or event.get("id")
            if not message_id:
                # Without identity, repeated copies cannot be reliably reconciled.
                message_id = f"{path}:line-{index}"
                missing.add("response_identity_unavailable")
            if message_id in old_ids:
                continue
            usage = token_usage_from_mapping(message.get("usage"))
            if usage.total_tokens is None:
                if message_id not in responses:
                    missing.add(message_id)
                continue
            missing.discard(message_id)
            agent = (
                event.get("agentId")
                or event.get("parent_tool_use_id")
                or (path if "subagents/" in path else "main")
            )
            earlier = responses.get(message_id)
            # A child's copied main context is not a new request.
            if earlier and earlier["native"] and earlier["agent"] == "main" and agent != "main":
                continue
            responses[message_id] = {
                "message_id": message_id,
                "path": path,
                "agent": agent,
                "usage": usage,
                "native": is_native,
            }

    observed = sum_token_usages([r["usage"] for r in responses.values()])
    main = sum_token_usages([r["usage"] for r in responses.values() if r["agent"] == "main"])
    warnings = []
    exact = finished and not missing and native_seen and bool(responses)
    if backend == "codex" and codex_totals:
        # Native counters are cumulative per rollout; subtract the start snapshot.
        # Root's stdout total may be session-cumulative or turn-only. Do not add it again.
        total = observed
        exact = exact and (
            _same(observed, terminal)
            or any(_same(total, terminal) for total in codex_totals.values())
        )
        basis = "native_rollout_deltas"
        if not exact:
            warnings.append("codex_native_usage_incomplete_or_unreconciled")
    elif exact and (_same(observed, terminal) or _same(main, terminal)):
        total = observed
        basis = "reconciled_unique_responses"
        if not _same(observed, terminal):
            warnings.append("terminal_excludes_subagents; native child usage included")
    else:
        total = _upper_bound(observed, terminal)
        exact = False
        basis = "unreconciled_provider_counters"
        warnings.append("native response usage and terminal total could not be fully reconciled")
    if not exact and total.total_tokens is not None:
        total = replace(total, measurement="partial")
    groups = {}
    for agent in sorted({str(r["agent"]) for r in responses.values()}):
        rows = [r["usage"] for r in responses.values() if str(r["agent"]) == agent]
        groups[agent] = asdict(sum_token_usages(rows))
    return {
        "backend": backend,
        "total": asdict(total),
        "accounting_basis": basis,
        "provider_credits": _credits(stdout, finished) if backend == "qodercli" else None,
        "terminal": asdict(terminal),
        "observed_responses": asdict(observed),
        "by_agent": groups,
        "response_count": len(responses),
        "responses": [
            {
                **{k: v for k, v in r.items() if k not in {"usage", "native"}},
                "usage": asdict(r["usage"]),
            }
            for r in responses.values()
        ],
        "warnings": warnings,
        "missing_usage": sorted(missing),
        "subagent_coverage": "native_transcripts_and_stream; unexported calls cannot be counted",
    }
