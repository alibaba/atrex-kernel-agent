"""Provider counters with a clearly labelled conversation-text fallback.

Claude totals use unique main-response counters throughout the invocation.
Provider settlements and child counters are retained separately for audit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Any

from aka.agent.adapter import (
    pi_event_usage,
    token_usage_from_mapping,
    token_usage_from_model_usage,
)
from aka.agent.model import (
    TokenUsage,
    merge_token_usage_evidence,
    subtract_token_usage,
    sum_token_usages,
)

COMPONENTS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens",
)


def _text_tokens(text: str) -> int:
    """Return a deterministic provider-neutral approximation.

    CJK code points are commonly close to one token each. Remaining UTF-8 text
    uses the conventional four-bytes-per-token heuristic. This intentionally
    avoids pretending that one provider/model tokenizer applies to every Agent.
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for character in text:
        code = ord(character)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            cjk += 1
        else:
            other += len(character.encode("utf-8"))
    return cjk + ((other + 3) // 4 if other else 0)


def _content_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value)


def _reasoning_text(value: object) -> str:
    """Extract reasoning/thinking payloads without adding them twice to total."""
    found: list[str] = []

    def visit(item: object, *, reasoning: bool = False) -> None:
        if isinstance(item, str):
            if reasoning:
                found.append(item)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, reasoning=reasoning)
            return
        if not isinstance(item, dict):
            return
        kind = str(item.get("type", "")).casefold()
        local = reasoning or kind in {"thinking", "reasoning", "reasoning_content"}
        for key, child in item.items():
            if key in {"type", "signature", "id"}:
                continue
            child_reasoning = local or key.casefold() in {
                "thinking",
                "reasoning",
                "reasoning_content",
            }
            visit(child, reasoning=child_reasoning)

    visit(value)
    return "\n".join(found)


def _event_content(event: dict[str, Any]) -> tuple[str | None, object]:
    message = event.get("message")
    if isinstance(message, dict):
        role = message.get("role")
        if not isinstance(role, str):
            role = (
                "assistant"
                if event.get("type") in {"assistant", "message", "message_end"}
                else None
            )
        return role, message.get("content", message.get("text"))

    item = event.get("item")
    if isinstance(item, dict):
        kind = str(item.get("type", "")).casefold()
        if kind in {"reasoning", "agent_message", "assistant_message", "function_call"}:
            return "assistant", item
        if kind in {"function_call_output", "tool_result", "user_message"}:
            return "user", item

    event_type = str(event.get("type", "")).casefold()
    role = event.get("role")
    if not isinstance(role, str):
        if event_type.startswith("assistant/") or event_type in {
            "assistant",
            "agent_message",
            "reasoning",
        }:
            role = "assistant"
        elif event_type.startswith("user/") or event_type in {
            "user",
            "tool_result",
            "function_call_output",
        }:
            role = "user"
        else:
            return None, None
    for key in ("content", "text", "output", "arguments", "result"):
        if key in event:
            return role, event[key]
    return None, None


class _TextEstimate:
    """Counters equivalent to estimating a newline-joined list of texts."""
    def __init__(self):
        self.characters = self.cjk = self.other_bytes = self.count = 0

    def append(self, text):
        if self.count:
            self.characters += 1
            self.other_bytes += 1
        self.count += 1
        self.characters += len(text)
        for char in text:
            code = ord(char)
            if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
                self.cjk += 1
            else:
                self.other_bytes += len(char.encode("utf-8"))

    def __bool__(self):
        return bool(self.count)

    def combine(self, other):
        result = _TextEstimate()
        separator = int(bool(self) and bool(other))
        result.characters = self.characters + other.characters + separator
        result.other_bytes = self.other_bytes + other.other_bytes + separator
        result.cjk = self.cjk + other.cjk
        result.count = self.count + other.count
        return result

    def tokens(self):
        return self.cjk + (self.other_bytes + 3) // 4


def estimate_usage_from_conversation(conversation: str) -> tuple[TokenUsage, dict[str, object]]:
    """Estimate a finalized invocation when every provider counter is absent.

    The reading projection is already de-duplicated where the backend exposes
    overlapping stream/native transcripts. The estimate counts only semantic
    messages and tool payloads, never JSON envelope metadata. Replayed context,
    hidden system prompts and provider cache behavior cannot be reconstructed.
    """
    runtime_inputs = _TextEstimate()
    stream_inputs = _TextEstimate()
    stream_outputs = _TextEstimate()
    stream_reasoning = _TextEstimate()
    native_inputs = _TextEstimate()
    native_outputs = _TextEstimate()
    native_reasoning = _TextEstimate()

    for line in conversation.splitlines() if isinstance(conversation, str) else conversation:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        record_type = record.get("type")
        if record_type == "message" and record.get("source") == "runtime_input":
            text = _content_text(record.get("content"))
            if text:
                runtime_inputs.append(text)
            continue
        if record_type != "provider_event":
            continue
        event = record.get("event")
        if not isinstance(event, dict):
            continue
        role, content = _event_content(event)
        text = _content_text(content)
        if not role or not text:
            continue
        is_stream = record.get("path") == "provider/stdout.stream-json"
        inputs = stream_inputs if is_stream else native_inputs
        outputs = stream_outputs if is_stream else native_outputs
        reasoning = stream_reasoning if is_stream else native_reasoning
        if role in {"user", "system", "tool"}:
            inputs.append(text)
        elif role == "assistant":
            outputs.append(text)
            thought = _reasoning_text(content)
            if thought:
                reasoning.append(thought)

    # Prefer the functional stream to avoid counting a native copy twice. If
    # it carries no semantic messages, fall back to captured native transcripts.
    prefer_stream = bool(stream_outputs)
    provider_inputs = stream_inputs if prefer_stream else native_inputs
    provider_outputs = stream_outputs if prefer_stream else native_outputs
    provider_reasoning = stream_reasoning if prefer_stream else native_reasoning
    input_text = runtime_inputs.combine(provider_inputs)
    output_text = provider_outputs
    reasoning_text = provider_reasoning
    input_tokens = input_text.tokens()
    output_tokens = output_text.tokens()
    reasoning_tokens = reasoning_text.tokens()
    total_tokens = input_tokens + output_tokens
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=None,
        cache_write_tokens=None,
        total_tokens=total_tokens,
        measurement="estimated",
        reasoning_tokens=reasoning_tokens,
    )
    return usage, {
        "method": "conversation_text_cjk_plus_utf8_bytes_v1",
        "input_characters": input_text.characters,
        "output_characters": output_text.characters,
        "reasoning_characters": reasoning_text.characters,
        "limitations": [
            "hidden provider system prompts are unavailable",
            "replayed context and cache billing cannot be reconstructed",
            "the estimate is provider-tokenizer independent",
        ],
    }


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
    cache_write = raw.get("cache_write_input_tokens", 0)
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in (cached, cache_write)
    ):
        return TokenUsage.unavailable()
    if value.input_tokens is None or value.input_tokens < cached:
        return TokenUsage.unavailable()
    return replace(
        value,
        input_tokens=value.input_tokens - cached,
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
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


class UsageAccumulator:
    """Fold newly captured records once; retain counters, never transcript text."""

    def __init__(self, backend: str, *, max_entries: int = 100_000):
        self.backend = backend
        if backend == "dsh":
            from aka.agent.dsh import UsageLedger
            self.dsh = UsageLedger()
            self.dsh_main_id = ""
            self.dsh_seeded = {}
            self.dsh_sessions = {}
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
        self.pi_settled = False

    def _usable_usage(self, usage: TokenUsage) -> TokenUsage:
        # Qoder emits zero placeholders when it only reports credits. Match
        # QoderAdapter: those placeholders must not replace unavailable tokens.
        if self.backend == "qodercli" and usage.total_tokens == 0:
            return TokenUsage.unavailable()
        return usage

    def _feed_pi(self, path: str, event: dict, index: int, is_native: bool) -> None:
        if not is_native and event.get("type") == "agent_settled":
            self.pi_settled = True
        if not is_native and event.get("type") in {"agent_start", "message_start", "compaction_start"}:
            self.pi_settled = False
        # Native entry IDs are not stream response IDs. Keep sources separate,
        # including repeated identical responses within the same source.
        if is_native and event.get("type") == "message":
            event = {**event, "type": "message_end"}
        elif is_native and event.get("type") == "compaction":
            event = {**event, "type": "compaction_end", "result": event}
        usage = pi_event_usage(event)
        if usage.total_tokens is None:
            return
        if not is_native:
            self.pi_settled = False
        key = f"{path}:line-{index}"
        self.responses[key] = {
            "message_id": key, "path": path, "agent": "main" if not is_native else path,
            "usage": usage, "native": is_native, "kind": event["type"],
        }
    def _pi_evidence(self, finished: bool) -> tuple[list[dict], TokenUsage, bool, list[str]]:
        by_path: dict[str, list[dict]] = {}
        for row in self.responses.values():
            by_path.setdefault(row["path"], []).append(row)
        stream = by_path.pop("provider/stdout.stream-json", [])
        self.terminal = sum_token_usages([row["usage"] for row in stream])
        if not self.pi_settled and self.terminal.total_tokens is not None:
            self.terminal = replace(self.terminal, measurement="partial")
        total = self.terminal
        rows = stream
        exact = finished and self.pi_settled and not self.missing and bool(stream)
        warnings = []
        for native in by_path.values():
            native_total = sum_token_usages([row["usage"] for row in native])
            if native_total.total_tokens is None:
                continue
            if any(
                getattr(native_total, key) is not None
                and (getattr(total, key) is None or getattr(native_total, key) > getattr(total, key))
                for key in (*COMPONENTS, "total_tokens")
            ):
                if total.total_tokens is None or native_total.total_tokens > total.total_tokens:
                    rows = native
                total = merge_token_usage_evidence(total, native_total)
                exact = False
                warnings = ["pi_stream_native_usage_mismatch"]
        # A settled full stream already includes compaction/tool usage. Native
        # copies can preserve missing counters, but cannot be added without a
        # shared response identity; native entry IDs do not prove disjointness.
        if not exact and total.total_tokens is not None:
            total = replace(total, measurement="partial")
        if not exact and not warnings:
            warnings.append("pi_stream_usage_incomplete")
        return rows, total, exact, warnings

    def mark_history_incomplete(self, path: str) -> None:
        self.missing.add(path + ":resume_history_incomplete")
        self.prior_codex.pop(path, None)

    def _feed_dsh_native(self, path, event, index, *, previous=False):
        from aka.agent.dsh import native_usage
        if event.get("type") == "session":
            self.dsh_seeded[path] = event.get("isSeeded") is True
            if isinstance(event.get("id"), str) and event["id"]:
                self.dsh_sessions[path] = event["id"]
        elif event.get("type") == "session/end-seed":
            self.dsh_seeded[path] = False
        elif (not self.dsh_seeded.get(path, False)
              and event.get("type") in {"assistant/message", "assistant/attempt"}):
            session = self.dsh_sessions.get(path, path.split('/')[-2])
            sequence = event.get("seq")
            if type(sequence) is not int or sequence < 0:
                if not previous:
                    self.missing.add(path + ":dsh_missing_sequence")
                sequence = index
            # A new JSONL generation preserves logical Session/sequence IDs.
            # Keep those IDs stable across migrations and invocation history.
            identity = f"dsh-native:{session}:{sequence}"
            if previous:
                self.old_ids.add(identity)
                self.responses.pop(identity, None)
                return
            if identity in self.old_ids:
                return
            usage = native_usage(event)
            if usage.total_tokens is None:
                self.missing.add(identity + ":usage_unavailable")
            prior = self.responses.get(identity)
            if prior is not None and prior["usage"] != usage:
                self.missing.add(identity + ":usage_changed")
            self.responses[identity] = {
                "message_id": identity, "path": path,
                "agent": session, "usage": usage, "native": True,
            }

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
                if is_native and not previous:
                    self.missing.add(path + ":malformed-json")
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            body = event.get("payload")
            if previous:
                if self.backend == "dsh" and is_native:
                    self._feed_dsh_native(path, event, index, previous=True)
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
            if self.backend == "pi":
                self._feed_pi(path, event, index, is_native)
                continue
            if self.backend == "dsh":
                if is_native:
                    self._feed_dsh_native(path, event, index)
                else:
                    if event.get("type") == "session" and isinstance(event.get("sessionId"), str):
                        self.dsh_main_id = event["sessionId"]
                    key, usage = self.dsh.feed(event)
                    if usage is not None:
                        identity = f"dsh:{key[0]}:{key[1]}"
                        self.responses[identity] = {
                            "message_id": identity, "path": path, "agent": "main",
                            "usage": usage, "native": False,
                        }
                continue
            if not is_native and event.get("type") in {"result", "turn.completed"}:
                self.credit_event = line
                if self.backend == "claude":
                    self.terminal = token_usage_from_mapping(event.get("usage"))
                else:
                    self.terminal = (
                        _codex_usage(event.get("usage"))
                        if self.backend == "codex"
                        else self._usable_usage(token_usage_from_model_usage(event.get("modelUsage")))
                    )
                    if self.terminal.total_tokens is None:
                        self.terminal = self._usable_usage(token_usage_from_mapping(event.get("usage")))
                continue
            if is_native and isinstance(body, dict) and body.get("type") == "token_count":
                info = body.get("info") or {}
                current = _codex_usage(info.get("total_token_usage"))
                if (
                    path + ":resume_history_incomplete" in self.missing
                    and path not in self.codex_totals
                ):
                    # A capped prefix cannot establish the invocation baseline.
                    # Do not count the whole historic cumulative total as new usage.
                    # The first live counter establishes a lower-bound cursor only.
                    if current.total_tokens is not None:
                        self.codex_totals[path] = current
                    continue
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
            usage = self._usable_usage(token_usage_from_mapping(message.get("usage")))
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
        responses = list(self.responses.values())
        if self.backend == "dsh":
            # Root native attempts are already folded into headless step usage.
            # A scoped child's own settlements are disjoint; seeded history is
            # excluded while reading and resumed prefixes are never re-fed.
            responses = [row for row in responses if not row["native"]
                         or (self.dsh_main_id and row["agent"] != self.dsh_main_id)]
        if self.backend == "pi":
            responses, pi_total, pi_exact, pi_warnings = self._pi_evidence(finished)
        observed = sum_token_usages([r["usage"] for r in responses])
        main = sum_token_usages(
            [r["usage"] for r in responses if r["agent"] == "main"]
        )
        warnings = []
        exact = finished and not self.missing and self.native_seen and bool(self.responses)
        if self.backend == "dsh":
            self.terminal = self.dsh.total()
            children = [row["usage"] for row in responses if row["native"]]
            total = sum_token_usages([self.terminal, *children]) if children else self.terminal
            exact = finished and not self.missing and self.native_seen and total.measurement == "exact"
            basis = "dsh_root_steps_and_native_child_attempts"
            if not exact:
                warnings.append("dsh_step_usage_incomplete")
        elif self.backend == "pi":
            total, exact, warnings = pi_total, pi_exact, pi_warnings
            basis = "pi_settled_stream" if exact else "pi_source_counter_bounds"
        elif self.backend == "claude":
            # A terminal result can cover only the final provider turn. Never
            # replace the invocation's accumulated responses when it arrives.
            total = main
            basis = "main_response_usage"
            if main.total_tokens is None:
                warnings.append("claude_main_response_usage_unavailable")
            if self.terminal.total_tokens is not None and not _same(main, self.terminal):
                warnings.append("claude_terminal_usage_differs_from_responses")
        elif self.backend == "codex" and self.codex_totals:
            # Native counters are cumulative per rollout; subtract the start snapshot.
            # Root's stdout total may be session-cumulative or turn-only. Do not add it again.
            total = observed
            exact = exact and (
                _same(observed, self.terminal)
                or any(
                    _same(rollout_total, self.terminal)
                    for rollout_total in self.codex_totals.values()
                )
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
            total = merge_token_usage_evidence(observed, self.terminal)
            exact = False
            basis = "unreconciled_provider_counters"
            warnings.append(
                "native response usage and terminal total could not be fully reconciled"
            )
        if not exact and total.total_tokens is not None:
            total = replace(total, measurement="partial")
        agent_rows = {}
        for response in responses:
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
            "response_count": len(responses),
            "responses": [
                {
                    **{k: v for k, v in r.items() if k not in {"usage", "native"}},
                    "usage": asdict(r["usage"]),
                }
                for r in responses
            ],
            "warnings": warnings,
            "missing_usage": sorted(self.missing),
            "subagent_coverage": (
                "headless root steps plus selected native child attempts; seeded history excluded"
                if self.backend == "dsh"
                else "pi_stream; native-only evidence conservatively merged without shared response IDs"
                if self.backend == "pi"
                else "native child counters are diagnostic only; not added to main responses"
                if self.backend == "claude"
                else "native_transcripts_and_stream; unexported calls cannot be counted"
            ),
        }
