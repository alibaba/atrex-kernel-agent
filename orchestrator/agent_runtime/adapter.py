from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from .markers import phase_marker_receipts
from .model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
    sum_token_usages,
)


def _counter(usage: Mapping[str, object], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def token_usage_from_mapping(usage: object) -> TokenUsage:
    if not isinstance(usage, Mapping):
        return TokenUsage.unavailable()
    input_tokens = _counter(usage, "input_tokens", "inputTokens", "input")
    output_tokens = _counter(usage, "output_tokens", "outputTokens", "output")
    cache_read_tokens = _counter(
        usage, "cache_read_input_tokens", "cacheReadInputTokens", "cacheRead"
    )
    cache_write_tokens = _counter(
        usage,
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
        "cacheWrite",
    )
    official_total = _counter(usage, "total_tokens", "totalTokens")
    components = (
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
    )
    if official_total is None and all(value is None for value in components):
        return TokenUsage.unavailable()
    total_tokens = official_total
    if total_tokens is None:
        total_tokens = sum(value for value in components if value is not None)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=total_tokens,
        measurement="exact",
    )


def token_usage_from_model_usage(model_usage: object) -> TokenUsage:
    if not isinstance(model_usage, Mapping):
        return TokenUsage.unavailable()
    return sum_token_usages(
        [token_usage_from_mapping(value) for value in model_usage.values()]
    )


def toml_config_value(value: object) -> str:
    """Encode the JSON-compatible subset accepted by Codex `-c key=value`."""
    if value is None or isinstance(value, dict):
        raise ValueError(
            "Codex config values must be strings, numbers, booleans, or arrays"
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Codex floating-point config values must be finite")
    if isinstance(value, list):
        if any(item is None or isinstance(item, (dict, list)) for item in value):
            raise ValueError(
                "Codex config arrays may contain only scalar values"
            )
        if any(
            isinstance(item, float) and not math.isfinite(item) for item in value
        ):
            raise ValueError("Codex floating-point config values must be finite")
    if isinstance(value, (str, int, float, list)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"unsupported Codex config value type: {type(value).__name__}")


def pi_settings_args(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "ATREX_PI_SESSION_SETTINGS must be a JSON object"
        ) from exc
    if not isinstance(settings, dict):
        raise ValueError("ATREX_PI_SESSION_SETTINGS must be a JSON object")
    unknown = set(settings) - {"provider", "model"}
    if unknown:
        raise ValueError(
            "unsupported Pi session setting(s): " + ", ".join(sorted(unknown))
        )
    arguments: list[str] = []
    for key in ("provider", "model"):
        value = settings.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pi {key} setting must be a non-empty string")
        arguments += [f"--{key}", value.strip()]
    return arguments


def codex_settings_args(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "ATREX_CODEX_SESSION_SETTINGS must be a JSON object or an array of key=value strings"
        ) from exc

    pairs: list[str] = []
    if isinstance(settings, dict):
        for key, value in settings.items():
            if not isinstance(key, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+", key
            ):
                raise ValueError(f"invalid Codex config key: {key!r}")
            pairs.append(f"{key}={toml_config_value(value)}")
    elif isinstance(settings, list):
        for item in settings:
            if (
                not isinstance(item, str)
                or "=" not in item
                or item.startswith("=")
            ):
                raise ValueError(
                    "ATREX_CODEX_SESSION_SETTINGS array entries must be key=value strings"
                )
            key = item.split("=", 1)[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
                raise ValueError(f"invalid Codex config key: {key!r}")
            pairs.append(item)
    else:
        raise ValueError(
            "ATREX_CODEX_SESSION_SETTINGS must be a JSON object or an array of key=value strings"
        )

    arguments: list[str] = []
    for pair in pairs:
        arguments += ["-c", pair]
    return arguments


class AgentBackendAdapter(ABC):
    id: str
    settings_variable: str
    capabilities = AgentRuntimeCapabilities(
        terminal_usage=False,
        usage_delta=False,
        phase_marker_receipt=False,
    )

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        ...

    @abstractmethod
    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        ...

    @abstractmethod
    def auth_hint(self) -> str:
        ...


def _phase_marker_receipts(event: Mapping[str, object]):
    event_type = event.get("type")
    roots: list[object] = []
    if event_type == "user":
        message = event.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, Mapping)
                        and item.get("type") == "tool_result"
                        and not item.get("is_error", False)
                    ):
                        roots.append(item.get("content"))
    elif event_type == "tool_result" and not event.get("is_error", False):
        roots.append(event.get("content"))
    elif event_type == "message_end":
        message = event.get("message")
        if (
            isinstance(message, Mapping)
            and message.get("role") == "toolResult"
            and not message.get("isError", False)
        ):
            roots.append(message.get("content"))
    elif event_type == "item.completed":
        item = event.get("item")
        if (
            isinstance(item, Mapping)
            and item.get("type") == "command_execution"
            and item.get("exit_code") == 0
            and item.get("status") == "completed"
        ):
            roots.append(item.get("aggregated_output"))

    for root in roots:
        yield from phase_marker_receipts(root)


def _json_events(stdout: str):
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


class ClaudeLikeAdapter(AgentBackendAdapter):
    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True,
        usage_delta=True,
        phase_marker_receipt=True,
    )

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        terminal = TokenUsage.unavailable()
        sequence = 0
        seen_usage_message_ids: set[str] = set()
        for event in _json_events(stdout):
            event_type = event.get("type")
            if event_type == "result":
                terminal = token_usage_from_mapping(event.get("usage"))
                if terminal.total_tokens is None:
                    terminal = token_usage_from_model_usage(event.get("modelUsage"))
                if terminal.total_tokens is not None:
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=sequence,
                            kind="terminal_usage",
                            usage=terminal,
                        )
                    )
                    sequence += 1
                continue
            usage: object = event.get("usage")
            message = event.get("message")
            if usage is None and isinstance(message, Mapping):
                usage = message.get("usage")
            parsed = token_usage_from_mapping(usage)
            message_id = (
                message.get("id")
                if isinstance(message, Mapping)
                and isinstance(message.get("id"), str)
                else None
            )
            duplicate_usage = bool(
                message_id and message_id in seen_usage_message_ids
            )
            if parsed.total_tokens is not None and not duplicate_usage:
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=sequence,
                        kind="usage_delta",
                        usage=parsed,
                    )
                )
                sequence += 1
                if message_id:
                    seen_usage_message_ids.add(message_id)
            for action, phase, marker_id in _phase_marker_receipts(event):
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=sequence,
                        kind="phase_marker",
                        phase=phase,
                        action=action,
                        marker_id=marker_id,
                    )
                )
                sequence += 1
        if terminal.total_tokens is None:
            deltas = [
                event.usage
                for event in normalized
                if event.kind == "usage_delta" and event.usage is not None
            ]
            terminal = sum_token_usages(deltas)
            if terminal.total_tokens is not None:
                terminal = replace(terminal, measurement="partial")
        return tuple(normalized), terminal


class ClaudeAdapter(ClaudeLikeAdapter):
    id = "claude"
    settings_variable = "ATREX_CLAUDE_SESSION_SETTINGS"

    def __init__(self, humanize_dir: Path) -> None:
        self._humanize_dir = humanize_dir

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        command = [
            "claude",
            "--print",
            "--verbose",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--session-id",
            session_id,
            "--effort",
            reasoning_effort,
        ]
        if settings:
            command += ["--settings", settings]
        marker = self._humanize_dir / "skills" / "humanize-gen-plan" / "SKILL.md"
        if marker.exists():
            command += ["--plugin-dir", str(self._humanize_dir)]
        command.append(prompt)
        return command

    def auth_hint(self) -> str:
        return 'run `claude auth status` and `claude --print "test"` to diagnose'


class QoderAdapter(ClaudeLikeAdapter):
    id = "qodercli"
    settings_variable = "ATREX_QODER_SESSION_SETTINGS"

    def __init__(self, humanize_dir: Path) -> None:
        del humanize_dir

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        events, terminal = super().normalize_stream(stdout)
        usage_events = [event for event in events if event.usage is not None]
        if (
            terminal.total_tokens == 0
            and usage_events
            and all(event.usage.total_tokens == 0 for event in usage_events if event.usage)
        ):
            markers = [event for event in events if event.kind == "phase_marker"]
            return (
                tuple(replace(event, sequence=index) for index, event in enumerate(markers)),
                TokenUsage.unavailable(),
            )
        return events, terminal

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        command = [
            "qodercli",
            "--print",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--session-id",
            session_id,
            "--no-session-persistence",
            "--reasoning-effort",
            reasoning_effort,
        ]
        if settings:
            command += ["--settings", settings]
        command.append(prompt)
        return command

    def auth_hint(self) -> str:
        return 'run `qodercli status` and `qodercli --print "test"` to diagnose'


class PiAdapter(AgentBackendAdapter):
    id = "pi"
    settings_variable = "ATREX_PI_SESSION_SETTINGS"
    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True,
        usage_delta=True,
        phase_marker_receipt=True,
    )

    def __init__(self, humanize_dir: Path) -> None:
        del humanize_dir

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        command = [
            "pi",
            "--mode",
            "json",
            "--session-id",
            session_id,
            "--approve",
            "--thinking",
            reasoning_effort,
        ]
        command += pi_settings_args(settings)
        command.append(prompt)
        return command

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        deltas: list[TokenUsage] = []
        settled = False
        for event in _json_events(stdout):
            if event.get("type") == "message_end":
                message = event.get("message")
                if isinstance(message, Mapping) and message.get("role") in {
                    "assistant",
                    "toolResult",
                }:
                    usage = token_usage_from_mapping(message.get("usage"))
                    if usage.total_tokens is not None:
                        deltas.append(usage)
                        normalized.append(
                            NormalizedAgentEvent(
                                sequence=len(normalized),
                                kind="usage_delta",
                                usage=usage,
                            )
                        )
                for action, phase, marker_id in _phase_marker_receipts(event):
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="phase_marker",
                            phase=phase,
                            action=action,
                            marker_id=marker_id,
                        )
                    )
            elif event.get("type") == "compaction_end":
                result = event.get("result")
                usage = token_usage_from_mapping(
                    result.get("usage") if isinstance(result, Mapping) else None
                )
                if usage.total_tokens is not None:
                    deltas.append(usage)
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="usage_delta",
                            usage=usage,
                        )
                    )
            elif event.get("type") == "agent_settled":
                settled = True

        terminal = sum_token_usages(deltas)
        if terminal.total_tokens is not None:
            if not settled:
                terminal = replace(terminal, measurement="partial")
            else:
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=len(normalized),
                        kind="terminal_usage",
                        usage=terminal,
                    )
                )
        return tuple(normalized), terminal

    def auth_hint(self) -> str:
        return 'run `pi --list-models` and `pi -p "reply ok"` to diagnose'


class CodexAdapter(AgentBackendAdapter):
    id = "codex"
    settings_variable = "ATREX_CODEX_SESSION_SETTINGS"
    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True,
        usage_delta=False,
        phase_marker_receipt=True,
    )

    def __init__(self, humanize_dir: Path) -> None:
        del humanize_dir

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        del session_id
        command = [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        command += codex_settings_args(settings)
        command.append(prompt)
        return command

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        terminal = TokenUsage.unavailable()
        for event in _json_events(stdout):
            if event.get("type") in {"turn.completed", "result"}:
                raw_usage = event.get("usage")
                terminal = token_usage_from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping):
                    cached = _counter(
                        raw_usage, "cached_input_tokens", "cachedInputTokens"
                    )
                    if cached is not None:
                        terminal = TokenUsage(
                            input_tokens=terminal.input_tokens,
                            output_tokens=terminal.output_tokens,
                            cache_read_tokens=cached,
                            cache_write_tokens=terminal.cache_write_tokens,
                            total_tokens=terminal.total_tokens,
                            measurement=terminal.measurement,
                        )
                if terminal.total_tokens is not None:
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="terminal_usage",
                            usage=terminal,
                        )
                    )
                continue
            for action, phase, marker_id in _phase_marker_receipts(event):
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=len(normalized),
                        kind="phase_marker",
                        phase=phase,
                        action=action,
                        marker_id=marker_id,
                    )
                )
        return tuple(normalized), terminal

    def auth_hint(self) -> str:
        return 'run `codex login status` and `codex exec "reply ok"` to diagnose'


AdapterFactory = Callable[[Path], AgentBackendAdapter]


class BackendAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, runtime_id: str, factory: AdapterFactory) -> None:
        if not runtime_id or runtime_id in self._factories:
            raise ValueError(f"agent backend already registered: {runtime_id!r}")
        self._factories[runtime_id] = factory

    def create(self, runtime_id: str, humanize_dir: Path) -> AgentBackendAdapter:
        factory = self._factories.get(runtime_id)
        if factory is None:
            raise ValueError(f"unsupported agent CLI: {runtime_id!r}")
        adapter = factory(humanize_dir)
        if adapter.id != runtime_id:
            raise ValueError(
                "agent backend registry key "
                f"{runtime_id!r} does not match adapter id {adapter.id!r}"
            )
        return adapter

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._factories)


DEFAULT_BACKEND_REGISTRY = BackendAdapterRegistry()
DEFAULT_BACKEND_REGISTRY.register("claude", ClaudeAdapter)
DEFAULT_BACKEND_REGISTRY.register("qodercli", QoderAdapter)
DEFAULT_BACKEND_REGISTRY.register("codex", CodexAdapter)
DEFAULT_BACKEND_REGISTRY.register("pi", PiAdapter)
