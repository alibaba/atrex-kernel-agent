from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ..agent_runtime.model import (
        AgentRuntimeCapabilities,
        NormalizedAgentEvent,
        TokenUsage,
        subtract_token_usage,
        sum_token_usages,
        token_usage_exceeds,
    )
except ImportError:  # direct script execution: python orchestrator/optimize.py
    from agent_runtime.model import (  # type: ignore[no-redef]
        AgentRuntimeCapabilities,
        NormalizedAgentEvent,
        TokenUsage,
        subtract_token_usage,
        sum_token_usages,
        token_usage_exceeds,
    )


PHASES = (
    "profile",
    "research",
    "planning",
    "implementation",
    "correctness",
    "benchmark",
    "recording",
)


def usage_dict(usage: TokenUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "total_tokens": usage.total_tokens,
        "measurement": usage.measurement,
    }


def _sum_phase_usages(usages: Sequence[TokenUsage]) -> TokenUsage:
    return sum_token_usages(usages) if usages else TokenUsage.zero()


def summarize_phase_tokens(
    *,
    events: Sequence[NormalizedAgentEvent],
    terminal_usage: TokenUsage,
    capabilities: AgentRuntimeCapabilities,
    observation_errors: Sequence[str],
) -> dict[str, Any]:
    """Attribute normalized usage deltas only to complete explicit phase pairs."""
    phase_intervals: dict[str, list[list[TokenUsage]]] = {
        phase: [] for phase in PHASES
    }
    reason_codes = list(observation_errors)
    all_deltas: list[TokenUsage] = []
    active_phase: str | None = None
    active_usage: list[TokenUsage] = []
    pending_start_usage: TokenUsage | None = None
    invalid_stack: list[str] = []

    if not capabilities.usage_delta or not capabilities.usage_delta_observed:
        reason_codes.append(
            "backend_has_no_usage_delta"
            if not capabilities.usage_delta
            else "backend_usage_delta_unobserved"
        )
        return {
            "terminal_usage": usage_dict(terminal_usage),
            "phases": {
                phase: {
                    "usage": None,
                    "interval_count": 0,
                    "measurement": "unavailable",
                    "reason": "phase_not_observed",
                }
                for phase in PHASES
            },
            "orchestration": None,
            "unattributed": (
                usage_dict(terminal_usage)
                if terminal_usage.total_tokens is not None
                else None
            ),
            "coverage": (
                0.0 if terminal_usage.total_tokens is not None else None
            ),
            "semantic_phase_coverage": (
                0.0 if terminal_usage.total_tokens is not None else None
            ),
            "accounted_coverage": (
                0.0 if terminal_usage.total_tokens is not None else None
            ),
            "measurement": "unavailable",
            "reconciliation_status": (
                "reconciled"
                if terminal_usage.total_tokens is not None
                else "unavailable"
            ),
            "reason_codes": sorted(set(reason_codes)),
        }

    for event in events:
        if event.kind == "usage_delta" and event.usage is not None:
            all_deltas.append(event.usage)
            if active_phase is not None and not invalid_stack:
                active_usage.append(event.usage)
                pending_start_usage = None
            elif not invalid_stack:
                pending_start_usage = event.usage
            continue
        if event.kind != "phase_marker":
            continue
        phase = event.phase or ""
        action = event.action
        if phase not in phase_intervals or action not in {"start", "end"}:
            reason_codes.append("invalid_phase_marker")
            pending_start_usage = None
            continue

        if invalid_stack:
            if action == "start":
                invalid_stack.append(phase)
                reason_codes.append("overlapping_phase")
            elif invalid_stack and phase == invalid_stack[-1]:
                invalid_stack.pop()
            elif phase in invalid_stack:
                reason_codes.append("mismatched_phase_end")
                while invalid_stack and invalid_stack[-1] != phase:
                    invalid_stack.pop()
                if invalid_stack:
                    invalid_stack.pop()
            else:
                reason_codes.append("orphan_phase_end")
            pending_start_usage = None
            continue

        if action == "start":
            if active_phase is None:
                active_phase = phase
                active_usage = (
                    [pending_start_usage] if pending_start_usage is not None else []
                )
                pending_start_usage = None
            else:
                reason_codes.append("overlapping_phase")
                invalid_stack = [active_phase, phase]
                active_phase = None
                active_usage = []
                pending_start_usage = None
            continue

        if active_phase is None:
            reason_codes.append("orphan_phase_end")
            pending_start_usage = None
        elif active_phase == phase:
            phase_intervals[phase].append(active_usage)
            active_phase = None
            active_usage = []
        else:
            reason_codes.append("mismatched_phase_end")
            invalid_stack = [active_phase]
            active_phase = None
            active_usage = []

    if active_phase is not None or invalid_stack:
        reason_codes.append("unclosed_phase")

    phase_payload: dict[str, Any] = {}
    attributed_usages: list[TokenUsage] = []
    for phase, intervals in phase_intervals.items():
        if not intervals:
            phase_payload[phase] = {
                "usage": None,
                "interval_count": 0,
                "measurement": "unavailable",
                "reason": "phase_not_observed",
            }
            continue
        interval_usages = [_sum_phase_usages(interval) for interval in intervals]
        phase_usage = _sum_phase_usages(interval_usages)
        attributed_usages.append(phase_usage)
        phase_payload[phase] = {
            "usage": usage_dict(phase_usage),
            "interval_count": len(intervals),
            "measurement": phase_usage.measurement,
            "reason": None,
            "intervals": [
                {"index": index, "usage": usage_dict(interval_usage)}
                for index, interval_usage in enumerate(interval_usages, start=1)
            ],
        }

    attributed = _sum_phase_usages(attributed_usages)
    observed_delta = _sum_phase_usages(all_deltas)
    terminal_total = terminal_usage.total_tokens
    attributed_total = attributed.total_tokens
    observed_delta_total = observed_delta.total_tokens
    orchestration: TokenUsage | None
    unattributed: TokenUsage | None
    accounted_coverage: float | None
    if terminal_total is None:
        orchestration = (
            subtract_token_usage(observed_delta, attributed)
            if observed_delta_total is not None
            and attributed_total is not None
            and not token_usage_exceeds(attributed, observed_delta)
            else None
        )
        unattributed = None
        coverage = None
        accounted_coverage = None
        reconciliation = "unavailable"
        measurement = "partial" if attributed_usages else "unavailable"
        reason_codes.append("terminal_usage_unavailable")
    elif (
        attributed_total is None
        or observed_delta_total is None
        or token_usage_exceeds(attributed, terminal_usage)
        or token_usage_exceeds(observed_delta, terminal_usage)
    ):
        orchestration = None
        unattributed = None
        coverage = None
        accounted_coverage = None
        reconciliation = "inconsistent"
        measurement = "partial"
        reason_codes.append("usage_delta_exceeds_terminal")
    else:
        orchestration = subtract_token_usage(terminal_usage, attributed)
        unattributed = TokenUsage.zero()
        coverage = (
            round(attributed_total / terminal_total, 6)
            if terminal_total > 0
            else (1.0 if attributed_total == 0 else None)
        )
        accounted_coverage = 1.0
        reconciliation = "reconciled"
        measurement = (
            "exact"
            if coverage == 1.0
            and not reason_codes
            and terminal_usage.measurement == "exact"
            and attributed.measurement == "exact"
            else ("partial" if attributed_usages else "unavailable")
        )

    return {
        "terminal_usage": usage_dict(terminal_usage),
        "phases": phase_payload,
        "orchestration": usage_dict(orchestration) if orchestration else None,
        "unattributed": usage_dict(unattributed) if unattributed else None,
        "coverage": coverage,
        "semantic_phase_coverage": coverage,
        "accounted_coverage": accounted_coverage,
        "measurement": measurement,
        "reconciliation_status": reconciliation,
        "reason_codes": sorted(set(reason_codes)),
    }


def _usage_from_dict(payload: object) -> TokenUsage:
    if not isinstance(payload, Mapping):
        return TokenUsage.unavailable()

    def value(name: str) -> int | None:
        item = payload.get(name)
        return int(item) if isinstance(item, int) and not isinstance(item, bool) else None

    measurement = payload.get("measurement")
    return TokenUsage(
        input_tokens=value("input_tokens"),
        output_tokens=value("output_tokens"),
        cache_read_tokens=value("cache_read_tokens"),
        cache_write_tokens=value("cache_write_tokens"),
        total_tokens=value("total_tokens"),
        measurement=(
            str(measurement)
            if measurement in {"exact", "partial", "unavailable"}
            else "unavailable"
        ),
    )


def aggregate_attempt_tokens(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    token_summaries = [
        attempt.get("phase_tokens")
        for attempt in attempts
        if isinstance(attempt.get("phase_tokens"), Mapping)
    ]
    terminal_usages = [
        _usage_from_dict(summary.get("terminal_usage"))
        for summary in token_summaries
    ]
    observed_terminal = [
        usage for usage in terminal_usages if usage.total_tokens is not None
    ]
    terminal = (
        _sum_phase_usages(observed_terminal)
        if observed_terminal
        else TokenUsage.unavailable()
    )
    if len(observed_terminal) != len(attempts) and terminal.total_tokens is not None:
        terminal = TokenUsage(
            terminal.input_tokens,
            terminal.output_tokens,
            terminal.cache_read_tokens,
            terminal.cache_write_tokens,
            terminal.total_tokens,
            "partial",
        )

    phases: dict[str, Any] = {}
    phase_aggregates: list[TokenUsage] = []
    for phase in PHASES:
        usages: list[TokenUsage] = []
        interval_count = 0
        unavailable_attempt = False
        for summary in token_summaries:
            phase_payload = summary.get("phases")
            value = (
                phase_payload.get(phase)
                if isinstance(phase_payload, Mapping)
                else None
            )
            if not isinstance(value, Mapping) or value.get("usage") is None:
                unavailable_attempt = True
                continue
            phase_usage = _usage_from_dict(value.get("usage"))
            if phase_usage.total_tokens is None:
                unavailable_attempt = True
                continue
            usages.append(phase_usage)
            count = value.get("interval_count")
            if isinstance(count, int) and not isinstance(count, bool):
                interval_count += count
        if not usages:
            phases[phase] = {
                "usage": None,
                "interval_count": 0,
                "measurement": "unavailable",
                "reason": "phase_not_observed",
            }
            continue
        aggregate = _sum_phase_usages(usages)
        if unavailable_attempt:
            aggregate = TokenUsage(
                aggregate.input_tokens,
                aggregate.output_tokens,
                aggregate.cache_read_tokens,
                aggregate.cache_write_tokens,
                aggregate.total_tokens,
                "partial",
            )
        phase_aggregates.append(aggregate)
        phases[phase] = {
            "usage": usage_dict(aggregate),
            "interval_count": interval_count,
            "measurement": aggregate.measurement,
            "reason": None,
        }

    displayed_attributed = _sum_phase_usages(phase_aggregates)
    displayed_attributed_total = displayed_attributed.total_tokens or 0
    reasons = {
        str(reason)
        for summary in token_summaries
        for reason in (summary.get("reason_codes") or [])
    }
    attempts_with_terminal = len(observed_terminal)
    if attempts_with_terminal != len(attempts):
        reasons.add("attempt_terminal_usage_unavailable")

    coverage_attributed_usages: list[TokenUsage] = []
    observed_orchestration: list[TokenUsage] = []
    observed_unattributed: list[TokenUsage] = []
    reconciliation_values: set[str] = set()
    for summary in token_summaries:
        attempt_terminal = _usage_from_dict(summary.get("terminal_usage"))
        if attempt_terminal.total_tokens is None:
            continue
        reconciliation_values.add(str(summary.get("reconciliation_status")))
        phase_payload = summary.get("phases")
        if isinstance(phase_payload, Mapping):
            for phase in PHASES:
                value = phase_payload.get(phase)
                if not isinstance(value, Mapping):
                    continue
                phase_usage = _usage_from_dict(value.get("usage"))
                if phase_usage.total_tokens is not None:
                    coverage_attributed_usages.append(phase_usage)
        attempt_orchestration = _usage_from_dict(summary.get("orchestration"))
        if attempt_orchestration.total_tokens is not None:
            observed_orchestration.append(attempt_orchestration)
        attempt_unattributed = _usage_from_dict(summary.get("unattributed"))
        if attempt_unattributed.total_tokens is not None:
            observed_unattributed.append(attempt_unattributed)

    coverage_attributed = _sum_phase_usages(coverage_attributed_usages)
    coverage_attributed_total = coverage_attributed.total_tokens or 0
    orchestration: TokenUsage | None
    accounted_coverage: float | None
    if "inconsistent" in reconciliation_values:
        reconciliation = "inconsistent"
        orchestration = None
        unattributed = None
        coverage = None
        accounted_coverage = None
    elif terminal.total_tokens is None:
        reconciliation = "unavailable"
        orchestration = None
        unattributed = None
        coverage = None
        accounted_coverage = None
    elif token_usage_exceeds(coverage_attributed, terminal):
        reconciliation = "inconsistent"
        orchestration = None
        unattributed = None
        coverage = None
        accounted_coverage = None
        reasons.add("usage_delta_exceeds_terminal")
    else:
        reconciliation = "reconciled"
        orchestration = _sum_phase_usages(observed_orchestration)
        unattributed = _sum_phase_usages(observed_unattributed)
        coverage = (
            round(coverage_attributed_total / terminal.total_tokens, 6)
            if terminal.total_tokens > 0
            else (1.0 if coverage_attributed_total == 0 else None)
        )
        accounted_total = coverage_attributed_total + (
            orchestration.total_tokens or 0
        )
        accounted_coverage = (
            round(accounted_total / terminal.total_tokens, 6)
            if terminal.total_tokens > 0
            else (1.0 if accounted_total == 0 else None)
        )

    measurement = (
        "exact"
        if coverage == 1.0
        and attempts_with_terminal == len(attempts)
        and terminal.measurement == "exact"
        and not reasons
        else ("partial" if displayed_attributed_total else "unavailable")
    )
    reasons = sorted(reasons)
    return {
        "terminal_usage": usage_dict(terminal),
        "phases": phases,
        "orchestration": usage_dict(orchestration) if orchestration else None,
        "unattributed": usage_dict(unattributed) if unattributed else None,
        "coverage": coverage,
        "semantic_phase_coverage": coverage,
        "accounted_coverage": accounted_coverage,
        "measurement": measurement,
        "reconciliation_status": reconciliation,
        "reason_codes": reasons,
        "attempts_with_terminal_usage": attempts_with_terminal,
        "attempt_count": len(attempts),
    }
