from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    TokenUsage,
    resequence_agent_events,
    sum_token_usages,
)
from orchestrator.telemetry.phase_tokens import (
    PHASES,
    aggregate_attempt_tokens,
    summarize_phase_tokens,
)

from .models import InvocationObservation


EPISODE_TELEMETRY_SCHEMA_VERSION = "atrex_long_horizon_episode_telemetry_v1"


def _fallback_phase_tokens(control_tokens: int) -> dict[str, Any]:
    terminal = TokenUsage(
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        total_tokens=max(0, int(control_tokens)),
        measurement="partial",
    )
    return summarize_phase_tokens(
        events=(),
        terminal_usage=terminal,
        capabilities=AgentRuntimeCapabilities(
            terminal_usage=False,
            usage_delta=False,
            phase_marker_receipt=False,
        ),
        observation_errors=("structured_usage_unavailable",),
    )


def summarize_episode(
    *,
    episode: int,
    version: int,
    status: str,
    accepted: bool,
    control_tokens: int,
    resume_count: int,
    invocations: Sequence[InvocationObservation],
) -> dict[str, Any]:
    """Summarize one episode without changing its existing control-token contract."""
    invocation_summaries: list[dict[str, Any]] = []
    for index, observation in enumerate(invocations, start=1):
        phase_tokens = summarize_phase_tokens(
            events=observation.events,
            terminal_usage=observation.terminal_usage,
            capabilities=observation.capabilities,
            observation_errors=observation.observation_errors,
        )
        invocation_summaries.append(
            {
                "invocation": index,
                "phase_tokens": phase_tokens,
                "measurement": phase_tokens["measurement"],
            }
        )

    qualified_resume = bool(
        len(invocations) > 1
        and all(observation.resume_usage_qualified for observation in invocations)
    )
    if qualified_resume:
        combined_events = []
        for observation in invocations:
            combined_events.extend(
                event for event in observation.events if event.kind != "terminal_usage"
            )
        combined_events = list(resequence_agent_events(combined_events))
        combined_terminal = sum_token_usages(
            [observation.terminal_usage for observation in invocations]
        )
        phase_tokens = summarize_phase_tokens(
            events=combined_events,
            terminal_usage=combined_terminal,
            capabilities=AgentRuntimeCapabilities(
                terminal_usage=True,
                usage_delta=True,
                phase_marker_receipt=True,
                usage_delta_observed=True,
            ),
            observation_errors=tuple(
                error
                for observation in invocations
                for error in observation.observation_errors
            ),
        )
    else:
        phase_tokens = (
            aggregate_attempt_tokens(invocation_summaries)
            if invocation_summaries
            else _fallback_phase_tokens(control_tokens)
        )
    phase_intervals: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in PHASES
    }
    if qualified_resume:
        for phase in PHASES:
            phase_payload = phase_tokens.get("phases", {}).get(phase, {})
            for interval in phase_payload.get("intervals") or []:
                interval_usage = interval.get("usage")
                if isinstance(interval_usage, Mapping) and isinstance(
                    interval_usage.get("total_tokens"), int
                ):
                    phase_intervals[phase].append(
                        {
                            "invocation": "episode",
                            "index": interval.get("index"),
                            "usage": interval_usage,
                        }
                    )
    for invocation_summary in (() if qualified_resume else invocation_summaries):
        invocation_number = int(invocation_summary["invocation"])
        invocation_tokens = invocation_summary.get("phase_tokens")
        invocation_tokens = (
            invocation_tokens if isinstance(invocation_tokens, Mapping) else {}
        )
        invocation_phases = invocation_tokens.get("phases")
        invocation_phases = (
            invocation_phases if isinstance(invocation_phases, Mapping) else {}
        )
        for phase in PHASES:
            phase_payload = invocation_phases.get(phase)
            phase_payload = phase_payload if isinstance(phase_payload, Mapping) else {}
            for interval in phase_payload.get("intervals") or []:
                if not isinstance(interval, Mapping):
                    continue
                interval_usage = interval.get("usage")
                if (
                    not isinstance(interval_usage, Mapping)
                    or not isinstance(interval_usage.get("total_tokens"), int)
                ):
                    continue
                phase_intervals[phase].append(
                    {
                        "invocation": invocation_number,
                        "index": interval.get("index"),
                        "usage": interval_usage,
                    }
                )
    reasons = set(str(value) for value in phase_tokens.get("reason_codes", []))
    if not invocation_summaries:
        reasons.add("structured_usage_unavailable")
    if (
        (resume_count > 0 or len(invocation_summaries) > 1)
        and not qualified_resume
    ):
        reasons.add("same_session_resume_usage_semantics_unqualified")
    structured_total = (phase_tokens.get("terminal_usage") or {}).get("total_tokens")
    if (
        isinstance(structured_total, int)
        and structured_total != max(0, int(control_tokens))
    ):
        reasons.add("control_token_total_mismatch")

    measurement = str(phase_tokens.get("measurement") or "unavailable")
    if reasons and measurement == "exact":
        measurement = "partial"
    phase_tokens["reason_codes"] = sorted(reasons)
    if reasons and phase_tokens.get("measurement") == "exact":
        phase_tokens["measurement"] = "partial"

    return {
        "schema_version": EPISODE_TELEMETRY_SCHEMA_VERSION,
        "episode": int(episode),
        "version": f"v{int(version)}",
        "status": str(status),
        "accepted": bool(accepted),
        "control_tokens": max(0, int(control_tokens)),
        "resume_count": max(0, int(resume_count)),
        "invocation_count": len(invocation_summaries),
        "invocations": invocation_summaries,
        "phase_tokens": phase_tokens,
        "phase_intervals": phase_intervals,
        "measurement": measurement,
        "reason_codes": sorted(reasons),
    }


def render_episode_brief(summary: Mapping[str, Any]) -> str:
    tokens = summary.get("phase_tokens")
    tokens = tokens if isinstance(tokens, Mapping) else {}
    phases = tokens.get("phases")
    phases = phases if isinstance(phases, Mapping) else {}
    phase_intervals = summary.get("phase_intervals")
    phase_intervals = (
        phase_intervals if isinstance(phase_intervals, Mapping) else {}
    )

    def cell(value: object) -> str:
        return f"{value:,}" if isinstance(value, int) else "—"

    rows: list[str] = []
    for phase in PHASES:
        payload = phases.get(phase)
        payload = payload if isinstance(payload, Mapping) else {}
        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        rows.append(
            "| "
            + " | ".join(
                [
                    phase,
                    cell(usage.get("input_tokens")),
                    cell(usage.get("output_tokens")),
                    cell(usage.get("cache_read_tokens")),
                    cell(usage.get("cache_write_tokens")),
                    cell(usage.get("total_tokens")),
                    str(payload.get("interval_count") or 0),
                    str(payload.get("measurement") or "unavailable"),
                ]
            )
            + " |"
        )
    for label in ("orchestration", "unattributed"):
        usage = tokens.get(label)
        usage = usage if isinstance(usage, Mapping) else {}
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    cell(usage.get("input_tokens")),
                    cell(usage.get("output_tokens")),
                    cell(usage.get("cache_read_tokens")),
                    cell(usage.get("cache_write_tokens")),
                    cell(usage.get("total_tokens")),
                    "—",
                    str(usage.get("measurement") or "unavailable"),
                ]
            )
            + " |"
        )

    interval_rows: list[str] = []
    for phase in PHASES:
        for interval in phase_intervals.get(phase) or []:
            if not isinstance(interval, Mapping):
                continue
            usage = interval.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            interval_rows.append(
                "| "
                + " | ".join(
                    [
                        phase,
                        str(interval.get("invocation") or "—"),
                        str(interval.get("index") or "—"),
                        cell(usage.get("input_tokens")),
                        cell(usage.get("output_tokens")),
                        cell(usage.get("cache_read_tokens")),
                        cell(usage.get("cache_write_tokens")),
                        cell(usage.get("total_tokens")),
                        str(usage.get("measurement") or "unavailable"),
                    ]
                )
                + " |"
            )

    terminal = tokens.get("terminal_usage")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    lines = [
        f"# Long-horizon Episode {summary.get('episode', 'unknown')}",
        "",
        f"- version: `{summary.get('version', 'unknown')}`",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- accepted: `{bool(summary.get('accepted'))}`",
        f"- invocations/resumes: `{summary.get('invocation_count', 0)}` / `{summary.get('resume_count', 0)}`",
        f"- control tokens: `{cell(summary.get('control_tokens'))}`",
        f"- structured terminal tokens: `{cell(terminal.get('total_tokens'))}`",
        f"- semantic coverage: `{tokens.get('semantic_phase_coverage', 'unavailable')}`",
        f"- accounted coverage: `{tokens.get('accounted_coverage', 'unavailable')}`",
        f"- measurement: `{summary.get('measurement', 'unavailable')}`",
        f"- reason codes: `{', '.join(summary.get('reason_codes') or []) or 'none'}`",
        "",
        "## Phase token usage",
        "",
        "| Phase | Input | Output | Cache read | Cache write | Total | Intervals | Measurement |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *rows,
    ]
    if interval_rows:
        lines.extend(
            [
                "",
                "## Phase interval token usage",
                "",
                "| Phase | Invocation | Interval | Input | Output | Cache read | Cache write | Total | Measurement |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                *interval_rows,
            ]
        )
    return "\n".join(lines)
