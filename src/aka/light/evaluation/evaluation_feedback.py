"""Bounded Agate diagnostics and timing evidence, without evaluator input material.

Error text is upstream text, not a Runtime paraphrase. The surrounding fields only
identify which stage/opaque shape/output failed. Never include input generators,
sample seeds, input values or whole job/request documents in the Agent response.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from typing import Any, cast

JsonValue = Any

FEEDBACK_KEYS = ("diagnostics", "performance_by_shape", "feedback_truncation")
_MAX_DIAGNOSTICS = 128
_MAX_FEEDBACK_BYTES = 256 * 1024
_MAX_KERNELS = 64


def _count(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _text(value: object) -> str:
    # Import lazily: projection also consumes this module's bounded feedback.
    from aka.light.supervisor.projection import credential_values, redact_diagnostics
    text = redact_diagnostics(str(value), private_values=credential_values(dict(os.environ)))
    return text.encode("utf-8")[:8192].decode("utf-8", errors="ignore")


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def evaluation_feedback(raw: object, shape_ids: Iterable[str]) -> dict[str, JsonValue]:
    """Extract actionable failures and compact timing data from one native Eval result."""
    if not isinstance(raw, dict):
        return {}
    payload = raw.get("result", raw)
    if not isinstance(payload, dict):
        return {}
    allowed = set(shape_ids)
    diagnostics: list[JsonValue] = []
    performance: dict[str, JsonValue] = {}
    omitted = 0

    def add(stage: str, shape_id: str, error: object, **context: JsonValue) -> None:
        nonlocal omitted
        if not error:
            return
        if len(diagnostics) >= _MAX_DIAGNOSTICS:
            omitted += 1
            return
        original = str(error)
        message = _text(original)
        diagnostics.append(
            {
                "stage": stage,
                "shape_id": shape_id,
                **context,
                "error": message,
                **({"text_truncated": True} if len(original.encode()) > 8192 else {}),
            }
        )

    # Top-level error text is evidence too; never copy whole job/request objects.
    records = (
        (("job", raw), ("evaluation", payload))
        if payload is not raw
        else (("evaluation", payload),)
    )
    for stage, record in records:
        error = record.get("error")
        if isinstance(error, str):
            add(stage, "", error)
        elif isinstance(error, dict):
            add(stage, "", error.get("message") or error.get("reason"))

    for stage in ("compile", "correctness", "performance"):
        section = payload.get(stage)
        shapes = section.get("shapes", {}) if isinstance(section, dict) else {}
        if not isinstance(shapes, dict):
            continue
        for sid, value in shapes.items():
            if sid not in allowed or not isinstance(value, dict):
                continue
            add(stage, sid, value.get("error"))
            cases = value.get("cases", [])
            if isinstance(cases, list):
                for index, case in enumerate(cases):
                    if not isinstance(case, dict):
                        continue
                    add(stage, sid, case.get("error"), case_index=index)
                    for kind in ("outputs", "mutated_inputs", "unexpected_mutations"):
                        checks = case.get(kind, [])
                        if not isinstance(checks, list):
                            continue
                        for check in checks:
                            if not isinstance(check, dict) or check.get("passed") is not False:
                                continue
                            add(
                                stage,
                                sid,
                                check.get("error")
                                or "Agate reported a failed check without an error message",
                                case_index=index,
                                check=kind,
                                name=_text(check.get("name", "")),
                                **{
                                    k: check[k]
                                    for k in (
                                        "max_elementwise_abs_diff",
                                        "max_elementwise_rel_diff",
                                        "relative_l2",
                                    )
                                    if _number(check.get(k))
                                },
                            )
            if stage == "performance":
                info: dict[str, JsonValue] = {
                    k: cast(JsonValue, value[k])
                    for k in (
                        "benchmark_mode",
                        "cache_flush_mb",
                        "capture_time_ms",
                    )
                    if isinstance(value.get(k), (str, bool)) or _number(value.get(k))
                }
                stats = value.get("timing_stats")
                if isinstance(stats, dict):
                    info["timing_stats"] = {
                        k: v
                        for k, v in stats.items()
                        if _number(v)
                    }
                kernels = value.get("kernel_events")
                if isinstance(kernels, list):
                    info["kernel_events"] = [
                        {
                            k: (_text(v) if isinstance(v, str) else v)
                            for k, v in kernel.items()
                            if k in ("name", "calls", "device_time_us")
                            and (isinstance(v, str) or _number(v))
                        }
                        for kernel in kernels[:_MAX_KERNELS]
                        if isinstance(kernel, dict)
                    ]
                    if len(kernels) > _MAX_KERNELS:
                        info["omitted_kernel_events"] = len(kernels) - _MAX_KERNELS
                if info:
                    performance[sid] = info

    passed = payload.get("passed")
    if isinstance(passed, dict):
        for stage in ("compile", "correctness", "performance"):
            verdicts = passed.get(stage, {})
            if isinstance(verdicts, dict):
                for sid, verdict in verdicts.items():
                    if (
                        sid in allowed
                        and isinstance(verdict, dict)
                        and str(verdict.get("status", "")).lower() in ("failed", "fail", "error")
                        and not any(
                            isinstance(d, dict)
                            and d.get("shape_id") == sid
                            and d.get("stage") == stage
                            for d in diagnostics
                        )
                    ):
                        add(stage, sid, verdict.get("reason") or "Agate reported stage failure")
    return bound_feedback(
        {
            "diagnostics": diagnostics,
            "performance_by_shape": performance,
            **({"feedback_truncation": {"omitted_diagnostics": omitted}} if omitted else {}),
        }
    )


def bound_feedback(value: dict[str, JsonValue], *, max_bytes: int = _MAX_FEEDBACK_BYTES) -> dict[str, JsonValue]:
    """Bound the whole feedback, keeping errors before optional timing detail."""
    raw_diagnostics = value.get("diagnostics")
    raw_performance = value.get("performance_by_shape")
    raw_truncation = value.get("feedback_truncation")
    diagnostics = list(raw_diagnostics[:_MAX_DIAGNOSTICS]) if isinstance(raw_diagnostics, list) else []
    performance = dict(raw_performance) if isinstance(raw_performance, dict) else {}
    truncation = dict(raw_truncation) if isinstance(raw_truncation, dict) else {}
    if isinstance(raw_diagnostics, list) and len(raw_diagnostics) > _MAX_DIAGNOSTICS:
        truncation["omitted_diagnostics"] = _count(truncation.get("omitted_diagnostics")) + len(raw_diagnostics) - _MAX_DIAGNOSTICS
    result: dict[str, JsonValue] = {"diagnostics": diagnostics, "performance_by_shape": performance}
    # Reserve room for the truncation annotation itself, including after merging.
    while len(json.dumps(result, ensure_ascii=False).encode()) > max_bytes - 1024:
        if performance:
            performance.pop(next(reversed(performance)))
            key = "omitted_performance_shapes"
        elif diagnostics:
            diagnostics.pop()
            key = "omitted_diagnostics"
        else:
            break
        truncation[key] = _count(truncation.get(key)) + 1
    if truncation:
        result["feedback_truncation"] = {
            **truncation,
            "reason": "Runtime response size limit; original errors retained first",
        }
    return result


def merge_feedback(values: Iterable[object]) -> dict[str, JsonValue]:
    """Carry projected feedback through batch aggregation without re-paraphrasing errors."""
    diagnostics: list[JsonValue] = []
    performance: dict[str, JsonValue] = {}
    omitted: dict[str, JsonValue] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        ds = value.get("diagnostics")
        if isinstance(ds, list):
            diagnostics.extend(ds)
        ps = value.get("performance_by_shape")
        if isinstance(ps, dict):
            performance.update(ps)
        truncation = value.get("feedback_truncation")
        if isinstance(truncation, dict):
            for key, count in truncation.items():
                if isinstance(count, int):
                    omitted[key] = _count(omitted.get(key)) + count
    return bound_feedback(
        {
            "diagnostics": diagnostics,
            "performance_by_shape": performance,
            **({"feedback_truncation": omitted} if omitted else {}),
        }
    )


def project_feedback(value: dict) -> dict:
    """Validate public feedback again when reading stored/remote result markers."""
    diagnostics = []
    for row in value.get("diagnostics", []) if isinstance(value.get("diagnostics"), list) else []:
        if not isinstance(row, dict):
            continue
        item = {key: _text(row[key]) for key in ("stage", "shape_id", "check", "name", "error")
                if isinstance(row.get(key), str)}
        item.update({key: row[key] for key in ("case_index", "max_elementwise_abs_diff",
                    "max_elementwise_rel_diff", "relative_l2") if _number(row.get(key))})
        if row.get("text_truncated") is True:
            item["text_truncated"] = True
        diagnostics.append(item)
    # Reuse the same timing allowlist used for native payloads.
    performance = value.get("performance_by_shape", {})
    performance = performance if isinstance(performance, dict) else {}
    projected = evaluation_feedback({"performance": {"shapes": performance}}, performance)
    projected["diagnostics"] = diagnostics
    truncation = value.get("feedback_truncation", {})
    projected["feedback_truncation"] = {
        key: count for key, count in truncation.items()
        if key.startswith("omitted_") and type(count) is int and count >= 0
    } if isinstance(truncation, dict) else {}
    return bound_feedback(projected)


def optimizer_result_with_feedback(payload, shape_ids, metadata, *, require_performance=True):
    from aka.legacy.evaluation.results import optimizer_result_from_eval
    from .public_result import correctness_summary
    result = optimizer_result_from_eval(payload, shape_ids, metadata, require_performance=require_performance)
    return {**result, "correctness": correctness_summary(payload, passed=result["all_pass"]),
            **evaluation_feedback(payload, shape_ids)}


def merge_results_with_feedback(results, shape_ids, metadata, *, require_performance=True):
    from aka.legacy.evaluation.results import merge_optimizer_results
    from .public_result import correctness_summary
    result = merge_optimizer_results(results, shape_ids, metadata, require_performance=require_performance)
    return {**result, "correctness": correctness_summary(results, passed=result["all_pass"]), **merge_feedback(results)}
