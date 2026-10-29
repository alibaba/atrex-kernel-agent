"""Complete Agate failure diagnostics and compact timing evidence.

Error text is upstream text, not a Runtime paraphrase. The surrounding fields only
identify which stage/opaque shape/output failed. Never include input generators,
input values or whole job/request documents in the Agent response.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from typing import Any, cast

JsonValue = Any

FEEDBACK_KEYS = ("diagnostics", "performance_by_shape", "feedback_truncation")
_MAX_FEEDBACK_BYTES = 256 * 1024
_MAX_KERNELS = 64


def _count(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _text(value: object) -> str:
    # Import lazily: projection also consumes this module's bounded feedback.
    from aka.light.evaluation.projection import credential_values, redact_diagnostics
    text = redact_diagnostics(str(value), private_values=credential_values(dict(os.environ)))
    return text


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

    def add(stage: str, shape_id: str, error: object, **context: JsonValue) -> None:
        if not error:
            return
        original = str(error)
        message = _text(original)
        diagnostics.append(
            {
                "stage": stage,
                "shape_id": shape_id,
                **context,
                "error": message,
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
            add(stage, "", json.dumps(error, ensure_ascii=False))

    for field in ("stdout", "stderr", "traceback"):
        add("evaluation", "", payload.get(field), log_stream=field)

    for stage in ("compile", "correctness", "performance"):
        section = payload.get(stage)
        if isinstance(section, dict):
            for field in ("error", "reason", "traceback", "stdout", "stderr"):
                add(stage, "", section.get(field), log_stream=field)
        shapes = section.get("shapes", {}) if isinstance(section, dict) else {}
        if not isinstance(shapes, dict):
            continue
        for sid, value in shapes.items():
            if sid not in allowed or not isinstance(value, dict):
                continue
            add(stage, sid, value.get("error"))
            for field in ("reason", "traceback", "stdout", "stderr"):
                add(stage, sid, value.get(field), log_stream=field)
            ranks = value.get("rank_results")
            rows = ([(rank, row) for rank, row in enumerate(ranks) if isinstance(row, dict)]
                    if isinstance(ranks, list) else [])
            for rank, row in rows or [(0, value)]:
                if row is not value:
                    for field in ("error", "reason", "traceback", "stdout", "stderr"):
                        add(stage, sid, row.get(field), rank=rank, log_stream=field)
                for index, case in enumerate(row.get("cases", [])):
                    if not isinstance(case, dict):
                        continue
                    context = {"rank": row.get("rank", rank), "case_index": index}
                    artifact = case.get("input_artifact")
                    if isinstance(artifact, dict) and type(artifact.get("seed")) is int:
                        context["seed"] = artifact["seed"]
                    add(stage, sid, case.get("error"), **context)
                    for kind in ("outputs", "mutated_inputs", "unexpected_mutations"):
                        for check in case.get(kind, []):
                            if not isinstance(check, dict) or check.get("passed") is not False:
                                continue
                            add(stage, sid, check.get("error") or "Evaluator check failed",
                                **context, check=kind, name=_text(check.get("name", "")),
                                tolerance=check.get("tolerance"),
                                **{k: check[k] for k in ("max_elementwise_abs_diff",
                                    "max_elementwise_rel_diff", "relative_l2") if _number(check.get(k))})
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
                if "status" in verdicts:
                    verdicts = {"": verdicts}
                for sid, verdict in verdicts.items():
                    if (
                        (sid in allowed or sid == "")
                        and isinstance(verdict, dict)
                        and str(verdict.get("status", "")).lower() in ("failed", "fail", "error")
                    ):
                        reason = verdict.get("reason")
                        details = [row for row in diagnostics if row["stage"] == stage
                                   and (not sid or row["shape_id"] == sid)]
                        if reason and not any(row["error"] == str(reason) for row in details):
                            add(stage, sid, reason)
                        elif not reason and not details:
                            add(stage, sid, "Agate reported stage failure")
    return bound_performance_feedback(
        {
            "diagnostics": diagnostics,
            "performance_by_shape": performance,
        }
    )


def bound_performance_feedback(value: dict[str, JsonValue], *, max_performance_bytes: int = _MAX_FEEDBACK_BYTES) -> dict[str, JsonValue]:
    """Bound optional timing detail; never discard failure diagnostics."""
    raw_diagnostics = value.get("diagnostics")
    raw_performance = value.get("performance_by_shape")
    raw_truncation = value.get("feedback_truncation")
    diagnostics = list(raw_diagnostics) if isinstance(raw_diagnostics, list) else []
    performance = dict(raw_performance) if isinstance(raw_performance, dict) else {}
    truncation = dict(raw_truncation) if isinstance(raw_truncation, dict) else {}
    result: dict[str, JsonValue] = {"diagnostics": diagnostics, "performance_by_shape": performance}
    # Reserve room for the truncation annotation itself, including after merging.
    while len(json.dumps(performance, ensure_ascii=False).encode()) > max_performance_bytes - 1024:
        if performance:
            performance.pop(next(reversed(performance)))
            key = "omitted_performance_shapes"
        else:
            break
        truncation[key] = _count(truncation.get(key)) + 1
    if truncation:
        result["feedback_truncation"] = {
            **truncation,
            "reason": "Optional timing detail omitted; failure diagnostics retained in full",
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
    return bound_performance_feedback(
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
        item = {key: _text(row[key]) for key in ("stage", "shape_id", "check", "name", "error", "log_stream")
                if isinstance(row.get(key), str)}
        item.update({key: row[key] for key in ("case_index", "rank", "seed", "max_elementwise_abs_diff",
                    "max_elementwise_rel_diff", "relative_l2") if _number(row.get(key))})
        if isinstance(row.get("tolerance"), dict):
            item["tolerance"] = {key: value for key, value in row["tolerance"].items()
                if key in {"mode", "relative_to", "atol", "rtol", "max_rel_l2", "max_mismatch_fraction"}
                and (value is None or isinstance(value, str) or _number(value))}
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
    return bound_performance_feedback(projected)


def optimizer_result_with_feedback(payload, shape_ids, metadata, *, require_performance=True):
    from .result_conversion import optimizer_result_from_eval
    from .public_result import correctness_summary
    result = optimizer_result_from_eval(payload, shape_ids, metadata, require_performance=require_performance)
    return {**result, "correctness": correctness_summary(payload, passed=result["all_pass"]),
            **evaluation_feedback(payload, shape_ids)}


def merge_results_with_feedback(results, shape_ids, metadata, *, require_performance=True):
    from .result_conversion import merge_optimizer_results
    from .public_result import correctness_summary
    result = merge_optimizer_results(results, shape_ids, metadata, require_performance=require_performance)
    return {**result, "correctness": correctness_summary(results, passed=result["all_pass"]), **merge_feedback(results)}
