"""Recognize device availability errors inside otherwise successful Eval jobs."""

from __future__ import annotations

from collections.abc import Iterator
import re

from typing import Any

JsonValue = Any

_DEVICE_ERRORS = (
    "all cuda-capable devices are busy or unavailable",
    "all hggc-capable devices are busy or unavailable",
    "cudaerrordevicesunavailable",
    "hggcerrordevicesunavailable",
    "cuda error: system not yet initialized",
    "cudaerrorsystemnotready",
    "no cuda gpus are available",
)


def _messages(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in ("message", "reason", "error", "traceback", "details"):
            yield from _messages(value.get(key))


def _device_error(message: str) -> bool:
    return any(marker in message.lower() for marker in _DEVICE_ERRORS)


def _results(value: JsonValue) -> Iterator[dict[str, JsonValue]]:
    if isinstance(value, list):
        for child in value:
            yield from _results(child)
    elif isinstance(value, dict):
        yield value
        # Follow only the Eval / native ABBA envelope, never source, stdout or metadata.
        for key in ("result", "abba", "sdk_results", "runs", "baseline", "candidate"):
            yield from _results(value.get(key))


def embedded_device_failure(job: dict[str, JsonValue]) -> str | None:
    """Return an explicit device-unavailable diagnostic, never a numerical failure.

    Agate can mark the Job succeeded while Bench reports a failed model.to(device)
    under passed.compile/correctness. Native ABBA nests those results in SDK runs.
    OOM, illegal access, compilation errors, timeouts and unknown errors are not
    inferred to be infrastructure failures. Mixed numerical failures stay negative.
    """
    if job.get("status") not in {"succeeded", "failed"}:
        return None
    messages: list[str] = []
    for result in _results(job.get("result")):
        messages.extend(_messages(result.get("error")))
        # Light's standalone ABBA driver carries compact evaluator results.
        # Inspect only their declared diagnostics, never arbitrary Dev stdout.
        failures = result.get("failures", [])
        if isinstance(failures, list):
            messages.extend(item for item in failures if isinstance(item, str))
            if any(isinstance(item, str) and not _device_error(item)
                   and re.search(r"mismatch|not close|\b(?:nan|inf|infinity|nonfinite)\b", item, re.I)
                   for item in failures):
                return None
        diagnostics = result.get("diagnostics", result.get("actionable_diagnostics", []))
        if isinstance(diagnostics, list):
            for diagnostic in diagnostics:
                messages.extend(_messages(diagnostic))
        passed = result.get("passed")
        if isinstance(passed, dict):
            for stage in ("compile", "correctness", "performance"):
                verdicts = passed.get(stage)
                if not isinstance(verdicts, dict):
                    continue
                entries = [verdicts] if "status" in verdicts else verdicts.values()
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("status") == "failed":
                        messages.extend(_messages(entry))
        for section in ("correctness", "performance"):
            data = result.get(section)
            shapes = data.get("shapes") if isinstance(data, dict) else None
            if not isinstance(shapes, dict):
                continue
            for shape in shapes.values():
                if not isinstance(shape, dict):
                    continue
                messages.extend(_messages(shape.get("error")))
                cases = shape.get("cases")
                if not isinstance(cases, list):
                    continue
                for case in cases:
                    if not isinstance(case, dict):
                        continue
                    messages.extend(_messages(case.get("error")))
                    for field in ("outputs", "mutated_inputs", "unexpected_mutations"):
                        comparisons = case.get(field)
                        if not isinstance(comparisons, list):
                            continue
                        for comparison in comparisons:
                            if not isinstance(comparison, dict):
                                continue
                            errors = list(_messages(comparison.get("error")))
                            if comparison.get("passed") is False and not any(
                                _device_error(error) for error in errors
                            ):
                                return None
                            messages.extend(errors)
    return next((message for message in messages if _device_error(message)), None)
