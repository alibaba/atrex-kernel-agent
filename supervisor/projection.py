"""Compact Agent projections adapted from simplified; no Record/Journal changes."""
from __future__ import annotations
import json
import math
import re
import subprocess
from typing import Any

from supervisor.errors import error_response
from supervisor.gateway_jobs import submission_rejection

MAX_AGENT_PROFILE_KERNELS = 32
MAX_AGENT_EVALUATION_SHAPES = 4096
MAX_AGENT_DISASSEMBLY_BYTES = 64 * 1024
MAX_AGENT_PROFILE_METRICS = 64
MAX_AGENT_DIAGNOSTICS = 16
SOURCE_ERROR_PREFIX = "[sandbox] SOURCE_ERROR_JSON="
GATEWAY_ERROR_PREFIX = "[sandbox] GATEWAY_ERROR_JSON="
MAX_GATEWAY_ERROR_BYTES = 64 * 1024
_SECRET_KEY = re.compile(
    r"(?:authorization|password|ak|sk|atrex_aka_identity_key|agate[_-]?(?:ak|sk|token)|"
    r"(?:.*[_-])?(?:api|access|secret)[_-]?key(?:[_-]?id)?|"
    r".*(?:token|password|secret|credential).*)", re.I
)
_SECRET_TEXT = re.compile(
    r"(?i)(\b(?:AGATE_AK|AGATE_SK|AGATE_TOKEN|API_KEY|ACCESS_KEY|SECRET_KEY|PASSWORD)"
    r"\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;\"'\]}]+)"
    r"|(\b(?:Authorization\s*:\s*)?Bearer\s+)[A-Za-z0-9._~+/-]+=*"
)
_SOURCE_VIOLATION = re.compile(
    r"(?:Blocked import|Forbidden attribute access): [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r"|Forbidden string literal matching: [A-Za-z_]\w*"
)


def source_error_result(value: dict[str, Any]) -> dict[str, Any]:
    """Only public candidate-source rules, never raw validation/evaluator text."""
    error = value.get("error")
    if (not isinstance(error, dict) or error.get("code") != "candidate_source_rejected"
            or error.get("job_submitted") is not False):
        raise ValueError("not a candidate source rejection")
    raw_violations = error.get("violations", [])
    if not isinstance(raw_violations, list):
        raise ValueError("source violations must be a list")
    violations = []
    for item in raw_violations[:MAX_AGENT_DIAGNOSTICS]:
        if (isinstance(item, str) and len(item) <= 256
                and _SOURCE_VIOLATION.fullmatch(item) and item not in violations):
            violations.append(item)
    return error_response(
        "Agate rejected candidate source before GPU submission.",
        code="candidate_source_rejected", error_class="code", reason="source_validation_failed",
        job_submitted=False, violations=violations,
        next_action=(
            "Remove or replace the listed forbidden imports, attribute accesses or string literals "
            "in kernel.py, then retry the same operation with the repaired source. "
            "Do not bypass source validation through Dev."
        ),
    )


def candidate_source_rejection(process: subprocess.CompletedProcess) -> dict[str, Any] | None:
    """Recognize a confirmed HTTP 400 candidate-source rejection, not an unknown outcome."""
    rejection = submission_rejection(process)
    if rejection is None or rejection.status != 400:
        return None
    text = (process.stderr or "") + "\n" + (process.stdout or "")
    # Direct HTTP errors preserve a JSON envelope, unlike the CLI's plain text.
    try:
        payload = json.loads(process.stdout or "")
        error = payload.get("error", payload.get("detail")) if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        if isinstance(message, str):
            text += "\n" + message
    except (ValueError, TypeError):
        pass
    if "source validation failed" not in text.lower():
        return None
    candidate_lines = re.findall(r"(?m)^\s*-\s*candidate:\s*([^\n]+)", text)
    if not candidate_lines:
        return None  # Reference/input validation can contain private information.
    violations = []
    for line in candidate_lines:
        match = re.match(r"(Blocked import|Forbidden attribute access):\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", line)
        if match:
            violations.append(f"{match[1]}: {match[2]}")
            continue
        match = re.match(r"Forbidden string literal matching ['\"]([A-Za-z_]\w*)['\"]:", line)
        if match:
            # Do not expose the matched literal, which may contain paths/data.
            violations.append(f"Forbidden string literal matching: {match[1]}")
    result = source_error_result({"error": {
        "code": "candidate_source_rejected", "job_submitted": False, "violations": violations,
    }})
    raw = gateway_error_from_process(process, "source_validation")
    if raw is not None:
        result["gateway_error"] = raw["error"]
    return result


def source_error_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith(SOURCE_ERROR_PREFIX):
            try:
                return source_error_result(json.loads(line[len(SOURCE_ERROR_PREFIX):]))
            except (ValueError, TypeError, AttributeError):
                continue
    return None


def bounded_error(value: Any) -> Any:
    """Retain the original JSON error, with an explicit bound for oversized errors."""
    # Redact credential fields before truncation can turn a JSON tree into a
    # text fragment and lose the field names needed for credential redaction.
    value = redact_diagnostics(value)
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(rendered.encode("utf-8")) <= MAX_GATEWAY_ERROR_BYTES:
        return value
    text, omitted = _bounded_utf8_text(rendered, MAX_GATEWAY_ERROR_BYTES, marker="error")
    return {"truncated": True, "bytes_omitted": omitted, "raw_error": text}


def gateway_error_result(value: dict[str, Any]) -> dict[str, Any]:
    """The original error, not the private request or the full Gateway envelope."""
    if "error" not in value:
        raise ValueError("Gateway error response omitted error")
    result = {key: value[key] for key in (
        "operation", "status", "job_id", "trace_id", "http_status", "batch", "outcome",
    ) if key in value and (value[key] is None or type(value[key]) in {str, int})}
    result["error"] = bounded_error(value["error"])
    return result


def gateway_error_from_process(process: subprocess.CompletedProcess, operation: str) -> dict[str, Any] | None:
    """Extract an actual job/HTTP error; a nonzero exit alone proves no submission outcome."""
    try:
        payload = json.loads(process.stdout or "")
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and (
        payload.get("error") is not None or (payload.get("job_id") and payload.get("status") != "succeeded")
    ):
        return gateway_error_result({"operation": operation, **payload, "error": payload.get("error")})
    rejection = submission_rejection(process)
    if rejection is None:
        return None
    error = payload.get("detail") if isinstance(payload, dict) and "detail" in payload else (
        payload if payload is not None else process.stderr or process.stdout or ""
    )
    return gateway_error_result({"operation": operation, "status": "rejected",
                                 "http_status": rejection.status, "error": error})


def gateway_errors_from_stdout(stdout: str) -> list[dict[str, Any]]:
    errors = []
    for line in stdout.splitlines():
        if line.startswith(GATEWAY_ERROR_PREFIX):
            try:
                value = json.loads(line[len(GATEWAY_ERROR_PREFIX):])
                if isinstance(value, dict):
                    errors.append(gateway_error_result(value))
            except (ValueError, TypeError):
                continue
    return errors


def credential_values(environment: dict[str, str]) -> tuple[str, ...]:
    return tuple(value for key, value in environment.items()
                 if _SECRET_KEY.fullmatch(key) and isinstance(value, str) and len(value) >= 4)


def emit_gateway_error(process: subprocess.CompletedProcess, operation: str) -> bool:
    result = gateway_error_from_process(process, operation)
    if result is not None:
        print(GATEWAY_ERROR_PREFIX + json.dumps(result, ensure_ascii=False))
    return result is not None


def gateway_http_error_result(status: int, detail: str, operation: str) -> dict[str, Any]:
    try:
        body = json.loads(detail)
    except ValueError:
        body = detail
    error = body.get("error", body.get("detail", body)) if isinstance(body, dict) else body
    return gateway_error_result({
        "operation": operation, "http_status": status, "error": error,
        "outcome": "failed" if operation == "env" else (
            "unknown" if status >= 500 and status != 503 else "rejected"
        ),
    })


def redact_diagnostics(value: Any, *, private_paths=(), private_values=()) -> Any:
    """Redact paths and credentials without replacing the original error classification/text."""
    if isinstance(value, dict):
        return {key: "<redacted>" if _SECRET_KEY.fullmatch(key) else
                redact_diagnostics(item, private_paths=private_paths, private_values=private_values)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_diagnostics(item, private_paths=private_paths, private_values=private_values) for item in value]
    if not isinstance(value, str):
        return value
    for secret in sorted(set(item for item in private_values if item), key=len, reverse=True):
        value = value.replace(secret, "<redacted>")
    for path in sorted(set(item for item in private_paths if item), key=len, reverse=True):
        value = value.replace(path, "<supervisor>")
    return _SECRET_TEXT.sub(lambda match: (match[1] or match[2]) + "<redacted>", value)

def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None

def bounded_text(value: object, limit: int = 1000) -> str:
    """Format an optional diagnostic value using a character-count limit."""
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"

def _profile_duration_us(kernel: dict[str, Any]) -> float | None:
    duration_us = _finite_number(kernel.get("duration_us"))
    if duration_us is not None and duration_us >= 0.0:
        return duration_us
    duration = _finite_number(kernel.get("duration"))
    unit = kernel.get("duration_unit")
    if duration is None or duration < 0.0 or not isinstance(unit, str):
        return None
    scale = {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1_000_000.0}.get(unit.casefold())
    return None if scale is None else duration * scale

def _agent_profile_metric(value: object) -> object | None:
    """Project one requested profiler counter without serving raw profiler text."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return bounded_text(value, 256)
    if not isinstance(value, dict):
        return None
    projected: dict[str, object] = {}
    for key, item in value.items():
        if key not in {"value", "unit"} or isinstance(item, bool):
            continue
        if isinstance(item, float) and not math.isfinite(item):
            continue
        if item is None or isinstance(item, (int, float)):
            projected[key] = item
        elif isinstance(item, str):
            projected[key] = bounded_text(item, 256)
    return projected or None

def _agent_profile_kernel(raw: dict[str, Any], *, total_duration_us: float) -> dict[str, Any]:
    """Create an explicit, bounded allowlist for one profiled Kernel."""
    projected: dict[str, Any] = {}
    name = raw.get("name", raw.get("kernel_name"))
    if isinstance(name, str) and name:
        projected["name"] = bounded_text(name, 512)
    duration_us = _profile_duration_us(raw)
    if duration_us is not None:
        projected["duration_us"] = duration_us
        if total_duration_us > 0.0:
            projected["duration_share_pct"] = duration_us * 100.0 / total_duration_us

    aliases = {
        "mem_sol_pct": "memory_sol_pct",
        "dram_pct": "dram_throughput_pct",
        "registers": "registers_per_thread",
        "smem_bytes": "shared_memory_bytes",
    }
    numeric_fields = (
        "compute_sol_pct",
        "memory_sol_pct",
        "registers_per_thread",
        "shared_memory_bytes",
        "occupancy_pct",
        "achieved_occupancy_pct",
        "dram_throughput_pct",
        "memory_throughput_pct",
        "sm_throughput_pct",
        "waves_per_sm",
    )
    for field in numeric_fields:
        value = _finite_number(raw.get(field))
        if value is None:
            source = next((source for source, target in aliases.items() if target == field), None)
            value = _finite_number(raw.get(source)) if source else None
        if value is not None:
            projected[field] = value

    # SOL is a roll-up of memory subsystems, not DRAM bandwidth. Preserve the
    # measured traffic alongside it so callers can distinguish cache activity
    # from device-memory traffic without exposing private workload parameters.
    traffic = raw.get("traffic")
    if isinstance(traffic, dict):
        projected_traffic = {}
        for key in (
            "dram_bytes", "dram_bytes_read", "dram_bytes_write", "l2_bytes",
            "duration_ns", "achieved_dram_gbps",
        ):
            value = _finite_number(traffic.get(key))
            if value is not None and value >= 0:
                projected_traffic[key] = traffic[key]
        if projected_traffic:
            projected["traffic"] = projected_traffic

    bound = raw.get("bound")
    if isinstance(bound, str) and bound:
        projected["bound"] = bounded_text(bound, 64)
    elif "compute_sol_pct" in projected and "memory_sol_pct" in projected:
        projected["bound"] = (
            "compute" if projected["compute_sol_pct"] > projected["memory_sol_pct"] else "memory"
        )

    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        projected_metrics: dict[str, object] = {}
        for key, value in list(metrics.items())[:MAX_AGENT_PROFILE_METRICS]:
            metric = _agent_profile_metric(value)
            if metric is not None:
                projected_metrics[bounded_text(key, 256)] = metric
        if projected_metrics:
            projected["metrics"] = projected_metrics
    return projected

def _agent_clock_lock(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    projected = {
        key: (bounded_text(item, 256) if isinstance(item, str) else item)
        for key, item in value.items()
        if key in {"requested", "applied", "locked", "supported", "status", "reason"}
        and isinstance(item, (bool, str, int, float))
    }
    return projected or None

def profile_result(result: dict[str, Any], *, generalized: bool = False) -> dict[str, Any]:
    """Bound Profile metrics without losing aggregates on repeated projection."""
    kernels_value = result.get("kernels")
    kernels = (
        [item for item in kernels_value if isinstance(item, dict)]
        if isinstance(kernels_value, list)
        else []
    )
    durations = [
        duration for kernel in kernels if (duration := _profile_duration_us(kernel)) is not None
    ]
    total_duration_us = sum(durations)
    kernel_count = len(kernels)
    recorded_count = result.get("kernel_count")
    recorded_omitted = result.get("kernels_omitted")
    truncated = (
        type(recorded_count) is int
        and type(recorded_omitted) is int
        and recorded_omitted > 0
        and recorded_count == len(kernels) + recorded_omitted
    )
    if truncated:
        # Gateway output may already have been projected for hidden Shapes.
        # Its retained Kernel list is not the population these aggregates describe.
        kernel_count = recorded_count
        recorded_duration = _finite_number(result.get("total_duration_us"))
        if recorded_duration is not None and recorded_duration >= total_duration_us:
            total_duration_us = recorded_duration
    projected_kernels = [
        _agent_profile_kernel(kernel, total_duration_us=total_duration_us)
        for kernel in kernels[:MAX_AGENT_PROFILE_KERNELS]
    ]
    projected_kernels = [kernel for kernel in projected_kernels if kernel]

    projected: dict[str, Any] = {}
    for key in ("status", "passed", "shape_id", "profile_level", "level"):
        value = result.get(key)
        if isinstance(value, (bool, str, int, float)) and not (
            isinstance(value, float) and not math.isfinite(value)
        ):
            projected[key] = bounded_text(value, 256) if isinstance(value, str) else value
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        projected["exit_code"] = exit_code
    clock_lock = _agent_clock_lock(result.get("clock_lock"))
    if clock_lock:
        if generalized:
            clock_lock.pop("reason", None)
        if clock_lock:
            projected["clock_lock"] = clock_lock
    summary = result.get("summary")
    if isinstance(summary, str) and summary and not generalized:
        projected["summary"] = bounded_text(summary, 2000)
    error = result.get("error")
    if error:
        projected["error"] = bounded_error(error)

    projected["kernel_count"] = kernel_count
    if total_duration_us > 0.0:
        projected["total_duration_us"] = total_duration_us
        dominant = max(
            kernels,
            key=lambda kernel: _profile_duration_us(kernel) or 0.0,
            default={},
        )
        dominant_name = dominant.get("name", dominant.get("kernel_name"))
        if truncated and isinstance(result.get("dominant_kernel"), str) and result["dominant_kernel"]:
            dominant_name = result["dominant_kernel"]
        if isinstance(dominant_name, str) and dominant_name:
            projected["dominant_kernel"] = bounded_text(dominant_name, 512)

    weighted_duration = 0.0
    weighted_sol = 0.0
    weighted_compute = 0.0
    weighted_memory = 0.0
    for kernel in projected_kernels:
        duration = _finite_number(kernel.get("duration_us"))
        compute = _finite_number(kernel.get("compute_sol_pct"))
        memory = _finite_number(kernel.get("memory_sol_pct"))
        if duration is None or compute is None or memory is None:
            continue
        weighted_duration += duration
        weighted_sol += max(compute, memory) * duration
        weighted_compute += compute * duration
        weighted_memory += memory * duration
    if weighted_duration > 0.0:
        projected["weighted_sol_pct"] = weighted_sol / weighted_duration
        projected["dominant_bound"] = "compute" if weighted_compute > weighted_memory else "memory"
    projected["kernels"] = projected_kernels
    if kernel_count > len(projected_kernels):
        projected["kernels_omitted"] = kernel_count - len(projected_kernels)
    return projected

def _agent_diagnostic_item(value: object) -> object | None:
    """Project one compiler diagnostic without exposing request or worker state."""
    if isinstance(value, str):
        return bounded_text(value, 2000)
    if not isinstance(value, dict):
        return None
    projected: dict[str, object] = {}
    text_fields = {
        "severity",
        "level",
        "stage",
        "message",
        "file",
        "kernel",
        "name",
        "kind",
    }
    numeric_fields = {
        "line",
        "column",
        "registers",
        "registers_per_thread",
        "spills",
        "spill_bytes",
        "shared_memory_bytes",
        "local_memory_bytes",
        "stack_frame_bytes",
    }
    for key in text_fields:
        item = value.get(key)
        if isinstance(item, str) and item:
            projected[key] = bounded_text(item, 2000 if key == "message" else 512)
    for key in numeric_fields:
        item = value.get(key)
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and (not isinstance(item, float) or math.isfinite(item))
        ):
            projected[key] = item
    return projected or None

def _agent_check_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return only actionable compilation and sanitizer facts to the Agent."""
    projected: dict[str, Any] = {}
    for key in (
        "status",
        "ok",
        "passed",
        "compile_ok",
        "launch_ok",
        "asm_available",
        "correctness_checked",
        "sanitize",
        "sanitizer_passed",
        "scope",
        "shape_id",
        "failure_stage",
    ):
        if key not in result:
            continue
        value = result.get(key)
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            projected[key] = bounded_text(value, 512) if isinstance(value, str) else value
    for key in (
        "registers",
        "registers_per_thread",
        "spills",
        "spill_bytes",
        "shared_memory_bytes",
        "local_memory_bytes",
        "stack_frame_bytes",
    ):
        value = result.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        ):
            projected[key] = value
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, list):
        projected_diagnostics = [
            item
            for raw in diagnostics[:MAX_AGENT_DIAGNOSTICS]
            if (item := _agent_diagnostic_item(raw)) is not None
        ]
        projected["diagnostics"] = projected_diagnostics
        if len(diagnostics) > MAX_AGENT_DIAGNOSTICS:
            projected["diagnostics_omitted"] = len(diagnostics) - MAX_AGENT_DIAGNOSTICS
    error = result.get("error")
    if error:
        projected["error"] = bounded_error(error)
    return projected

def _bounded_utf8_text(
    value: str,
    limit: int,
    *,
    marker: str = "assembly",
) -> tuple[str, int]:
    if limit < 0:
        raise ValueError("UTF-8 text limit must be non-negative")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value, 0
    marker_bytes = f"\n... <{marker} truncated by Supervisor Runtime> ...\n".encode()
    if limit < len(marker_bytes):
        # The marker must fit too; tiny limits retain only a valid UTF-8 prefix.
        text = encoded[:limit].decode("utf-8", errors="ignore")
        return text, len(encoded) - len(text.encode("utf-8"))
    budget = limit - len(marker_bytes)
    head_size = budget * 2 // 3
    tail_size = budget - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore") if tail_size else ""
    text = head + marker_bytes.decode() + tail
    return text, len(encoded) - len(text.encode("utf-8"))

def _agent_disassembly_result(
    result: dict[str, Any]
) -> dict[str, Any]:
    """Return bounded SASS/PTX text plus useful Kernel resource facts."""
    projected: dict[str, Any] = {}
    for key in (
        "status",
        "ok",
        "passed",
        "compile_ok",
        "launch_ok",
        "scope",
        "shape_id",
        "format",
        "arch",
        "failure_stage",
    ):
        if key not in result:
            continue
        value = result.get(key)
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            projected[key] = bounded_text(value, 512) if isinstance(value, str) else value
    if "format" not in projected and isinstance(result.get("fmt"), str):
        projected["format"] = bounded_text(result["fmt"], 64)

    kernels = result.get("kernels")
    if isinstance(kernels, list):
        public_kernels = [
            _agent_profile_kernel(kernel, total_duration_us=0.0)
            for kernel in kernels[:MAX_AGENT_PROFILE_KERNELS]
            if isinstance(kernel, dict)
        ]
        projected["kernels"] = [kernel for kernel in public_kernels if kernel]
        if len(kernels) > MAX_AGENT_PROFILE_KERNELS:
            projected["kernels_omitted"] = len(kernels) - MAX_AGENT_PROFILE_KERNELS

    assembly: str | None = None
    exports = result.get("exports")
    assembly_name: str | None = None
    if isinstance(exports, dict):
        preferred = str(result.get("format") or "") + ".txt"
        candidates = [preferred, "sass.txt", "ptx.txt"]
        for name in candidates:
            exported = exports.get(name)
            if isinstance(exported, dict) and isinstance(exported.get("text"), str):
                assembly = exported["text"]
                assembly_name = name
                break
            if isinstance(exported, str):
                assembly = exported
                assembly_name = name
                break
    recorded_assembly: dict[str, Any] | None = None
    if assembly is None and isinstance(result.get("assembly"), dict):
        recorded_assembly = result["assembly"]
        if isinstance(recorded_assembly.get("text"), str):
            assembly = recorded_assembly["text"]
            assembly_name = str(recorded_assembly.get("format") or "unknown")
    if assembly is None:
        for key in ("assembly", "asm", "sass", "ptx", "text"):
            value = result.get(key)
            if isinstance(value, str) and value:
                assembly = value
                assembly_name = key
                break
    if assembly is not None:
        text, omitted = _bounded_utf8_text(assembly, MAX_AGENT_DISASSEMBLY_BYTES)
        recorded_size = (
            recorded_assembly.get("size_bytes")
            if recorded_assembly is not None
            else None
        )
        size_bytes = (
            recorded_size
            if isinstance(recorded_size, int)
            and not isinstance(recorded_size, bool)
            and recorded_size >= len(text.encode("utf-8", errors="replace"))
            else len(assembly.encode("utf-8", errors="replace"))
        )
        recorded_omitted = (
            recorded_assembly.get("bytes_omitted")
            if recorded_assembly is not None
            else None
        )
        bytes_omitted = (
            recorded_omitted
            if isinstance(recorded_omitted, int)
            and not isinstance(recorded_omitted, bool)
            and recorded_omitted > 0
            else omitted
        )
        projected["assembly"] = {
            "format": str(result.get("format") or assembly_name or "unknown").removesuffix(".txt"),
            "text": text,
            "size_bytes": size_bytes,
            "truncated": bool(bytes_omitted)
            or bool(recorded_assembly and recorded_assembly.get("truncated") is True),
        }
        if bytes_omitted:
            projected["assembly"]["bytes_omitted"] = bytes_omitted
    error = result.get("error")
    if error:
        projected["error"] = bounded_error(error)
    return projected

def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def evaluation(result: dict) -> dict:
    keys = {"all_pass", "latency_us_geomean", "latency_us_arith_mean",
            "performance_score", "performance_objective", "max_abs_err", "max_rel_err",
            "shape_ids_are_opaque", "hidden_case_details", "mode", "n_shapes",
            "correctness_passed", "correctness_total", "speedup_vs_ref_geomean",
            "speedup_vs_ref_mean", "kernel_sha256", "utilization_pct", "sol_utilization_pct"}
    value = {key: item for key, item in result.items()
             if key in keys and (item is None or isinstance(item, (str, int, float, bool)))
             and (not isinstance(item, float) or math.isfinite(item))}
    # SOL-ExecBench names full-workload coverage ``passed``/``total``. Normalize
    # those public scalar counts to the same contract as typed Atrex-Bench
    # without exposing the evaluator's richer internal payload.
    for source, target in (("passed", "correctness_passed"), ("total", "correctness_total")):
        item = result.get(source)
        if type(item) is int and item >= 0:
            value[target] = item
    shapes = result.get("latency_us_by_shape")
    if isinstance(shapes, dict):
        value["latency_us_by_shape"] = {
            str(key)[:128]: item for key, item in list(shapes.items())[:MAX_AGENT_EVALUATION_SHAPES]
            if _finite_number(item) is not None
        }
    value["failures"] = [bounded_text(item) for item in (result.get("failures") or [])[:8]]
    value["actionable_diagnostics"] = [
        {key: bounded_text(item) for key, item in row.items() if key in {"stage", "shape_id", "message"}}
        for row in (result.get("actionable_diagnostics") or [])[:8] if isinstance(row, dict)
    ]
    if result.get("error") is not None:
        value["error"] = bounded_error(result["error"])
    return value


NUMERICAL_RESULT_PREFIX = "__ATREX_NUMERICAL_RESULT__="


def numerical_result(payload: dict) -> dict:
    """Public probe receipts, never raw inputs, exception text or workload kwargs."""
    public = {"schema_version": 1, "runs": [], "all_pass": payload.get("all_pass") is True}
    for row in payload.get("runs", [])[:3]:
        item = {key: row.get(key) for key in (
            "case_id", "passed", "exit_code", "expected_probes", "observed_probes",
            "failed_probes", "shape_count", "seeds", "world_size", "selection_digest")}
        item["input_error"] = "input generation or evaluator execution incomplete; private diagnostics withheld" if row.get("input_error") else ""
        result = row.get("result")
        item["result"] = None
        if isinstance(result, dict):
            item["result"] = {
                "all_pass": result.get("all_pass"),
                "nonfinite_outputs": [role for role in ("reference", "candidate", "unknown")
                                      if role in result.get("nonfinite_outputs", [])],
                "numerical_metrics": {key: value for key, value in (result.get("numerical_metrics") or {}).items()
                    if key in {"relative_l2", "max_row_relative_l2", "max_elementwise_abs_diff",
                               "max_elementwise_rel_diff", "max_abs_err", "max_rel_err"}
                    and (value is None or type(value) in {int, float} and math.isfinite(value))},
            }
        public["runs"].append(item)
    if payload.get("error"):
        public["error"] = "numerical evaluator failed; private diagnostics withheld"
    return public


def project_response(process: subprocess.CompletedProcess, *, generalized=False, wiki=False,
                     private_paths=(), private_values=(), public_dev=False, operation=None) -> dict:
    """Keep legacy sentinels parseable, bound diagnostics, omit transport envelopes."""
    stdout, stderr = process.stdout or "", process.stderr or ""
    numerical = any(line.startswith(NUMERICAL_RESULT_PREFIX) for line in stdout.splitlines())
    projected = []
    for line in stdout.splitlines():
        prefix = next((prefix for prefix in (
            "[test_kernel] RESULT_JSON=", "[sandbox] PROFILE_JSON=",
            "[sandbox] CHECK_JSON=", "[sandbox] DISASSEMBLE_JSON=",
            "[sandbox] ABBA_JSON=",
            "[sandbox] ENV_JSON=",
            NUMERICAL_RESULT_PREFIX,
            SOURCE_ERROR_PREFIX,
            GATEWAY_ERROR_PREFIX,
        ) if line.startswith(prefix)), None)
        if prefix:
            try:
                raw = json.loads(line[len(prefix):])
                if not isinstance(raw, dict):
                    raise ValueError("not an object")
                raw = redact_diagnostics(raw, private_paths=private_paths, private_values=private_values)
                formatter = {
                    "[test_kernel] RESULT_JSON=": evaluation,
                    "[sandbox] PROFILE_JSON=": lambda value: profile_result(value, generalized=generalized),
                    "[sandbox] CHECK_JSON=": _agent_check_result,
                    "[sandbox] DISASSEMBLE_JSON=": _agent_disassembly_result,
                    # compare() already emits abba()'s public metric projection.
                    "[sandbox] ABBA_JSON=": lambda value: dict(value, error=bounded_error(value["error"])) if value.get("error") else dict(value),
                    "[sandbox] ENV_JSON=": dict,
                    NUMERICAL_RESULT_PREFIX: numerical_result,
                    SOURCE_ERROR_PREFIX: source_error_result,
                    GATEWAY_ERROR_PREFIX: gateway_error_result,
                }[prefix]
                value = redact_diagnostics(formatter(raw), private_paths=private_paths, private_values=private_values)
                # Source-repair records can also carry the original rejection.
                if prefix == SOURCE_ERROR_PREFIX and "gateway_error" in raw:
                    value["gateway_error"] = redact_diagnostics(
                        bounded_error(raw["gateway_error"]), private_paths=private_paths, private_values=private_values,
                    )
                projected.append(prefix + json.dumps(value, ensure_ascii=False))
            except (ValueError, TypeError, RecursionError, OverflowError):
                projected.append(json.dumps({"error": "Gateway result could not be projected"}))
        else:
            projected.append(redact_diagnostics(line, private_paths=private_paths, private_values=private_values))
    stdout = "\n".join(projected) + ("\n" if projected else "")
    if generalized and (process.returncode or numerical) and not wiki and not public_dev and operation != "env":
        # A completed comparison can fail correctness while still providing
        # useful, public per-side results. Keep them without exposing raw logs.
        stdout = "\n".join(line for line in projected if line.startswith((
            "[test_kernel] RESULT_JSON=", "[sandbox] ABBA_JSON=",
            "[sandbox] PROFILE_JSON=", "[sandbox] CHECK_JSON=", "[sandbox] DISASSEMBLE_JSON=",
            NUMERICAL_RESULT_PREFIX, SOURCE_ERROR_PREFIX, GATEWAY_ERROR_PREFIX,
        )))
        errors = gateway_errors_from_stdout(stdout)
        if source_error_from_stdout(stdout) is not None:
            stderr = "Candidate source rejected; fix the violations in SOURCE_ERROR_JSON and retry.\n"
        elif errors:
            from supervisor.errors import UNKNOWN_OUTCOME
            stderr = UNKNOWN_OUTCOME + "\n" if any(error.get("outcome") == "unknown" for error in errors) else ""
        elif stdout and not numerical:
            # Operation-specific results already include failure/diagnostic
            # fields. Do not contradict them with an infrastructure-sounding error.
            stderr = ""
        else:
            stderr = "GPU request failed; hidden-case diagnostics withheld. Ask the operator to inspect the failure.\n"
        if numerical and process.returncode == 0:
            stderr = ""
    # Structured lines were redacted before JSON encoding. Re-running free-form
    # regexes over encoded JSON can eat delimiters and corrupt result markers.
    stderr = redact_diagnostics(stderr, private_paths=private_paths, private_values=private_values)
    result = {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr}
    for key, limit in (("stdout", 256 * 1024 if wiki else 384 * 1024), ("stderr", 16 * 1024)):
        encoded = result[key].encode()
        if len(encoded) > limit:
            if key == "stdout" and any(line.startswith("[") for line in projected):
                result[key] = json.dumps({"error": "Projected result exceeds context limit; narrow the request."}) + "\n"
                result["exit_code"] = process.returncode or 2
            else:
                result[key] = encoded[:limit].decode("utf-8", "ignore") + "\n<output truncated>\n"
            result["truncated"] = True
    return result


def _abba_side_metrics(
    rows: list[dict[str, Any]], side: str, repeats: int, shape_ids: list[str]
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("revision") == side]
    successful = [
        row
        for row in selected
        if row.get("exit_code") == 0
        and isinstance(row.get("result"), dict)
        and row["result"].get("all_pass") is True
    ]
    by_shape: dict[str, float] = {}
    for shape_id in shape_ids:
        values = [
            _positive_number(row["result"].get("latency_us_by_shape", {}).get(shape_id))
            for row in successful
            if isinstance(row["result"].get("latency_us_by_shape"), dict)
        ]
        numbers = [value for value in values if value is not None]
        if len(numbers) == repeats:
            by_shape[shape_id] = math.exp(sum(math.log(value) for value in numbers) / len(numbers))
    run_latencies = [
        _positive_number(row["result"].get("latency_us_geomean")) for row in successful
    ]
    valid_run_latencies = [value for value in run_latencies if value is not None]
    latency = (
        math.exp(sum(math.log(value) for value in valid_run_latencies) / len(valid_run_latencies))
        if len(valid_run_latencies) == repeats
        else None
    )
    scores = [_positive_number(row["result"].get("performance_score")) for row in successful]
    score = (
        sum(value for value in scores if value is not None) / repeats
        if len(scores) == repeats and all(value is not None for value in scores)
        else None
    )
    result = {
        "correct": (
            len(successful) == repeats and set(by_shape) == set(shape_ids)
            and latency is not None
        ),
        "latency_us_geomean": latency,
        "latency_us_by_shape": by_shape,
        "performance_score": score,
        "max_abs_err": max(
            (row["result"].get("max_abs_err") or 0 for row in successful), default=0,
        ),
        "max_rel_err": max(
            (row["result"].get("max_rel_err") or 0 for row in successful), default=0,
        ),
    }
    # Keep the evaluator's public failure diagnostics for the affected side;
    # never forward the private per-run stdout_tail/stderr_tail logs.
    for row in selected:
        if isinstance(row.get("result"), dict) and row["result"].get("all_pass") is not True:
            failure = evaluation(row["result"])
            for key in ("failures", "actionable_diagnostics"):
                if failure.get(key):
                    result.setdefault(key, []).extend(failure[key])
                    result[key] = result[key][:MAX_AGENT_DIAGNOSTICS]
            if "error" in failure:
                result["error"] = failure["error"]
    return result

def _agent_abba_public_result(
    payload: dict[str, Any],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
    repeats: int,
) -> dict[str, Any]:
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("ABBA result omitted its measurement rows")
    if [{"revision": row.get("revision"), "repeat": row.get("repeat")} for row in rows] != schedule:
        raise ValueError("ABBA result did not execute the exact requested schedule")
    baseline = _abba_side_metrics(rows, "incumbent", repeats, shape_ids)
    candidate = _abba_side_metrics(rows, "candidate", repeats, shape_ids)
    baseline_us = _positive_number(baseline.get("latency_us_geomean"))
    candidate_us = _positive_number(candidate.get("latency_us_geomean"))
    correct = (
        baseline["correct"] is True and candidate["correct"] is True and not payload.get("error")
    )
    speedup = baseline_us / candidate_us if correct and baseline_us and candidate_us else None
    return {
        "status": "succeeded" if correct and payload.get("error") is None else "failed",
        "operation": "same_allocation_abba",
        "correct": correct,
        "comparison": {"method": "abba", "repeats": repeats},
        "schedule": [
            {
                "side": "A" if step["revision"] == "incumbent" else "B",
                "repeat": step["repeat"],
            }
            for step in schedule
        ],
        "baseline": baseline,
        "candidate": candidate,
        "speedup": speedup,
        "improvement_pct": (speedup - 1.0) * 100.0 if speedup is not None else None,
        "measurements": [
            {
                "side": "A" if row.get("revision") == "incumbent" else "B",
                "repeat": row.get("repeat"),
                "correct": isinstance(row.get("result"), dict)
                and row["result"].get("all_pass") is True,
                "latency_us_geomean": (
                    row["result"].get("latency_us_geomean")
                    if isinstance(row.get("result"), dict)
                    else None
                ),
                "latency_us_by_shape": (
                    row["result"].get("latency_us_by_shape", {})
                    if isinstance(row.get("result"), dict)
                    else {}
                ),
            }
            for row in rows
        ],
        "shape_batch_count": payload.get("shape_batch_count", 1),
        "error": bounded_error(payload.get("error")) if payload.get("error") else None,
    }

def abba(payload, schedule, shape_ids, repeats):
    result = _agent_abba_public_result(payload, schedule, shape_ids, repeats)
    for key in ("measurements", "shape_batch_count", "schedule"):
        result.pop(key, None)
    return result
