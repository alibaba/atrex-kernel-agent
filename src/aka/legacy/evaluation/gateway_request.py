"""Build Legacy measurement requests from explicit, read-only CLI inputs."""
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from .metadata import distributed_evaluation_world_size as _distributed_evaluation_world_size
from aka.task.problem.inputs import validate_performance_baseline
FP4_MAX_REL_L2 = 0.2

def _option_value(parts: list[str], name: str, default: Any = None) -> Any:
    """Read a simple ``--flag value``/``--flag=value`` option from command argv."""
    command = parts
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1] if index + 1 < len(command) else default
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default

def _option_values(parts: list[str], name: str) -> list[str]:
    """Read every repeated ``--flag value``/``--flag=value`` option."""
    command = parts
    values: list[str] = []
    for index, token in enumerate(command):
        if token == name:
            if index + 1 >= len(command):
                raise ValueError(f"{name} requires a value")
            values.append(command[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    return values

def _json_object(path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ValueError(f"required typed-gateway input is missing: {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value

def _candidate_languages(
    solution: dict[str, Any], *, candidate_sources: Iterable[str] = ()
) -> list[str]:
    """Return the candidate languages declared by current or legacy manifests."""
    spec = solution.get("spec")
    languages = spec.get("languages") if isinstance(spec, dict) else None
    if not isinstance(languages, list):
        languages = solution.get("languages")
    if isinstance(languages, list):
        return [str(value) for value in languages]
    # Preserve the pre-refactor fallback for legacy manifests that omitted the
    # language declaration while importing the internal pinned framework.
    if any("flydsl" in source.casefold() for source in candidate_sources):
        return ["flydsl"]
    return []

def _is_fp4_dtype(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "").replace("_", "")
    return "fp4" in normalized or "float4" in normalized

def _metadata_has_fp4_dtype(metadata: object) -> bool:
    if isinstance(metadata, dict):
        if any(
            _is_fp4_dtype(metadata.get(field))
            for field in ("dtype", "dtype_compute")
        ):
            return True
        return any(_metadata_has_fp4_dtype(value) for value in metadata.values())
    if isinstance(metadata, list):
        return any(_metadata_has_fp4_dtype(value) for value in metadata)
    return False

def _fp4_correctness_max_rel_l2(
    metadata: dict[str, Any] | None,
    operator: object = None,
) -> float | None:
    return (
        FP4_MAX_REL_L2
        if _metadata_has_fp4_dtype(metadata) or _is_fp4_dtype(operator)
        else None
    )

def _parse_env_items(items: Iterable[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for item in items:
        if "=" not in item or item.startswith("="):
            raise ValueError(f"invalid --env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid --env key {key!r}")
        env_vars[key] = value
    return env_vars


def prepare_request(
    workspace: Path,
    hardware: str,
    timeout: int,
    env_items: list[str],
    command: list[str],
    kind: str,
    *,
    reference_dir: Path,
    requirements: Iterable[str] = (),
    profiler: str | None = None,
    profile_level: str = "sol",
    counters: Iterable[str] = (),
    kernel_regex: str | None = None,
    top_kernels: int | None = None,
) -> dict[str, Any]:
    """Build the public run/profile request without importing the agate package."""
    shapes = _json_object(
        reference_dir / "shapes.json", required=True
    )
    assert shapes is not None
    requested_shape_ids = _option_values(command, "--shape-id")
    if requested_shape_ids:
        unknown_shape_ids = [
            shape_id for shape_id in requested_shape_ids if shape_id not in shapes
        ]
        if unknown_shape_ids:
            raise ValueError(
                "unknown --shape-id values: " + ", ".join(unknown_shape_ids)
            )
        # Preserve command order while dropping accidental duplicates. The typed
        # gateway must honor the adapter's targeted-smoke contract instead of
        # silently expanding a one-shape request back to the complete workload.
        requested_shape_ids = list(dict.fromkeys(requested_shape_ids))
        shapes = {shape_id: shapes[shape_id] for shape_id in requested_shape_ids}
    try:
        multi_seed = int(_option_value(command, "--multi-seed", 0))
        bench_iters = int(_option_value(command, "--timed-runs", 20))
        atol = float(_option_value(command, "--atol", 1e-2))
        rtol = float(_option_value(command, "--rtol", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluator command option: {exc}") from exc
    if multi_seed < 0 or bench_iters < 1:
        raise ValueError(
            "--multi-seed must be non-negative and --timed-runs must be positive"
        )

    solution = _json_object(workspace / "solution.json") or {}
    candidate = (workspace / "kernel.py").read_text(encoding="utf-8")
    languages = _candidate_languages(solution, candidate_sources=(candidate,))
    reference: dict[str, Any] = {
        "operator": workspace.name,
        "reference_py": (workspace / "reference.py").read_text(encoding="utf-8"),
        "input_py": (workspace / "input.py").read_text(encoding="utf-8"),
        "shapes": shapes,
    }
    for filename, field in (
        ("metadata.json", "metadata"),
        ("roofline.json", "roofline"),
    ):
        value = _json_object(reference_dir / filename)
        if value is not None:
            if requested_shape_ids and isinstance(value.get("shapes"), dict):
                value = dict(value)
                value["shapes"] = {
                    shape_id: value["shapes"][shape_id]
                    for shape_id in requested_shape_ids
                    if shape_id in value["shapes"]
                }
                if filename == "metadata.json" and "num_shapes" in value:
                    value["num_shapes"] = len(requested_shape_ids)
            reference[field] = value

    spec: dict[str, Any] = {
        "languages": [str(value) for value in languages],
        "target_hardware": [hardware],
    }
    num_gpus = _distributed_evaluation_world_size(reference.get("metadata"))
    if num_gpus > 1:
        spec["num_gpus"] = num_gpus
    request: dict[str, Any] = {
        "name": f"{workspace.name}_{kind}",
        "spec": spec,
        "candidate": candidate,
        "reference": reference,
        "options": {
            "num_correctness_cases": 1 + multi_seed,
            "bench_iters": bench_iters,
            "atol": atol,
            "rtol": rtol,
            "timeout_s": timeout,
        },
        "env_vars": _parse_env_items(env_items),
    }
    correctness_max_rel_l2 = _fp4_correctness_max_rel_l2(
        reference.get("metadata")
        if isinstance(reference.get("metadata"), dict)
        else None,
        reference.get("operator"),
    )
    runtime_version = os.environ.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", "").strip()
    if runtime_version:
        request["atrex_bench_version"] = runtime_version
    if requirements:
        request["requirements"] = list(requirements)
    if kind == "run" and correctness_max_rel_l2 is not None:
        request["options"]["correctness_max_rel_l2"] = correctness_max_rel_l2
    if kind == "run":
        # Ordinary measurements need correctness and latency. Kernel attribution
        # is a separate profiler pass and belongs to the explicit Profile route.
        request["runner_overrides"] = {"skip_kernel_attribution": True}
        version = str(_option_value(command, "--version", "v0"))
        request["mode"] = (
            "correctness_only"
            if multi_seed > 0 and version not in {"v0", "v1"}
            else "full"
        )
        if request["mode"] != "correctness_only":
            validate_performance_baseline(
                shapes, reference.get("metadata"), hardware
            )
    else:
        if profiler:
            request["profiler"] = profiler
        if profile_level:
            request["level"] = profile_level
        if counters:
            request["counters"] = list(counters)
        if kernel_regex:
            request["kernel_regex"] = kernel_regex
        if top_kernels is not None:
            request["top_kernels"] = top_kernels
    return request
