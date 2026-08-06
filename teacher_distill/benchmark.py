from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from orchestrator import optimize

from .bundle import ValidatedTeacherBundle
from .models import canonical_json


SOL_CONFIG = {
    "seed": 200,
    "warmup_runs": 10,
    "iterations": 50,
    "benchmark_reference": True,
}

SandboxRunner = Callable[[Path, list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MaterializedTeacherWorkspace:
    workspace: Path
    kind: str
    expected_shape_keys: tuple[str, ...]
    workload_hash: str
    evaluator_hash: str
    measurement_config_hash: str


@dataclass(frozen=True)
class TeacherBenchmarkResult:
    geomean_latency_us: float
    latency_us_by_shape: Mapping[str, float]
    workload_hash: str
    evaluator_hash: str
    measurement_config_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency_us_by_shape",
            MappingProxyType(dict(self.latency_us_by_shape)),
        )


def _hash_named_bytes(values: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(values):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _directory_hash_values(root: Path, prefix: str) -> list[tuple[str, bytes]]:
    values: list[tuple[str, bytes]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        values.append((f"{prefix}/{relative.as_posix()}", path.read_bytes()))
    return values


def _copy_files(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = source / name
        if not path.is_file():
            raise ValueError("operator ground truth is missing %s" % name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _sol_shape_keys(workload_path: Path) -> tuple[str, ...]:
    keys: list[str] = []
    for line_number, line in enumerate(workload_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid workload.jsonl line %d" % line_number) from exc
        key = value.get("uuid") if isinstance(value, dict) else None
        if not isinstance(key, str) or not key:
            raise ValueError("workload.jsonl line %d has no uuid" % line_number)
        keys.append(key)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("workload.jsonl must contain unique workload UUIDs")
    return tuple(keys)


def _native_shape_keys(shapes_path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(shapes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid shapes.json") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("shapes.json must contain a non-empty object")
    return tuple(sorted((str(key) for key in value), key=lambda key: (0, int(key)) if key.isdigit() else (1, key)))


def materialize_teacher_workspace(
    bundle: ValidatedTeacherBundle,
    op_dir: Path | str,
    destination: Path | str,
    *,
    framework: str,
    atrex_bench_root: Path | str | None = None,
    measurement_context: Mapping[str, object] | None = None,
) -> MaterializedTeacherWorkspace:
    """Create a private evaluator-faithful Teacher workspace without executing code."""
    if not isinstance(bundle, ValidatedTeacherBundle):
        raise TypeError("bundle must be validated before materialization")
    if bundle.provenance.target.framework.casefold() != framework.casefold():
        raise ValueError("Teacher framework does not match materialization framework")
    op = Path(op_dir).resolve()
    workspace = Path(destination).resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("private Teacher workspace is not empty")
    workspace.mkdir(parents=True, exist_ok=True)

    is_sol = all((op / name).is_file() for name in ("definition.json", "reference.py", "workload.jsonl"))
    is_native = (op / "shapes.json").is_file() and (op / "reference.py").is_file()
    if is_sol:
        kind = "sol"
        if bundle.entry_point != "kernel.py::run":
            raise ValueError("SOL Teacher entry_point must be kernel.py::run")
        ground_truth = ("definition.json", "reference.py", "workload.jsonl")
        _copy_files(op, workspace, ground_truth)
        shutil.copy2(optimize.REPO_ROOT / "reference" / "test_kernel.py", workspace / "test_kernel.py")
        (workspace / "config.json").write_text(
            json.dumps(SOL_CONFIG, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected_shape_keys = _sol_shape_keys(workspace / "workload.jsonl")
        workload_files = ground_truth
        evaluator_values = [
            ("test_kernel.py", (workspace / "test_kernel.py").read_bytes()),
        ]
    elif is_native:
        kind = "native"
        if bundle.entry_point != "kernel.py::Model":
            raise ValueError("native Teacher entry_point must be kernel.py::Model")
        required = ("reference.py", "input.py", "shapes.json", "metadata.json")
        optional = tuple(
            name for name in ("roofline.json", "valid.py") if (op / name).is_file()
        )
        ground_truth = (*required, *optional)
        _copy_files(op, workspace, ground_truth)
        shutil.copy2(optimize.ATREX_BENCH_HARNESS, workspace / "test_kernel.py")
        if atrex_bench_root is None:
            raise ValueError("native Teacher benchmark requires atrex_bench_root")
        atrex_root = Path(atrex_bench_root).resolve()
        evaluator = atrex_root / "scripts" / "run_eval.py"
        package = atrex_root / "src" / "atrex_bench"
        if not evaluator.is_file() or not package.is_dir():
            raise ValueError("invalid atrex_bench_root")
        (workspace / "atrex-bench").symlink_to(atrex_root)
        expected_shape_keys = _native_shape_keys(workspace / "shapes.json")
        workload_files = ground_truth
        evaluator_values = [
            ("test_kernel.py", (workspace / "test_kernel.py").read_bytes()),
            ("atrex-bench/scripts/run_eval.py", evaluator.read_bytes()),
            *_directory_hash_values(package, "atrex-bench/src/atrex_bench"),
        ]
    else:
        raise ValueError("operator is neither SOL nor native Atrex-Bench format")

    for relative in bundle.source_paths:
        source = bundle.root / relative
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(bundle.root / "solution.json", workspace / "solution.json")
    shutil.copy2(bundle.root / "provenance.json", workspace / "provenance.json")
    (workspace / "benchmark_runs").mkdir(exist_ok=True)

    workload_hash = _hash_named_bytes(
        [(name, (workspace / name).read_bytes()) for name in workload_files]
    )
    evaluator_hash = _hash_named_bytes(evaluator_values)
    measurement_contract = {
        "kind": kind,
        "commands": {
            "single-seed": ["python", "test_kernel.py", "--version", "vteacher", "--no-memory"],
            "multi-seed": [
                "python",
                "test_kernel.py",
                "--version",
                "vteacher",
                "--multi-seed",
                "5",
                "--no-memory",
            ],
            "benchmark": ["python", "test_kernel.py", "--version", "vteacher", "--no-memory"],
        },
        "config": SOL_CONFIG if kind == "sol" else {"canonical_atrex_bench": True},
        "runtime_context": dict(measurement_context or {}),
    }
    measurement_config_hash = hashlib.sha256(
        canonical_json(measurement_contract).encode("utf-8")
    ).hexdigest()
    return MaterializedTeacherWorkspace(
        workspace=workspace,
        kind=kind,
        expected_shape_keys=expected_shape_keys,
        workload_hash=workload_hash,
        evaluator_hash=evaluator_hash,
        measurement_config_hash=measurement_config_hash,
    )


def _default_sandbox_runner(
    materialized: MaterializedTeacherWorkspace,
    *,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
) -> SandboxRunner:
    if not hardware:
        raise ValueError("Teacher benchmark requires sandbox hardware")

    def run(workspace: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        return optimize._sandbox_command(
            workspace,
            hardware,
            profile,
            url,
            timeout,
            command,
            gateway_kind="run",
        )

    return run


def _validate_result(
    stage: str,
    process: subprocess.CompletedProcess[str],
    expected_shape_keys: tuple[str, ...],
) -> dict:
    if process.returncode != 0:
        raise RuntimeError("Teacher %s correctness command failed (exit=%d)" % (stage, process.returncode))
    try:
        result = optimize._test_result_from_stdout(process.stdout)
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Teacher %s produced no usable result" % stage) from exc
    if not result.get("all_pass"):
        raise RuntimeError("Teacher %s correctness validation failed" % stage)
    by_shape = result.get("latency_us_by_shape")
    if not isinstance(by_shape, dict) or set(by_shape) != set(expected_shape_keys):
        raise RuntimeError("Teacher %s shape coverage does not match the workload" % stage)
    for key, value in by_shape.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError("Teacher %s reported invalid latency for shape %s" % (stage, key))
    geomean = result.get("latency_us_geomean")
    if isinstance(geomean, bool) or not isinstance(geomean, (int, float)) or not math.isfinite(geomean) or geomean <= 0:
        raise RuntimeError("Teacher %s reported invalid geomean latency" % stage)
    return result


def benchmark_teacher(
    materialized: MaterializedTeacherWorkspace,
    *,
    framework: str,
    sandbox_runner: SandboxRunner | None = None,
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = 600,
) -> TeacherBenchmarkResult:
    """Validate and measure the Teacher before any optimization Agent runs."""
    violations = optimize.production_kernel_violations(materialized.workspace, framework)
    if violations:
        raise RuntimeError("Teacher production policy rejected the bundle: " + "; ".join(violations))
    runner = sandbox_runner or _default_sandbox_runner(
        materialized,
        hardware=sandbox_hardware,
        profile=sandbox_profile,
        url=sandbox_url,
        timeout=sandbox_timeout,
    )
    commands = (
        ("single-seed", ["python", "test_kernel.py", "--version", "vteacher", "--no-memory"]),
        (
            "multi-seed",
            [
                "python",
                "test_kernel.py",
                "--version",
                "vteacher",
                "--multi-seed",
                "5",
                "--no-memory",
            ],
        ),
        ("benchmark", ["python", "test_kernel.py", "--version", "vteacher", "--no-memory"]),
    )
    measured: dict[str, dict] = {}
    for stage, command in commands:
        process = runner(materialized.workspace, command)
        result = _validate_result(stage, process, materialized.expected_shape_keys)
        measured[stage] = result
        record = {
            "schema_version": 1,
            "stage": stage,
            "command": command,
            "result": result,
        }
        (materialized.workspace / "benchmark_runs" / (stage + ".json")).write_text(
            json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    benchmark = measured["benchmark"]
    return TeacherBenchmarkResult(
        geomean_latency_us=float(benchmark["latency_us_geomean"]),
        latency_us_by_shape={
            str(key): float(value)
            for key, value in benchmark["latency_us_by_shape"].items()
        },
        workload_hash=materialized.workload_hash,
        evaluator_hash=materialized.evaluator_hash,
        measurement_config_hash=materialized.measurement_config_hash,
    )
