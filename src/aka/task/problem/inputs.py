"""Existing public-input projection and private-input checks; no Agent execution."""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import hashlib
import math
import re
from aka.contracts.content import PublicInputContent, WorkspaceFile
from .layout import (AGENT_PROBLEM_FILENAME, is_sol_op, has_agent_problem, validate_private_shapes, validate_agent_problem, validate_generated_agent_problem, agent_visible_operator_files)


class PerformanceBaselineError(ValueError):
    """A production task cannot produce a hardware-specific performance score."""

    def __init__(self, hardware: str, shape_id: str, reason: str, detail: str):
        self.hardware = hardware
        self.shape_id = shape_id
        self.reason = reason
        super().__init__(
            f"invalid performance metadata for hardware={hardware or 'unknown'}: "
            f"sid={shape_id}: {detail}"
        )


def _positive_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _hardware_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _matches_hardware(name: str, record: dict, hardware: str) -> bool:
    target = _hardware_key(hardware)
    if not target:
        return False
    labels = [name]
    labels.extend(
        str(record.get(field) or "")
        for field in ("hardware", "platform", "measured_on")
    )
    for label in labels:
        label_key = _hardware_key(label)
        if label_key and (
            target == label_key or target in label_key or label_key in target
        ):
            return True
    return False


def _direct_trace_performance_values(payload: object) -> list[float]:
    if isinstance(payload, dict):
        values: list[float] = []
        direct = _positive_finite(payload.get("performance_us"))
        if direct is not None and payload.get("direct_trace_shape") is True:
            values.append(direct)
        for value in payload.values():
            values.extend(_direct_trace_performance_values(value))
        return values
    if isinstance(payload, list):
        values: list[float] = []
        for value in payload:
            values.extend(_direct_trace_performance_values(value))
        return values
    return []


def _trace_hardware_matches(shape: dict, metadata: dict, hardware: str) -> bool:
    target = _hardware_key(hardware)
    if not target:
        return True
    labels = [metadata.get("perf_gpu"), metadata.get("verify_gpu")]
    target_gpus = shape.get("target_gpus")
    if isinstance(target_gpus, list):
        labels.extend(target_gpus)
    keys = [_hardware_key(label) for label in labels if label]
    return not keys or any(target in key or key in target for key in keys)


def validate_performance_baseline(
    shapes: object, metadata: object, hardware: str
) -> dict[str, float]:
    """Return one positive production baseline per requested shape."""
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes must contain a non-empty JSON object")
    metadata_shapes = metadata.get("shapes") if isinstance(metadata, dict) else None
    baselines: dict[str, float] = {}
    for raw_shape_id in shapes:
        shape_id = str(raw_shape_id)
        shape = (
            metadata_shapes.get(shape_id)
            if isinstance(metadata_shapes, dict)
            else None
        )
        production = (
            shape.get("production_performance") if isinstance(shape, dict) else None
        )
        if not isinstance(production, dict):
            trace_values = (
                _direct_trace_performance_values(
                    shape.get("trace_performance_baseline")
                )
                if isinstance(shape, dict)
                and isinstance(metadata, dict)
                and _trace_hardware_matches(shape, metadata, hardware)
                else []
            )
            if len(trace_values) == 1:
                baselines[shape_id] = trace_values[0]
                continue
            raise PerformanceBaselineError(
                hardware,
                shape_id,
                "missing",
                "missing production_performance",
            )
        direct = _positive_finite(production.get("performance_us"))
        if direct is not None:
            baselines[shape_id] = direct
            continue
        candidates = [
            (str(name), value, record)
            for name, record in production.items()
            if isinstance(record, dict)
            if (value := _positive_finite(record.get("performance_us"))) is not None
        ]
        matching = [
            (name, value)
            for name, value, record in candidates
            if _matches_hardware(name, record, hardware)
        ]
        if len(matching) == 1:
            baselines[shape_id] = matching[0][1]
            continue
        if not matching and len(candidates) == 1:
            baselines[shape_id] = candidates[0][1]
            continue
        if candidates:
            platforms = ",".join(sorted(name for name, _, _ in candidates))
            raise PerformanceBaselineError(
                hardware,
                shape_id,
                "ambiguous" if matching else "hardware_unmatched",
                "production_performance has no unique baseline for the requested "
                f"hardware; available platforms={platforms}",
            )
        raise PerformanceBaselineError(
            hardware,
            shape_id,
            "non_positive",
            "production_performance.performance_us must be a positive finite number",
        )
    return baselines


@dataclass
class BenchInputs:
    operator_dir: Path
    workspace: Path
    optimization_mode: str
    bench_root: str
    generated_digest: str = ""

    @cached_property
    def private_reference_dir(self) -> Path | None:
        """Return evaluator-only native inputs behind a generalized public problem."""
        op_dir = self.operator_dir
        use_generalized = (
            self.optimization_mode == "production"
            and not is_sol_op(op_dir)
            and (op_dir / "shapes.json").is_file()
        )
        if not self.bench_root or not use_generalized:
            return None
        shapes_path = op_dir / "shapes.json"
        validate_private_shapes(shapes_path)
        if has_agent_problem(op_dir):
            validate_agent_problem(
                op_dir / AGENT_PROBLEM_FILENAME,
                private_shapes_path=shapes_path,
            )
        return op_dir


    def assert_private(self) -> None:
        """Fail closed if exact evaluator artifacts appear in the agent workspace."""
        private_dir = self.private_reference_dir
        if private_dir is None:
            return
        public_problem = self.workspace / AGENT_PROBLEM_FILENAME
        if not public_problem.is_file():
            raise RuntimeError(
                "generalized Atrex-Bench workspace is missing agent_problem.json; "
                "start a fresh workspace"
            )
        try:
            provided_problem = private_dir / AGENT_PROBLEM_FILENAME
            if provided_problem.is_file():
                validate_agent_problem(
                    public_problem,
                    private_shapes_path=private_dir / "shapes.json",
                )
                if public_problem.read_bytes() != provided_problem.read_bytes():
                    raise ValueError(
                        "workspace agent_problem.json differs from the user-provided contract"
                    )
            else:
                validate_generated_agent_problem(
                    public_problem,
                    private_shapes_path=private_dir / "shapes.json",
                )
                if (
                    self.generated_digest
                    and hashlib.sha256(public_problem.read_bytes()).hexdigest()
                    != self.generated_digest
                ):
                    raise ValueError(
                        "workspace agent_problem.json was modified after automatic generation"
                    )
        except ValueError as exc:
            raise RuntimeError(
                f"generalized Atrex-Bench workspace has an invalid public problem: {exc}; "
                "start a fresh workspace"
            ) from exc
        leaked = [
            name
            for name in ("shapes.json", "metadata.json", "roofline.json", "valid.py")
            if (self.workspace / name).exists()
        ]
        if leaked:
            raise RuntimeError(
                "generalized Atrex-Bench workspace exposes evaluator-only files: "
                + ", ".join(leaked)
                + "; start a fresh workspace"
            )


    def prepare_public(self) -> PublicInputContent:
        """Describe the public contract before any production optimization session.

        A user-authored contract is copied verbatim. When production receives only detailed
        evaluator shapes, a dedicated clean AKA session derives the public contract in a temporary
        directory; the later baseline/optimization sessions never receive ``shapes.json``.
        """
        private_dir = self.private_reference_dir
        if private_dir is None:
            return PublicInputContent(True)
        destination = self.workspace / AGENT_PROBLEM_FILENAME
        shapes_path = private_dir / "shapes.json"
        provided = private_dir / AGENT_PROBLEM_FILENAME
        if provided.is_file():
            validate_agent_problem(provided, private_shapes_path=shapes_path)
            print(
                f"[orchestrator] generalized problem: using user-provided {provided}",
                flush=True,
            )
            return PublicInputContent(True, (WorkspaceFile(AGENT_PROBLEM_FILENAME, source=provided),))
        if destination.is_file():
            validate_generated_agent_problem(
                destination,
                private_shapes_path=shapes_path,
            )
            self.generated_digest = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            print(
                "[orchestrator] generalized problem: reusing workspace-generated "
                f"{destination}",
                flush=True,
            )
            return PublicInputContent(True)
        if self.optimization_mode != "production":
            raise RuntimeError(
                "a generalized non-production campaign requires a user-provided "
                f"{AGENT_PROBLEM_FILENAME}"
            )

        return PublicInputContent(False)

    def operator_files(self) -> tuple[WorkspaceFile, ...]:
        generalized = self.private_reference_dir is not None
        return tuple(WorkspaceFile(name, source=self.operator_dir / name)
                     for name in agent_visible_operator_files(self.operator_dir, generalized=generalized)
                     if (self.operator_dir / name).is_file())

    def authoring_assets(self) -> tuple[WorkspaceFile, ...]:
        private_dir = self.private_reference_dir
        return tuple(WorkspaceFile(name, source=private_dir / name)
                     for name in ("reference.py", "input.py", "shapes.json", "metadata.json")
                     if (private_dir / name).is_file())

    def validate_generated(self, generated: Path) -> None:
        validate_generated_agent_problem(generated, private_shapes_path=self.private_reference_dir / "shapes.json")

    def generated_file(self, generated: Path) -> WorkspaceFile:
        return WorkspaceFile(AGENT_PROBLEM_FILENAME, source=generated)

    def record_generated(self) -> None:
        self.generated_digest = hashlib.sha256((self.workspace / AGENT_PROBLEM_FILENAME).read_bytes()).hexdigest()
