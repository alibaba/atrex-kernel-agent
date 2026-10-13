"""Queue budget shared by the sandbox transport and its supervising process."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

DEFAULT_QUEUE_WAIT_GRACE = 14_400


def distributed_evaluation_world_size(metadata: object) -> int:
    """GPU count shared by Agate submission and production source-policy checks."""
    if not isinstance(metadata, dict):
        return 1
    required = metadata.get("hardware", {}).get("required_gpus", 1)
    if type(required) is not int or required < 1:
        raise ValueError("hardware.required_gpus must be a positive integer")
    contract = metadata.get("benchmark_contract")
    if not isinstance(contract, dict):
        if required > 1:
            raise ValueError("required_gpus > 1 requires a distributed_evaluation contract")
        return 1
    evaluation = contract.get("distributed_evaluation")
    if evaluation is None:
        if required > 1:
            raise ValueError("required_gpus > 1 requires a distributed_evaluation contract")
        return 1
    prefix = "benchmark_contract.distributed_evaluation"
    if not isinstance(evaluation, dict):
        raise ValueError(f"{prefix} must be an object")
    if evaluation.get("launcher") != "torchrun":
        raise ValueError(f"{prefix}.launcher must be 'torchrun'")
    if evaluation.get("backend") != "nccl":
        raise ValueError(f"{prefix}.backend must be 'nccl'")
    count = evaluation.get("world_size")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{prefix}.world_size must be an integer")
    if not 2 <= count <= 64:
        raise ValueError(f"{prefix}.world_size must be between 2 and 64")
    if required != 1 and required != count:
        raise ValueError("hardware.required_gpus differs from distributed_evaluation.world_size")
    return count


def evaluator_num_gpus(reference_dir: Path) -> int:
    """Read evaluator-owned metadata, never the candidate's dependency manifest."""
    path = reference_dir / "metadata.json"
    if not path.is_file():
        return 1
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON in metadata.json") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    return distributed_evaluation_world_size(metadata)


def queue_wait_grace(environment: Mapping[str, str]) -> int:
    name = "ATREX_SANDBOX_QUEUE_WAIT_GRACE"
    try:
        value = int(environment.get(name, str(DEFAULT_QUEUE_WAIT_GRACE)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
