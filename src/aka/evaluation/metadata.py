"""Evaluation-owned interpretation of the distributed benchmark contract."""
import json
from pathlib import Path

def distributed_evaluation_world_size(metadata: object) -> int:
    """GPU count shared by Agate submission and production source-policy checks."""
    if not isinstance(metadata, dict):
        return 1
    contract = metadata.get("benchmark_contract")
    # Legacy metadata may describe multi-GPU hardware without opting into a
    # distributed evaluator. Preserve its single-GPU submission and source gate.
    if not isinstance(contract, dict) or "distributed_evaluation" not in contract:
        return 1
    evaluation = contract["distributed_evaluation"]
    prefix = "benchmark_contract.distributed_evaluation"
    if not isinstance(evaluation, dict):
        raise ValueError(f"{prefix} must be an object")
    hardware = metadata.get("hardware", {})
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object")
    if "required_gpus" in hardware:
        required = hardware["required_gpus"]
        if type(required) is not int or required < 1:
            raise ValueError("hardware.required_gpus must be a positive integer")
    if evaluation.get("launcher") != "torchrun":
        raise ValueError(f"{prefix}.launcher must be 'torchrun'")
    if evaluation.get("backend") != "nccl":
        raise ValueError(f"{prefix}.backend must be 'nccl'")
    count = evaluation.get("world_size")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{prefix}.world_size must be an integer")
    if not 2 <= count <= 64:
        raise ValueError(f"{prefix}.world_size must be between 2 and 64")
    if "required_gpus" in hardware and hardware["required_gpus"] != count:
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
