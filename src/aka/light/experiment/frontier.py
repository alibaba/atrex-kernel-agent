"""Observed per-Shape minima from the Campaign's standard full Evaluations."""
from __future__ import annotations

import logging
import math

from aka.light.evaluation.provider import result_from_response
from aka.light.experiment.store import MeasurementStore, RECORD_ID, read_record_header


def kernel_pareto_frontier(store: MeasurementStore) -> dict:
    """Read existing facts only; a per-Shape envelope is not a measured composite Kernel.

    Use the same ordinary-full-Evaluate eligibility as nomination, never Profile,
    ABBA, custom-input or subset timings. The store is Campaign-scoped and shared
    across Episodes. Unlike Runtime's Registry, Light does not distinguish Agent
    and controller-origin full Evaluations; both are eligible saved observations.
    """
    winners: dict[str, tuple[float, str, str]] = {}
    evaluation_count = 0
    for path in (store.root / "records").glob("gateway-*/record.json"):
        record_id = path.parent.name
        try:
            if not RECORD_ID.fullmatch(record_id):
                continue
            # Do not load response bodies or source artifacts for other operations.
            header = read_record_header(path)
            if header["operation"] != "evaluate":
                continue
            record = store.read(record_id)
            if (record["operation"] != "evaluate" or record["request"]["operation"] != "evaluate"
                    or record.get("cacheable") is not True):
                continue
            options = record["request"]["options"]
            command = options.get("command", [])
            if (
                (options.get("evaluation_mode") or "full") != "full"
                or options.get("evaluation_policy", {}).get("mode", "full") != "full"
                or options.get("evaluation_input_path")
                or options.get("evaluation_shapes_path")
                or options.get("baseline_path")
                or command[:2] != ["python3", "test_kernel.py"]
                or any(arg == "--shape-id" or arg.startswith("--shape-id=") for arg in command)
            ):
                continue
            response = record["response"]
            if response.get("exit_code") != 0 or response.get("truncated"):
                continue
            result = result_from_response(response, "evaluate")
            if (not result or result.get("all_pass") is not True or result.get("error")
                    or result.get("mode", "full") != "full"):
                continue
            values = result.get("latency_us_by_shape")
            if not isinstance(values, dict):
                continue
            latencies = {
                shape: float(value) for shape, value in values.items()
                if isinstance(shape, str) and type(value) in {int, float}
                and math.isfinite(value) and value > 0
            }
            if not latencies:
                continue
            kernel_id = record["kernels"]["candidate"]["kernel_id"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError):
            # An interrupted/corrupt historical record must not break every query;
            # details stay in the Supervisor log, not in the Agent response.
            logging.getLogger(__name__).warning("Skipping invalid frontier evidence %s", record_id)
            continue
        evaluation_count += 1
        for shape, latency in latencies.items():
            candidate = (latency, kernel_id, record_id)
            if shape not in winners or candidate < winners[shape]:
                winners[shape] = candidate
    return {
        "best_by_shape": {
            shape: {"kernel_id": kernel, "gateway_record_id": record, "latency_us": latency}
            for shape, (latency, kernel, record) in sorted(winners.items())
        },
        "shape_count": len(winners),
        "winner_kernel_ids": sorted({kernel for _, kernel, _ in winners.values()}),
        "evaluation_count": evaluation_count,
    }
