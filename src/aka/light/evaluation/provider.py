"""Build exact measurement plans and interpret results without submitting jobs."""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib
import math
import statistics
from .public_result import PREFIXES, correctness_summary

def result_from_response(response: dict, operation: str) -> dict | None:
    prefix = PREFIXES.get(operation)
    for line in reversed(response["stdout"].splitlines()):
        if prefix and line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            return value if isinstance(value, dict) else None
    return None


def _median_side(sides: list[dict]) -> dict:
    maps = [side.get("latency_us_by_shape", {}) for side in sides]
    if not maps[0] or any(set(values) != set(maps[0]) for values in maps):
        raise ValueError("Repeated measurement Shape coverage differs; no aggregate is available")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for values in maps for value in values.values()):
        raise ValueError("Repeated measurement contains an invalid latency")
    result = dict(sides[-1])
    values = {key: statistics.median(row[key] for row in maps) for key in maps[0]}
    result["latency_us_by_shape"] = values
    result["latency_us_geomean"] = math.exp(statistics.mean(math.log(value) for value in values.values()))
    result["latency_us_arith_mean"] = statistics.mean(values.values())
    for key in ("max_abs_err", "max_rel_err"):
        errors = [side[key] for side in sides if isinstance(side.get(key), (int, float))]
        if errors:
            result[key] = max(errors)
    result["correctness"] = correctness_summary(sides, passed=all(
        side.get("correct", side.get("all_pass")) is True for side in sides))
    # Scores computed on an individual run must not be mislabeled as median facts.
    for key in ("performance_score", "speedup_vs_ref_geomean", "speedup_vs_ref_mean", "utilization_pct", "sol_utilization_pct"):
        result.pop(key, None)
    return result


def aggregate(results: list[dict], operation: str) -> dict:
    if len(results) == 1:
        return results[0]
    if operation == "evaluate":
        if any(row.get("all_pass") is not True for row in results):
            raise ValueError("Cannot aggregate incomplete or rejected evaluations")
        return _median_side(results)
    if any(row.get("correct") is not True for row in results):
        raise ValueError("Cannot aggregate incomplete or rejected comparisons")
    value = dict(results[-1])
    value["baseline"] = _median_side([row["baseline"] for row in results])
    value["candidate"] = _median_side([row["candidate"] for row in results])
    value["speedup"] = value["baseline"]["latency_us_geomean"] / value["candidate"]["latency_us_geomean"]
    value["improvement_pct"] = (value["speedup"] - 1.0) * 100
    value.pop("measurements", None)
    return value



@dataclass(frozen=True)
class EvaluationPlan:
    request: dict
    inputs: dict[str, bytes]
    repetitions: int
    runtime_pins: dict

    @property
    def operation(self) -> str:
        return self.request["operation"]


class GatewayEvaluation:
    def prepare(self, args, staged, environment, *, identity_key, mode, repetitions) -> EvaluationPlan:
        from aka.light.evaluation.gateway import measurement_inputs, runtime_requirements
        request, inputs = measurement_inputs(args, staged, environment, identity_key=identity_key)
        pins = {}
        if value := environment.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", "").strip():
            pins["version"] = value
        if path := environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "").strip():
            pins["requirements"] = runtime_requirements(path)
        if pins:
            request["options"]["runtime_pins_digest"] = hashlib.sha256(
                json.dumps(pins, sort_keys=True).encode()).hexdigest()
        request["supervisor_policy"] = {"mode": mode, "repetitions": repetitions}
        count = repetitions if request["operation"] in {"evaluate", "same_allocation_abba"} else 1
        if (args.evaluation_mode == "correctness_only"
                or request["options"].get("evaluation_policy", {}).get("mode") == "correctness_only"):
            count = 1
        return EvaluationPlan(request, inputs, count, pins)

    def project(self, process, plan, *, generalized, environment, private_paths):
        from aka.light.evaluation.gateway import public_dev_output
        from aka.light.evaluation.projection import credential_values, project_response
        return project_response(
            process, generalized=generalized, operation=plan.operation,
            public_dev=plan.operation == "dev" and public_dev_output(plan.request["options"]["command"]),
            private_values=credential_values(environment), private_paths=private_paths,
        )

    def result(self, response, operation):
        return result_from_response(response, operation)

    def aggregate(self, samples, plan):
        from aka.light.evaluation.gateway import metadata_speedup_mean
        value = aggregate(samples, plan.operation)
        if "metadata.json" in plan.inputs:
            sides = [value] if plan.operation == "evaluate" else [value["baseline"], value["candidate"]]
            for side in sides:
                latencies = side.get("latency_us_by_shape", {})
                score, failures = metadata_speedup_mean(json.loads(plan.inputs["metadata.json"]), list(latencies), latencies)
                if not failures and score is not None:
                    side.update(performance_score=score, speedup_vs_ref_mean=score)
        return value
