"""Build exact measurement plans and interpret results without submitting jobs."""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib
from .public_result import PREFIXES
from aka.evaluation.aggregation import aggregate

def result_from_response(response: dict, operation: str) -> dict | None:
    prefix = PREFIXES.get(operation)
    for line in reversed(response["stdout"].splitlines()):
        if prefix and line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            return value if isinstance(value, dict) else None
    return None



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
        from aka.light.supervisor.gateway import measurement_inputs, runtime_requirements
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
        from aka.light.supervisor.gateway import public_dev_output
        from aka.light.supervisor.projection import credential_values, project_response
        return project_response(
            process, generalized=generalized, operation=plan.operation,
            public_dev=plan.operation == "dev" and public_dev_output(plan.request["options"]["command"]),
            private_values=credential_values(environment), private_paths=private_paths,
        )

    def result(self, response, operation):
        return result_from_response(response, operation)

    def aggregate(self, samples, plan):
        from aka.light.supervisor.gateway import metadata_speedup_mean
        value = aggregate(samples, plan.operation)
        if "metadata.json" in plan.inputs:
            sides = [value] if plan.operation == "evaluate" else [value["baseline"], value["candidate"]]
            for side in sides:
                latencies = side.get("latency_us_by_shape", {})
                score, failures = metadata_speedup_mean(json.loads(plan.inputs["metadata.json"]), list(latencies), latencies)
                if not failures and score is not None:
                    side.update(performance_score=score, speedup_vs_ref_mean=score)
        return value
