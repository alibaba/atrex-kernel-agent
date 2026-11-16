"""Bind portable SOL evaluation to the selected Task and complete workload."""
import hashlib
import json
import math

from aka.contracts.execution import ExecutionFile
from .private_inputs import selected_content


def trusted_inputs(evaluation_input):
    content = selected_content(evaluation_input)
    harness = evaluation_input.harness
    if harness is None or harness.path != "test_kernel.py":
        raise ValueError("SOL evaluation requires the selected test_kernel.py harness")
    data = harness.source.read_bytes() if harness.source is not None else harness.text.encode()
    content[harness.path] = ExecutionFile(harness.path, data)
    if not {"definition.json", "reference.py", "workload.jsonl"} <= content.keys():
        raise ValueError("SOL evaluation requires complete selected Task inputs")
    return content


def input_identity(content):
    return hashlib.sha256(json.dumps({name: hashlib.sha256(item.content).hexdigest()
        for name, item in content.items()}, sort_keys=True).encode()).hexdigest()


def workload_ids(content):
    rows = [json.loads(line) for line in content["workload.jsonl"].content.splitlines() if line.strip()]
    ids = tuple(row.get("uuid") if isinstance(row, dict) else None for row in rows)
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("SOL workload must contain distinct nonempty workload UUIDs")
    return ids


def single_source_solution(content):
    """Only claim the kernel.py source identity when SOL executes that source."""
    if "solution.json" not in content:
        raise ValueError("SOL evaluation requires solution.json")
    solution = json.loads(content["solution.json"].content)
    if not isinstance(solution, dict):
        raise ValueError("SOL solution.json must be an object")
    sources = solution.get("sources")
    spec = solution.get("spec")
    return (sources == [{"path": "kernel.py"}] and isinstance(spec, dict)
        and isinstance(spec.get("entry_point"), str) and spec["entry_point"].startswith("kernel.py::"))


def matches_task(identity, evaluation_input):
    if evaluation_input is None or evaluation_input.adapter != "legacy-sol-v1":
        return False
    return (identity.method == "legacy-sol/eval"
        and identity.parameters.get("sol_input_binding") == "closed-task/1"
        and identity.parameters.get("sol_input_digest") == input_identity(trusted_inputs(evaluation_input))
        and tuple(identity.parameters.get("command", ())) == ("python3", "test_kernel.py")
        and identity.parameters.get("workload_scope") == "full-task")


def validate_result(payload, expected_ids):
    """Reject partial/self-inconsistent receipts before they become verdicts."""
    total, passed = payload.get("total"), payload.get("passed")
    if (type(total) is not int or total != len(expected_ids) or type(passed) is not int
            or not 0 <= passed <= total or type(payload.get("all_pass")) is not bool
            or payload.get("error")):
        raise ValueError("SOL receipt does not cover the selected workload")
    if payload["all_pass"]:
        latencies = payload.get("latency_us_by_shape")
        if (passed != total or payload.get("failures") or not isinstance(latencies, dict)
                or set(latencies) != set(expected_ids)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (*latencies.values(), payload.get("latency_us_geomean"), payload.get("performance_score")))):
            raise ValueError("SOL passing receipt has incomplete correctness or performance evidence")
    elif passed == total and not payload.get("failures"):
        raise ValueError("SOL failing receipt has no measured failure")
