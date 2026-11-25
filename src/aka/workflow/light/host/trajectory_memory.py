"""Project durable hypotheses beside the measured Campaign frontier."""

import json

from aka.evaluation.tool_results import cacheable_evaluation
from aka.workflow.light.host.workspace import read_input

HYPOTHESES_PATH = "scratch/trajectory-hypotheses.json"
_MAX_BYTES = 32 * 1024


def _tested_trial(trial, store, visible_ids):
    if not isinstance(trial, dict):
        return None
    record_id = trial.get("gateway_record_id")
    if not isinstance(record_id, str) or record_id not in visible_ids:
        return None
    record = store.read(record_id)
    request, response = record["request"], record["response"]
    candidate = record["kernels"].get("candidate", {})
    commit = candidate.get("kernel_id")
    if not commit or commit != trial.get("kernel_commit_id"):
        return None
    if request["operation"] not in {"evaluate", "same_allocation_abba"}:
        return None
    if not cacheable_evaluation(response, request):
        return None
    return commit, record_id


def render_trajectory_memory(workspace, store, visible_ids) -> bytes:
    lines = [
        "\n## Trajectory memory",
        "",
        "Measured outcomes apply to the exact kernel variant, shapes and settings "
        "in each Gateway Record under records/.",
        "",
        f"Keep pending hypotheses in {HYPOTHESES_PATH} as a JSON array (up to 32 rows). "
        "Each row has id, hypothesis and test. Optional basis lists related Gateway "
        "Record IDs. After evaluating the exact variant, tested can contain "
        "kernel_commit_id and gateway_record_id.",
        "",
    ]
    try:
        rows = json.loads(read_input(workspace, HYPOTHESES_PATH, limit=_MAX_BYTES))
    except FileNotFoundError:
        return "\n".join(lines).encode()
    except (OSError, ValueError):
        rows = None
    if not isinstance(rows, list) or len(rows) > 32:
        lines.append(f"- Review the JSON array at {HYPOTHESES_PATH}.")
        return ("\n".join(lines) + "\n").encode()

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = [row.get(field) for field in ("id", "hypothesis", "test")]
        if not all(isinstance(value, str) and value.strip() and len(value) <= 600 for value in values):
            continue
        identity, hypothesis, test = (" ".join(value.split()) for value in values)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            trial = _tested_trial(row.get("tested"), store, visible_ids)
        except (OSError, ValueError, KeyError):
            trial = None
        if trial is None:
            lines.append(f"- Pending {identity}. {hypothesis} Test: {test}")
        else:
            commit, record_id = trial
            lines.append(
                f"- Tested variant {commit} for {identity}. "
                f"See records/{record_id}.json for the measured result and scope."
            )
        basis = row.get("basis")
        if isinstance(basis, list):
            referenced = set(item for item in basis if isinstance(item, str)) & visible_ids
            if referenced:
                lines.append("  Related records: " + ", ".join(sorted(referenced)[:8]))
    return ("\n".join(lines) + "\n").encode()
