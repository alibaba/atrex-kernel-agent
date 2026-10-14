"""Original supplemental numerical plan and receipt interpretation."""

import json
from .drivers.remote_numerical import validate_suite


def _validate_review(value, digest):
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("evidence_digest") != digest
    ):
        raise ValueError("supplemental review is bound to different evidence")
    if value.get("action") not in {"complete", "probe"}:
        raise ValueError(
            "numerical reviewers must propose probes, not reject candidates"
        )
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        raise ValueError("supplemental review requires an evidence-based summary")
    suite = value.get("suite")
    if value["action"] == "probe":
        validate_suite(suite)
        for case in suite["cases"]:
            if case.get("shape_ids") is not None:
                raise ValueError(
                    "reviewers select public input constraints, not private shape IDs"
                )
            evidence = case.get("evidence", [])
            if not any(str(e).startswith("candidate/") for e in evidence) or not any(
                str(e).startswith("trusted/") for e in evidence
            ):
                raise ValueError(
                    "each probe needs candidate and input-contract evidence"
                )
    elif suite is not None:
        raise ValueError("complete reviews must not leave unexecuted probes")
    return value


def _probe_status(batch, plan, suite, shapes):
    """A complete receipt is required before labeling a result a counterexample."""
    rows = batch.get("runs", [])
    if batch.get("error") or len(rows) != 1:
        return "needs_validation"
    row = rows[0]
    expected = (
        len(
            {
                json.dumps(shapes[sid].get("input_kwargs") or {}, sort_keys=True)
                for sid in plan["shape_ids"]
            }
        )
        * len(plan["seeds"])
        * suite["world_size"]
    )
    if (
        row.get("case_id") != plan["case_id"]
        or row.get("selection_digest") != plan["selection_digest"]
        or row.get("shape_count") != len(plan["shape_ids"])
        or row.get("seeds") != plan["seeds"]
        or row.get("world_size") != suite["world_size"]
        or row.get("expected_probes") != expected
        or type(row.get("observed_probes")) is not int
        or not 0 <= row["observed_probes"] <= expected
        or not isinstance(row.get("result"), dict)
    ):
        return "needs_validation"
    result = row["result"]
    if row.get("input_error") or {"reference", "unknown"}.intersection(
        result.get("nonfinite_outputs", [])
    ):
        return "needs_validation"
    # A single fully observed workload/seed can disprove correctness. Passing
    # still requires every requested workload, seed and rank to finish.
    if (
        0 < row.get("failed_probes", 0) <= row["observed_probes"]
        and result.get("all_pass") is False
    ):
        return "needs_repair"
    if row["observed_probes"] != expected:
        return "needs_validation"
    if (
        row.get("passed") is True
        and row.get("exit_code") == 0
        and result.get("all_pass") is True
    ):
        return "passed"
    if result.get("all_pass") is False:
        return "needs_repair"
    return "needs_validation"
