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


def _probe_verdict(status, reason, *, expected=None, observed=None, failed=0):
    """Return a bounded receipt verdict suitable for public aggregation."""
    return {
        "status": status,
        "reason": reason,
        "expected_probes": expected,
        "observed_probes": observed,
        "failed_probes": failed,
    }


def _probe_status(batch, plan, suite, shapes):
    """Classify one probe receipt without exposing private evaluator contents."""
    if not isinstance(batch, dict):
        return _probe_verdict("needs_validation", "receipt_missing")
    rows = batch.get("runs")
    if batch.get("error"):
        return _probe_verdict("needs_validation", "batch_error")
    if rows is None or rows == []:
        return _probe_verdict("needs_validation", "receipt_missing")
    if not isinstance(rows, list):
        return _probe_verdict("needs_validation", "receipt_schema_mismatch")
    if len(rows) != 1:
        return _probe_verdict("needs_validation", "run_count_mismatch")
    row = rows[0]
    if not isinstance(row, dict):
        return _probe_verdict("needs_validation", "receipt_schema_mismatch")
    try:
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
    except (KeyError, TypeError, AttributeError):
        return _probe_verdict("needs_validation", "receipt_schema_mismatch")
    observed = row.get("observed_probes")
    failed = row.get("failed_probes", 0)
    counts = {
        "expected": expected,
        "observed": observed if type(observed) is int and observed >= 0 else None,
        "failed": failed if type(failed) is int and failed >= 0 else 0,
    }
    if (
        row.get("case_id") != plan["case_id"]
        or row.get("selection_digest") != plan["selection_digest"]
    ):
        return _probe_verdict(
            "needs_validation",
            "receipt_identity_mismatch",
            expected=counts["expected"],
            observed=counts["observed"],
            failed=counts["failed"],
        )
    if (
        row.get("shape_count") != len(plan["shape_ids"])
        or row.get("seeds") != plan["seeds"]
        or row.get("world_size") != suite["world_size"]
        or row.get("expected_probes") != expected
        or type(observed) is not int
        or not 0 <= observed <= expected
        or type(failed) is not int
        or not 0 <= failed <= observed
        or not isinstance(row.get("result"), dict)
    ):
        return _probe_verdict(
            "needs_validation",
            "receipt_schema_mismatch",
            expected=counts["expected"],
            observed=counts["observed"],
            failed=counts["failed"],
        )
    if row.get("input_error"):
        return _probe_verdict(
            "needs_validation", "input_error", expected=expected,
            observed=observed, failed=failed,
        )
    result = row["result"]
    nonfinite = result.get("nonfinite_outputs", [])
    if not isinstance(nonfinite, list) or not all(
        value in {"reference", "candidate", "unknown"} for value in nonfinite
    ):
        return _probe_verdict(
            "needs_validation", "receipt_schema_mismatch", expected=expected,
            observed=observed, failed=failed,
        )
    if "reference" in nonfinite:
        return _probe_verdict(
            "needs_validation", "reference_nonfinite", expected=expected,
            observed=observed, failed=failed,
        )
    if "unknown" in nonfinite:
        return _probe_verdict(
            "needs_validation", "nonfinite_role_unknown", expected=expected,
            observed=observed, failed=failed,
        )
    # A single fully observed workload/seed can disprove correctness. Passing
    # still requires every requested workload, seed and rank to finish.
    if (
        0 < failed <= observed
        and result.get("all_pass") is False
    ):
        return _probe_verdict(
            "needs_repair", "candidate_failure", expected=expected,
            observed=observed, failed=failed,
        )
    if observed != expected:
        return _probe_verdict(
            "needs_validation", "coverage_incomplete", expected=expected,
            observed=observed, failed=failed,
        )
    if (
        row.get("passed") is True
        and row.get("exit_code") == 0
        and result.get("all_pass") is True
    ):
        return _probe_verdict(
            "passed", "passed", expected=expected,
            observed=observed, failed=failed,
        )
    if result.get("all_pass") is False:
        return _probe_verdict(
            "needs_repair", "candidate_failure", expected=expected,
            observed=observed, failed=failed,
        )
    return _probe_verdict(
        "needs_validation", "receipt_schema_mismatch", expected=expected,
        observed=observed, failed=failed,
    )
