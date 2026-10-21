"""Original supplemental numerical plan and receipt interpretation."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from aka.contracts.evaluation import NumericalProbeResult
from .drivers.remote_numerical import PREFIX, validate_suite


def validate_numerical_review(value, digest):
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
        return _probe_verdict("needs_validation", "receipt_schema_mismatch")
    if type(batch.get("schema_version")) is not int or batch["schema_version"] != 1:
        return _probe_verdict("needs_validation", "receipt_schema_mismatch")
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
        type(row.get("shape_count")) is not int
        or row.get("shape_count") != len(plan["shape_ids"])
        or not isinstance(row.get("seeds"), list)
        or any(type(seed) is not int for seed in row["seeds"])
        or row["seeds"] != plan["seeds"]
        or type(row.get("world_size")) is not int
        or row.get("world_size") != suite["world_size"]
        or type(row.get("expected_probes")) is not int
        or row.get("expected_probes") != expected
        or type(observed) is not int
        or not 0 <= observed <= expected
        or type(failed) is not int
        or not 0 <= failed <= observed
        or type(row.get("passed")) is not bool
        or type(row.get("exit_code")) is not int
        or not isinstance(row.get("input_error", ""), str)
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
    metrics = result.get("numerical_metrics", {})
    if (
        not isinstance(nonfinite, list)
        or not all(isinstance(value, str) and value in {"reference", "candidate", "unknown"}
                   for value in nonfinite)
        or type(result.get("all_pass")) is not bool
        or not isinstance(metrics, dict)
        or not all(value is None or type(value) in {int, float} and _finite_metric(value)
                   for value in metrics.values())
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
        and failed == 0
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


_METRIC_NAMES = frozenset({
    "relative_l2", "max_row_relative_l2", "max_elementwise_abs_diff",
    "max_elementwise_rel_diff", "max_abs_err", "max_rel_err",
})


def _finite_metric(value):
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def interpret_numerical_output(
    stdout: str, plan: Mapping[str, Any], suite: Mapping[str, Any],
    shapes: Mapping[str, Any], *, transport_returncode: int = 0,
) -> NumericalProbeResult:
    """Decode and interpret the first numerical receipt from captured output.

    Missing/invalid receipts never constitute a measured candidate failure.
    Shape/selection data are evaluator inputs; only bounded aggregate facts leave
    this module. Transport retry remains the caller's responsibility.
    """
    receipt_status = "missing"
    batch = {"schema_version": 1, "runs": []}
    for line in stdout.splitlines():
        if line.startswith(PREFIX):
            try:
                batch = json.loads(line[len(PREFIX):])
            except (ValueError, RecursionError):
                receipt_status = "invalid_json"
                batch = {"schema_version": 1, "runs": "invalid"}
            else:
                receipt_status = "valid" if isinstance(batch, dict) else "invalid_schema"
            break
    if transport_returncode:
        # Retain any safe aggregate diagnostics, but never admit this receipt.
        batch = dict(batch) if isinstance(batch, dict) else {"schema_version": 1}
        batch["error"] = "numerical probe transport failed"
    verdict = _probe_status(batch, plan, suite, shapes)
    rows = batch.get("runs") if isinstance(batch, dict) else None
    row = (rows[0] if isinstance(rows, list) and len(rows) == 1
           and isinstance(rows[0], dict) else {})
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    metrics = result.get("numerical_metrics")
    metrics = {
        name: value for name, value in (metrics.items() if isinstance(metrics, dict) else ())
        if name in _METRIC_NAMES
        and (value is None or type(value) in {int, float} and _finite_metric(value))
    }
    roles = result.get("nonfinite_outputs")
    roles = tuple(role for role in ("reference", "candidate", "unknown")
                  if isinstance(roles, list) and role in roles)
    if "unknown" in roles:
        diagnosis = "non-finite output diagnostic has unknown roles; repair or verify the probe evaluator"
    elif "reference" in roles:
        diagnosis = ("reference output is non-finite under the requested probe inputs; "
                     "repair the input distribution or packed encoding, preserving the original risk case")
    elif "candidate" in roles:
        diagnosis = "candidate output is non-finite while the reference output is finite"
    elif transport_returncode:
        diagnosis = "numerical probe transport failed; private diagnostics withheld"
    elif isinstance(batch, dict) and batch.get("error"):
        diagnosis = "numerical evaluator failed; private diagnostics withheld"
    elif row.get("input_error"):
        diagnosis = "input generation or evaluator execution incomplete; private diagnostics withheld"
    else:
        diagnosis = ""
    return NumericalProbeResult(
        **verdict, receipt_status=receipt_status, metrics=metrics,
        nonfinite_outputs=roles,
        exit_code=row.get("exit_code") if type(row.get("exit_code")) is int else None,
        diagnosis=diagnosis,
    )
