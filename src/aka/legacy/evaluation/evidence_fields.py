"""Pure evidence normalization shared by independent Journal implementations."""

from __future__ import annotations
import math
import re
from typing import Any

WIKI_USAGE_DISPOSITIONS = {
    "applied",
    "partially_applied",
    "reference_only",
    "rejected",
}
WIKI_USAGE_STATUSES = {"declared", "no_material_use", "not_queried"}
EVALUATION_CORRECTNESS = {"pass", "fail", "unknown"}
EVALUATION_PERFORMANCE = {"improved", "not_improved", "unknown"}
PPU_DIAGNOSTIC_ROUTES = {"acu", "comparison", "envelope", "timeline", "joint"}
PPU_EVIDENCE_SCHEMAS = {
    "acu": {"ppu-acu-extraction/v4"},
    "comparison": {"ppu-acu-comparison/v1"},
    "envelope": {"ppu-envelope-measurement/v1"},
    "timeline": {
        "ppu-fixed-slot-receipt/v5",
        "ppu-critical-path-report/v3",
    },
    "joint": {"ppu-joint-profile/v4"},
}
PPU_DIAGNOSTIC_TEXT_FIELDS = (
    "question",
    "kernel_specialization",
    "workload_identity",
    "device_identity",
    "launch_topology",
    "control_pipeline_identity",
    "finding",
    "decision_impact",
)
QUERY_ID_RE = re.compile(r"wiki-query-[0-9a-f]{32}")
WIKI_ID_RE = re.compile(r"(?:gpu_wiki|internal_gpu_wiki)::[A-Za-z0-9][A-Za-z0-9._:-]*")


def normalize_wiki_usage(raw: object) -> tuple[list[dict[str, str]], list[str]]:
    """Validate Wiki attribution without putting telemetry on the promotion path."""
    if not isinstance(raw, list):
        return [], ["wiki_usage must be a list"]
    normalized: list[dict[str, str]] = []
    errors: list[str] = []
    for index, row in enumerate(raw):
        label = f"wiki_usage[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        query_id = row.get("query_id")
        wiki_id = row.get("wiki_id")
        disposition = row.get("disposition")
        use = row.get("use", "")
        evidence = row.get("evidence", "")
        if not isinstance(query_id, str) or not QUERY_ID_RE.fullmatch(query_id.strip()):
            errors.append(f"{label}.query_id must be an emitted Wiki query_id")
            continue
        if not isinstance(wiki_id, str) or not WIKI_ID_RE.fullmatch(wiki_id.strip()):
            errors.append(f"{label}.wiki_id must be an emitted canonical store::record id")
            continue
        if disposition not in WIKI_USAGE_DISPOSITIONS:
            errors.append(
                f"{label}.disposition must be one of " + ", ".join(sorted(WIKI_USAGE_DISPOSITIONS))
            )
            continue
        if not isinstance(use, str) or not isinstance(evidence, str):
            errors.append(f"{label}.use and {label}.evidence must be strings")
            continue
        normalized.append(
            {
                "query_id": query_id.strip(),
                "wiki_id": wiki_id.strip(),
                "disposition": disposition,
                "use": use.strip(),
                "evidence": evidence.strip(),
            }
        )
    return normalized, errors


def normalize_experiment_evaluation(
    raw: object,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize the bounded outcome fields used for Wiki effectiveness joins."""
    if not isinstance(raw, dict):
        return None, ["evaluation must be an object"]
    correctness = raw.get("correctness")
    performance = raw.get("performance")
    errors: list[str] = []
    if correctness not in EVALUATION_CORRECTNESS:
        errors.append(
            "evaluation.correctness must be one of " + ", ".join(sorted(EVALUATION_CORRECTNESS))
        )
    if performance not in EVALUATION_PERFORMANCE:
        errors.append(
            "evaluation.performance must be one of " + ", ".join(sorted(EVALUATION_PERFORMANCE))
        )
    latency_us = raw.get("latency_us")
    if latency_us is not None and (
        isinstance(latency_us, bool)
        or not isinstance(latency_us, (int, float))
        or latency_us < 0
        or (isinstance(latency_us, float) and not math.isfinite(latency_us))
    ):
        errors.append("evaluation.latency_us must be a non-negative number or null")
    kernel_hash = raw.get("kernel_hash")
    if kernel_hash is not None and not isinstance(kernel_hash, str):
        errors.append("evaluation.kernel_hash must be a string or null")
    if errors:
        return None, errors
    return {
        "correctness": correctness,
        "performance": performance,
        "latency_us": latency_us,
        "kernel_hash": kernel_hash.strip() if isinstance(kernel_hash, str) else None,
    }, []


def normalize_accepted_ppu_diagnostics(
    raw: object,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate reusable PPU evidence selected at terminal handoff."""
    if not isinstance(raw, list):
        return [], ["accepted_ppu_diagnostics must be a list"]
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(raw):
        label = f"accepted_ppu_diagnostics[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        route = row.get("route")
        if route not in PPU_DIAGNOSTIC_ROUTES:
            errors.append(
                f"{label}.route must be one of " + ", ".join(sorted(PPU_DIAGNOSTIC_ROUTES))
            )
            continue
        item: dict[str, Any] = {"route": route}
        item_errors: list[str] = []
        for field in PPU_DIAGNOSTIC_TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                item_errors.append(f"{label}.{field} must be a non-empty string")
            else:
                item[field] = value.strip()
        invalidation_conditions = row.get("invalidation_conditions")
        if (
            not isinstance(invalidation_conditions, list)
            or not invalidation_conditions
            or any(
                not isinstance(value, str) or not value.strip() for value in invalidation_conditions
            )
        ):
            item_errors.append(f"{label}.invalidation_conditions must be a non-empty string list")
        else:
            item["invalidation_conditions"] = [value.strip() for value in invalidation_conditions]
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            item_errors.append(f"{label}.evidence must be an object")
        else:
            normalized_evidence: dict[str, str] = {}
            for field in ("artifact", "sha256", "schema", "evidence_id"):
                value = evidence.get(field)
                if not isinstance(value, str) or not value.strip():
                    item_errors.append(f"{label}.evidence.{field} must be a non-empty string")
                else:
                    normalized_evidence[field] = value.strip()
            if (
                normalized_evidence.get("schema")
                and normalized_evidence["schema"] not in PPU_EVIDENCE_SCHEMAS[route]
            ):
                item_errors.append(f"{label}.evidence.schema is not valid for route {route}")
            digest = normalized_evidence.get("sha256", "")
            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                item_errors.append(f"{label}.evidence.sha256 must be a lowercase SHA-256 digest")
            item["evidence"] = normalized_evidence
        if item_errors:
            errors.extend(item_errors)
            continue
        normalized.append(item)
    return normalized, errors


def normalize_wiki_attribution(entry: dict[str, Any]) -> None:
    """Apply the explicit use/no-use contract while preserving legacy declarations."""
    raw_usage_present = "wiki_usage" in entry
    errors: list[str] = []
    if raw_usage_present:
        entry["wiki_usage"], usage_errors = normalize_wiki_usage(entry["wiki_usage"])
        errors.extend(usage_errors)
    raw_query_ids = entry.get("wiki_query_ids")
    query_ids: list[str] = []
    if raw_query_ids is not None:
        if not isinstance(raw_query_ids, list) or any(
            not isinstance(item, str) or not QUERY_ID_RE.fullmatch(item.strip())
            for item in raw_query_ids
        ):
            errors.append("wiki_query_ids must contain only emitted Wiki query_ids")
        else:
            query_ids = list(dict.fromkeys(item.strip() for item in raw_query_ids))
            entry["wiki_query_ids"] = query_ids
    usage_query_ids = list(
        dict.fromkeys(
            row["query_id"]
            for row in entry.get("wiki_usage", [])
            if isinstance(row, dict) and row.get("query_id")
        )
    )
    status = entry.get("wiki_usage_status")
    if status is None and entry.get("wiki_usage"):
        # Compatibility for journals written before the explicit status contract.
        status = "declared"
        entry["wiki_usage_status"] = status
        entry["wiki_usage_status_inferred"] = True
    elif status is not None and status not in WIKI_USAGE_STATUSES:
        errors.append("wiki_usage_status must be one of " + ", ".join(sorted(WIKI_USAGE_STATUSES)))
    if status == "declared" and not entry.get("wiki_usage"):
        errors.append("wiki_usage_status=declared requires at least one valid wiki_usage row")
    if status == "declared":
        if not query_ids:
            query_ids = usage_query_ids
            entry["wiki_query_ids"] = query_ids
        elif not set(usage_query_ids).issubset(query_ids):
            errors.append("wiki_query_ids must include every wiki_usage query_id")
    if status == "no_material_use" and not query_ids:
        errors.append("wiki_usage_status=no_material_use requires non-empty wiki_query_ids")
    if status in {"no_material_use", "not_queried"} and entry.get("wiki_usage"):
        errors.append(f"wiki_usage_status={status} requires an empty or omitted wiki_usage")
    if status == "not_queried" and query_ids:
        errors.append("wiki_usage_status=not_queried requires empty or omitted wiki_query_ids")
    if errors:
        existing = entry.get("wiki_usage_errors")
        entry["wiki_usage_errors"] = [
            *([str(item) for item in existing] if isinstance(existing, list) else []),
            *errors,
        ]
