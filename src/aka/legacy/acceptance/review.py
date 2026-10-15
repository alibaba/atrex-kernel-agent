"""Acceptance review implementation and callable interface."""
from __future__ import annotations

from pathlib import Path
import hashlib
DEPENDENCY_REVIEW_SCHEMA_VERSION = 2
import re


_DEPENDENCY_ALLOW_CATEGORIES = {
    "toolchain_plumbing",
    "framework_runtime",
    "support_utility",
}


_DEPENDENCY_REJECT_CATEGORIES = {
    "prebuilt_compute",
    "alternate_framework",
    "hidden_dispatch",
    "external_code",
    "unresolved",
}


_PRODUCTION_REVIEW_CHECK_CATEGORIES = {
    "framework_compliance": {
        "allow": {"framework_compliant"},
        "reject": {"alternate_framework", "unresolved"},
    },
    "compute_provenance": {
        "allow": {"self_authored_compute"},
        "reject": {
            "prebuilt_compute",
            "torch_compute",
            "hidden_dispatch",
            "external_code",
            "unresolved",
        },
    },
    "dependency_inventory": {
        "allow": {"inventory_complete"},
        "reject": {"unresolved"},
    },
    "external_code_loading": {
        "allow": {"no_external_code"},
        "reject": {"external_code", "hidden_dispatch", "unresolved"},
    },
    "solution_manifest": {
        "allow": {"manifest_consistent", "not_applicable"},
        "reject": {"manifest_mismatch", "unresolved"},
    },
}


def _production_review_candidate_paths(workspace: Path) -> list[Path]:
    """Return the complete, bounded source set shown to the production reviewer."""
    paths = [
        workspace / "kernel.py",
        workspace / "solution.json",
    ]
    return [path for path in paths if path.is_file()]


def _production_review_digest(
    workspace: Path,
    framework: str,
    require_gluon: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"production-review-v{DEPENDENCY_REVIEW_SCHEMA_VERSION}\0".encode())
    digest.update(framework.encode("utf-8", errors="replace"))
    digest.update(b"\0gluon-required\0" if require_gluon else b"\0selected-framework\0")
    for path in _production_review_candidate_paths(workspace):
        relative = path.relative_to(workspace).as_posix()
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _validate_review_item(
    item: object,
    *,
    label: str,
    allow_categories: set[str],
    reject_categories: set[str],
    candidate_files: frozenset[str],
) -> tuple[str, str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"production review {label} must be an object")
    decision = item.get("decision")
    category = item.get("category")
    reason = item.get("reason")
    evidence = item.get("evidence")
    if decision not in {"allow", "reject"}:
        raise ValueError(f"production review decision is invalid for {label}")
    categories = allow_categories if decision == "allow" else reject_categories
    if category not in categories:
        raise ValueError(
            f"production review category {category!r} is inconsistent with "
            f"decision {decision!r} for {label}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"production review reason is empty for {label}")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(value, str) and value.strip() for value in evidence)
    ):
        raise ValueError(f"production review evidence is invalid for {label}")
    candidate_evidence_found = False
    for value in evidence:
        stripped = value.strip()
        match = re.fullmatch(
            r"candidate/(kernel\.py|solution\.json)(?::.+)?",
            stripped,
        )
        if match is not None and match.group(1) in candidate_files:
            candidate_evidence_found = True
            continue
        # The review request is trusted supervisor context rather than candidate
        # evidence.  Reviewers occasionally cite it alongside candidate source when
        # explaining the selected framework.  Permit that supplemental citation,
        # while still requiring source-backed candidate evidence for every item.
        if re.fullmatch(r"review_request\.json:[A-Za-z0-9_.-]+", stripped):
            continue
        raise ValueError(
            "production review evidence may only cite supplied candidate files "
            f"or review_request.json metadata for {label}"
        )
    if not candidate_evidence_found:
        raise ValueError(
            f"production review evidence must cite a supplied candidate file for {label}"
        )
    return decision, str(category), reason.strip()


def _validate_production_review(
    payload: object,
    *,
    candidate_files: frozenset[str] = frozenset({"kernel.py", "solution.json"}),
) -> tuple[list[str], str]:
    """Validate a complete candidate verdict and translate rejections into errors."""
    if not isinstance(payload, dict):
        raise ValueError("production review must be a JSON object")
    if payload.get("schema_version") != DEPENDENCY_REVIEW_SCHEMA_VERSION:
        raise ValueError("production review has an unsupported schema_version")
    verdict = payload.get("verdict")
    if verdict not in {"allow", "reject"}:
        raise ValueError("production review verdict must be allow or reject")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("production review summary must be non-empty")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("production review checks must be a list")

    expected = set(_PRODUCTION_REVIEW_CHECK_CATEGORIES)
    reviewed: set[str] = set()
    rejected: list[str] = []
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("production review check must be an object")
        check_id = item.get("id")
        if not isinstance(check_id, str) or check_id not in expected:
            raise ValueError(
                f"production review returned unexpected check id: {check_id!r}"
            )
        if check_id in reviewed:
            raise ValueError(f"production review duplicated check id: {check_id}")
        categories = _PRODUCTION_REVIEW_CHECK_CATEGORIES[check_id]
        decision, _category, reason = _validate_review_item(
            item,
            label=f"check {check_id}",
            allow_categories=categories["allow"],
            reject_categories=categories["reject"],
            candidate_files=candidate_files,
        )
        reviewed.add(check_id)
        if decision == "reject":
            rejected.append(
                "production candidate rejected by independent reviewer: "
                f"{check_id}: {reason}"
            )

    missing = sorted(expected - reviewed)
    if missing:
        raise ValueError("production review omitted check ids: " + ", ".join(missing))

    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("production review dependencies must be a list")
    dependency_names: set[str] = set()
    for item in dependencies:
        if not isinstance(item, dict):
            raise ValueError("production review dependency must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("production review dependency name must be non-empty")
        normalized_name = name.strip().lower()
        if normalized_name in dependency_names:
            raise ValueError(f"production review duplicated dependency: {name.strip()}")
        dependency_names.add(normalized_name)
        decision, _category, reason = _validate_review_item(
            item,
            label=f"dependency {name.strip()}",
            allow_categories=_DEPENDENCY_ALLOW_CATEGORIES,
            reject_categories=_DEPENDENCY_REJECT_CATEGORIES,
            candidate_files=candidate_files,
        )
        if decision == "reject":
            rejected.append(
                "third-party dependency rejected by independent reviewer: "
                f"{name.strip()}: {reason}"
            )

    expected_verdict = "reject" if rejected else "allow"
    if verdict != expected_verdict:
        raise ValueError(
            f"production review verdict {verdict!r} disagrees with item decisions"
        )
    return rejected, summary.strip()
