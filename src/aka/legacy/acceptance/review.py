"""Acceptance review facts and deterministic policy evaluation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DEPENDENCY_REVIEW_SCHEMA_VERSION = 3


_COMPUTE_KINDS = {
    "self_authored",
    "framework_builtin",
    "prebuilt_operator",
    "external_source",
    "unresolved",
}

_COMPUTE_ROLES = {
    "primary",
    "fallback",
    "partial",
    "dispatch_only",
}

_DEPENDENCY_KINDS = {
    "toolchain_plumbing",
    "framework_runtime",
    "support_utility",
    "prebuilt_operator",
    "external_source",
    "alternate_framework",
    "unresolved",
}

_EXTERNAL_CODE_STATUSES = {
    "none",
    "candidate_source",
    "declared_dependency",
    "hidden_dispatch",
    "unresolved",
}

_MANIFEST_STATUSES = {
    "consistent",
    "not_applicable",
    "mismatch",
    "unresolved",
}


def _production_review_candidate_paths(workspace: Path) -> list[Path]:
    """Return the complete, bounded source set shown to the production reviewer."""
    paths = [workspace / "kernel.py", workspace / "solution.json"]
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


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"production review {label} must be an object")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"production review {label} must be non-empty")
    return value.strip()


def _validate_evidence(
    value: object,
    *,
    label: str,
    candidate_files: frozenset[str],
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"production review evidence is invalid for {label}")
    evidence = [str(item).strip() for item in value]
    candidate_evidence_found = False
    for item in evidence:
        match = re.fullmatch(r"candidate/([^:]+)(?::.+)?", item)
        if match is not None and match.group(1) in candidate_files:
            candidate_evidence_found = True
            continue
        if re.fullmatch(r"review_request\.json:[A-Za-z0-9_.-]+", item):
            continue
        raise ValueError(
            "production review evidence may only cite supplied candidate files "
            f"or review_request.json metadata for {label}"
        )
    if not candidate_evidence_found:
        raise ValueError(
            "production review evidence must cite a supplied candidate file "
            f"for {label}"
        )
    return evidence


def _validate_explanation(
    value: object,
    *,
    label: str,
    candidate_files: frozenset[str],
) -> tuple[str, list[str]]:
    item = _require_object(value, label=label)
    if "decision" in item or "verdict" in item:
        raise ValueError(
            f"production review {label} must report facts, not decision or verdict"
        )
    reason = _require_text(item.get("reason"), label=f"reason for {label}")
    evidence = _validate_evidence(
        item.get("evidence"), label=label, candidate_files=candidate_files
    )
    return reason, evidence


def _derived_check(
    check_id: str,
    *,
    rejected_reasons: list[str],
    facts: object,
) -> dict[str, object]:
    return {
        "id": check_id,
        "decision": "reject" if rejected_reasons else "allow",
        "reasons": rejected_reasons,
        "facts": facts,
    }


def _validate_production_review(
    payload: object,
    *,
    candidate_files: frozenset[str] = frozenset({"kernel.py", "solution.json"}),
    dsl_policy: str = "pinned",
    compute_policy: str = "self_authored_only",
) -> tuple[list[str], str, dict[str, object]]:
    """Validate reviewer facts and deterministically apply final-source policy."""
    if dsl_policy not in {"open", "guided", "pinned"}:
        raise ValueError(f"unsupported production review dsl_policy: {dsl_policy!r}")
    if compute_policy not in {
        "declared_external_allowed",
        "self_authored_only",
    }:
        raise ValueError(
            f"unsupported production review compute_policy: {compute_policy!r}"
        )

    review = _require_object(payload, label="payload")
    if review.get("schema_version") != DEPENDENCY_REVIEW_SCHEMA_VERSION:
        raise ValueError("production review has an unsupported schema_version")
    if "verdict" in review or "decision" in review:
        raise ValueError(
            "production review must report source facts; Runtime derives "
            "decisions and verdict"
        )
    summary = _require_text(review.get("summary"), label="summary")

    errors: list[str] = []
    checks: list[dict[str, object]] = []

    framework = _require_object(review.get("framework"), label="framework")
    framework_reason, _ = _validate_explanation(
        framework, label="framework", candidate_files=candidate_files
    )
    technologies = framework.get("technologies")
    if (
        not isinstance(technologies, list)
        or not technologies
        or not all(isinstance(item, str) and item.strip() for item in technologies)
    ):
        raise ValueError("production review framework technologies are invalid")
    for field in (
        "evaluator_supported",
        "configured_framework_used",
        "required_phase_used",
    ):
        if not isinstance(framework.get(field), bool):
            raise ValueError(f"production review framework {field} must be boolean")
    unexpected_technologies = framework.get("unexpected_core_technologies")
    if not isinstance(unexpected_technologies, list) or not all(
        isinstance(item, str) and item.strip() for item in unexpected_technologies
    ):
        raise ValueError(
            "production review framework unexpected_core_technologies is invalid"
        )
    framework_rejections = []
    if not framework["evaluator_supported"]:
        framework_rejections.append(framework_reason)
    if dsl_policy == "pinned" and (
        not framework["configured_framework_used"]
        or not framework["required_phase_used"]
        or bool(unexpected_technologies)
    ):
        framework_rejections.append(framework_reason)
    checks.append(
        _derived_check(
            "framework_compliance",
            rejected_reasons=framework_rejections,
            facts=framework,
        )
    )

    origins = review.get("compute_origins")
    if not isinstance(origins, list) or not origins:
        raise ValueError("production review compute_origins must be a non-empty list")
    origin_rejections: list[str] = []
    derived_origins: list[dict[str, object]] = []
    for index, value in enumerate(origins):
        label = f"compute origin {index}"
        origin = _require_object(value, label=label)
        reason, _ = _validate_explanation(
            origin, label=label, candidate_files=candidate_files
        )
        kind = origin.get("kind")
        role = origin.get("role")
        technology = _require_text(
            origin.get("technology"), label=f"technology for {label}"
        )
        if kind not in _COMPUTE_KINDS:
            raise ValueError(f"production review kind is invalid for {label}")
        if role not in _COMPUTE_ROLES:
            raise ValueError(f"production review role is invalid for {label}")
        for field in ("declared", "reproducible", "hidden_dispatch"):
            if not isinstance(origin.get(field), bool):
                raise ValueError(
                    f"production review {field} must be boolean for {label}"
                )

        rejected = (
            kind == "unresolved"
            or bool(origin["hidden_dispatch"])
            or not bool(origin["declared"])
            or not bool(origin["reproducible"])
            or (
                kind in {"framework_builtin", "prebuilt_operator", "external_source"}
                and compute_policy != "declared_external_allowed"
            )
        )
        if rejected:
            origin_rejections.append(f"{technology} ({role}, {kind}): {reason}")
        derived_origin = dict(origin)
        derived_origin["decision"] = "reject" if rejected else "allow"
        derived_origins.append(derived_origin)
    checks.append(
        _derived_check(
            "compute_provenance",
            rejected_reasons=origin_rejections,
            facts=derived_origins,
        )
    )

    inventory = _require_object(
        review.get("dependency_inventory"), label="dependency_inventory"
    )
    inventory_reason, _ = _validate_explanation(
        inventory,
        label="dependency_inventory",
        candidate_files=candidate_files,
    )
    if not isinstance(inventory.get("complete"), bool):
        raise ValueError(
            "production review dependency_inventory complete must be boolean"
        )
    inventory_rejections = [] if inventory["complete"] else [inventory_reason]
    checks.append(
        _derived_check(
            "dependency_inventory",
            rejected_reasons=inventory_rejections,
            facts=inventory,
        )
    )

    external = _require_object(
        review.get("external_code_loading"), label="external_code_loading"
    )
    external_reason, _ = _validate_explanation(
        external,
        label="external_code_loading",
        candidate_files=candidate_files,
    )
    external_status = external.get("status")
    if external_status not in _EXTERNAL_CODE_STATUSES:
        raise ValueError("production review external_code_loading status is invalid")
    for field in ("declared", "reproducible", "hidden_dispatch"):
        if not isinstance(external.get(field), bool):
            raise ValueError(
                f"production review external_code_loading {field} must be boolean"
            )
    external_rejected = (
        external_status in {"hidden_dispatch", "unresolved"}
        or bool(external["hidden_dispatch"])
        or not bool(external["declared"])
        or not bool(external["reproducible"])
        or (
            external_status == "declared_dependency"
            and compute_policy != "declared_external_allowed"
        )
    )
    external_rejections = [external_reason] if external_rejected else []
    checks.append(
        _derived_check(
            "external_code_loading",
            rejected_reasons=external_rejections,
            facts=external,
        )
    )

    manifest = _require_object(
        review.get("solution_manifest"), label="solution_manifest"
    )
    manifest_reason, _ = _validate_explanation(
        manifest,
        label="solution_manifest",
        candidate_files=candidate_files,
    )
    manifest_status = manifest.get("status")
    if manifest_status not in _MANIFEST_STATUSES:
        raise ValueError("production review solution_manifest status is invalid")
    manifest_rejections = (
        [] if manifest_status in {"consistent", "not_applicable"} else [manifest_reason]
    )
    checks.append(
        _derived_check(
            "solution_manifest",
            rejected_reasons=manifest_rejections,
            facts=manifest,
        )
    )

    dependencies = review.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("production review dependencies must be a list")
    dependency_names: set[str] = set()
    derived_dependencies: list[dict[str, object]] = []
    dependency_rejections: list[str] = []
    for index, value in enumerate(dependencies):
        label = f"dependency {index}"
        dependency = _require_object(value, label=label)
        reason, _ = _validate_explanation(
            dependency, label=label, candidate_files=candidate_files
        )
        name = _require_text(dependency.get("name"), label=f"name for {label}")
        normalized_name = name.lower()
        if normalized_name in dependency_names:
            raise ValueError(f"production review duplicated dependency: {name}")
        dependency_names.add(normalized_name)
        kind = dependency.get("kind")
        if kind not in _DEPENDENCY_KINDS:
            raise ValueError(f"production review kind is invalid for dependency {name}")
        _require_text(dependency.get("role"), label=f"role for dependency {name}")
        for field in ("declared", "reproducible", "hidden_dispatch"):
            if not isinstance(dependency.get(field), bool):
                raise ValueError(
                    f"production review {field} must be boolean for dependency {name}"
                )
        rejected = (
            kind in {"alternate_framework", "unresolved"}
            or bool(dependency["hidden_dispatch"])
            or not bool(dependency["declared"])
            or not bool(dependency["reproducible"])
            or (
                kind in {"prebuilt_operator", "external_source"}
                and compute_policy != "declared_external_allowed"
            )
        )
        if rejected:
            dependency_rejections.append(f"{name}: {reason}")
        derived_dependency = dict(dependency)
        derived_dependency["decision"] = "reject" if rejected else "allow"
        derived_dependencies.append(derived_dependency)

    for check in checks:
        if check["decision"] == "reject":
            for reason in check["reasons"]:
                errors.append(
                    "final-source candidate rejected by deterministic policy: "
                    f"{check['id']}: {reason}"
                )
    for reason in dependency_rejections:
        errors.append(
            "final-source dependency rejected by deterministic policy: " + reason
        )

    derived = {
        "verdict": "reject" if errors else "allow",
        "dsl_policy": dsl_policy,
        "compute_policy": compute_policy,
        "checks": checks,
        "dependencies": derived_dependencies,
    }
    return errors, summary, derived
