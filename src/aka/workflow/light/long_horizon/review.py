"""Thin AKA Light projection of the upstream production review contract."""
from __future__ import annotations

from aka.acceptance.review import _validate_production_review


def validate_light_production_review(
    payload: object,
    *,
    candidate_files: frozenset[str],
    dsl_policy: str,
    allow_provided_bootstrap: bool = False,
) -> tuple[list[str], str, dict[str, object]]:
    """Validate upstream schema v2, then apply Light's DSL applicability.

    Fast never invokes this adapter. Performance projects only its applicable
    entries, without rewriting Reviewer decisions. Ultra projects every entry.
    """
    if dsl_policy not in {"guided", "pinned"}:
        raise ValueError(
            "independent final-source review is only applicable to "
            "Performance or Ultra submissions"
        )

    # This is the authoritative upstream schema/category/evidence validator.
    _validate_production_review(payload, candidate_files=candidate_files,
                                allow_provided_bootstrap=allow_provided_bootstrap)
    assert isinstance(payload, dict)

    applicable_check_ids = {
        "compute_provenance",
        "dependency_inventory",
        "external_code_loading",
        "solution_manifest",
    }
    if dsl_policy == "pinned":
        applicable_check_ids.add("framework_compliance")

    checks: list[dict[str, object]] = []
    for raw in payload["checks"]:
        assert isinstance(raw, dict)
        if raw["id"] in applicable_check_ids:
            checks.append(dict(raw))

    dependencies = [
        dict(raw)
        for raw in payload["dependencies"]
        if not (
            dsl_policy == "guided"
            and isinstance(raw, dict)
            and raw.get("category") == "alternate_framework"
        )
    ]
    errors: list[str] = []
    for item in checks:
        if item["decision"] == "reject":
            errors.append(
                "production candidate rejected by independent reviewer: "
                f"{item['id']}: {item['reason']}"
            )
    for item in dependencies:
        if item["decision"] == "reject":
            errors.append(
                "third-party dependency rejected by independent reviewer: "
                f"{item['name']}: {item['reason']}"
            )

    policy_result: dict[str, object] = {
        "schema_version": payload["schema_version"],
        "verdict": "reject" if errors else "allow",
        "checks": checks,
        "dependencies": dependencies,
        "summary": payload["summary"],
    }
    return errors, str(payload["summary"]).strip(), policy_result
