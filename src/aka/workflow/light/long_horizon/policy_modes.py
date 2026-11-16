"""Canonical final-source policies for AKA Light submissions."""
from __future__ import annotations


SUBMISSION_POLICY_MODES = ("fast", "performance", "ultra")
ALL_SUBMISSION_POLICY_MODES = SUBMISSION_POLICY_MODES


def canonical_submission_policy(mode: str, *, specify_dsl: bool = True) -> str:
    """Validate one of the three explicit submission policies."""
    if mode not in SUBMISSION_POLICY_MODES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    return mode


def dsl_policy(mode: str, *, specify_dsl: bool = True) -> str:
    policy = canonical_submission_policy(mode, specify_dsl=specify_dsl)
    return {
        "fast": "open",
        "performance": "guided",
        "ultra": "pinned",
    }[policy]


def compute_policy(mode: str, *, specify_dsl: bool = True) -> str:
    policy = canonical_submission_policy(mode, specify_dsl=specify_dsl)
    return (
        "declared_external_allowed"
        if policy == "fast"
        else "self_authored_only"
    )


def requires_explicit_dsl(mode: str, *, specify_dsl: bool = True) -> bool:
    return dsl_policy(mode, specify_dsl=specify_dsl) == "pinned"


def permits_hidden_shapes(mode: str, *, specify_dsl: bool = True) -> bool:
    """Hidden evaluator inputs remain restricted to strict final-source modes."""
    return canonical_submission_policy(mode, specify_dsl=specify_dsl) in {
        "performance",
        "ultra",
    }


def review_fields(mode: str, *, specify_dsl: bool = True) -> tuple[str, ...]:
    """Return only checks that are meaningful in the Agent-facing receipt."""
    policy = canonical_submission_policy(mode, specify_dsl=specify_dsl)
    common = (
        "dependency_inventory",
        "external_code_loading",
        "solution_manifest",
    )
    if policy == "fast":
        return ()
    if policy == "performance":
        return ("compute_provenance", *common)
    return ("dsl_policy", "compute_provenance", *common)
