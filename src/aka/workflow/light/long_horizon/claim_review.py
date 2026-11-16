"""Bounded validation for independent research-claim reviews."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


VERDICTS = frozenset({"supported", "partially_supported", "unsupported"})
CONFIDENCE = frozenset({"high", "medium", "low"})
MAX_FINDINGS = 6

# Drafts and unavailable reviewer runs are distinct from reviewer verdicts.
def canonical_review_quality(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _text(value: object, field: str, *, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"research review {field} must be non-empty text")
    return value.strip()


def validate_review(value: object) -> dict[str, Any]:
    """Return a canonical strict three-tier review or reject malformed output."""
    if not isinstance(value, Mapping):
        raise ValueError("research review must be a JSON object")
    allowed = {
        "verdict", "primary_issue", "findings", "scope_safe_interpretation",
        "next_check", "confidence",
    }
    if set(value) != allowed:
        raise ValueError("research review fields do not match the contract")
    verdict = value.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("research review verdict is invalid")
    confidence = value.get("confidence")
    if confidence not in CONFIDENCE:
        raise ValueError("research review confidence is invalid")
    findings = value.get("findings")
    if not isinstance(findings, list) or not 1 <= len(findings) <= MAX_FINDINGS:
        raise ValueError("research review requires one to six findings")
    normalized = []
    for index, raw in enumerate(findings):
        if not isinstance(raw, Mapping) or set(raw) != {
            "claim", "judgment", "decision_critical", "reason", "evidence"
        }:
            raise ValueError(f"research review finding {index} fields are invalid")
        judgment = raw.get("judgment")
        if judgment not in VERDICTS:
            raise ValueError(f"research review finding {index} judgment is invalid")
        if type(raw.get("decision_critical")) is not bool:
            raise ValueError(
                f"research review finding {index} decision_critical must be boolean"
            )
        evidence = raw.get("evidence")
        if (
            not isinstance(evidence, list)
            or len(evidence) > 32
            or any(not isinstance(item, str) or not item.strip() or len(item) > 1024
                   for item in evidence)
        ):
            raise ValueError(f"research review finding {index} evidence is invalid")
        normalized.append({
            "claim": _text(raw.get("claim"), f"findings.{index}.claim"),
            "judgment": judgment,
            "decision_critical": raw["decision_critical"],
            "reason": _text(raw.get("reason"), f"findings.{index}.reason"),
            "evidence": [item.strip() for item in evidence],
        })
    critical = [item for item in normalized if item["decision_critical"]]
    if not critical:
        raise ValueError("research review requires a decision-critical finding")
    critical_judgments = [item["judgment"] for item in critical]
    if verdict == "supported" and any(
        item != "supported" for item in critical_judgments
    ):
        raise ValueError(
            "supported review requires every decision-critical finding to be supported"
        )
    if verdict == "partially_supported" and (
        "partially_supported" not in critical_judgments
        or "unsupported" in critical_judgments
    ):
        raise ValueError(
            "partially_supported review requires a decision-critical partial finding "
            "and no decision-critical unsupported finding"
        )
    if verdict == "unsupported" and "unsupported" not in critical_judgments:
        raise ValueError(
            "unsupported review requires a decision-critical unsupported finding"
        )
    return {
        "verdict": verdict,
        "primary_issue": _text(value.get("primary_issue"), "primary_issue"),
        "findings": deepcopy(normalized),
        "scope_safe_interpretation": _text(
            value.get("scope_safe_interpretation"), "scope_safe_interpretation"
        ),
        "next_check": _text(value.get("next_check"), "next_check"),
        "confidence": confidence,
    }
