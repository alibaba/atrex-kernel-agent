"""Portable, append-only Direction genealogy validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

RELATIONSHIPS = ("retry", "refinement", "reimplementation", "correction", "port", "combination")
RELATIONSHIP_FIELDS = frozenset(
    {
        "relationship",
        "derived_from_direction_ids",
        "derived_from_experiment_ids",
        "supersedes_direction_id",
    }
)


def relationship_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Omit empty defaults so historical Directions acquire no invented edges."""
    return {key: value[key] for key in sorted(RELATIONSHIP_FIELDS) if value.get(key)}


def _ids(raw: object, prefix: str, field: str) -> list[str]:
    if not isinstance(raw, (list, tuple)) or len(raw) > 32:
        raise ValueError(f"{field} must contain at most 32 unique IDs")
    values = list(raw)
    if any(
        not isinstance(item, str) or not re.fullmatch(prefix + r"_[0-9a-f]{32}", item)
        for item in values
    ):
        raise ValueError(f"{field} contains an invalid {prefix} ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicate IDs")
    return values


def validate_relationship(
    direction_id: str,
    value: Mapping[str, Any],
    directions: Mapping[str, Mapping[str, Any]],
    experiments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate Agent-authored ancestry against exactly the caller's visible history."""
    kind = value.get("relationship")
    parents = _ids(
        value.get("derived_from_direction_ids", []), "direction", "derived_from_direction_ids"
    )
    evidence = _ids(
        value.get("derived_from_experiment_ids", []), "experiment", "derived_from_experiment_ids"
    )
    supersedes = value.get("supersedes_direction_id")
    if supersedes is not None:
        _ids([supersedes], "direction", "supersedes_direction_id")
    if kind is None and not parents and not evidence and supersedes is None:
        return {}
    if kind not in RELATIONSHIPS:
        raise ValueError(f"relationship must be one of {', '.join(RELATIONSHIPS)}")
    for parent in parents:
        if parent not in directions:
            raise ValueError(
                f"Parent Direction {parent} is not visible; load an existing Direction first"
            )
    effective_parents = set(parents)
    for experiment_id in evidence:
        experiment = experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(
                f"Parent Experiment {experiment_id} is not visible; "
                "load an existing Experiment first"
            )
        parent = str(experiment.get("direction_id"))
        if parent not in directions:
            raise ValueError(f"Parent Experiment {experiment_id} has no visible owning Direction")
        effective_parents.add(parent)
    if not effective_parents:
        raise ValueError("relationship requires at least one parent Direction or Experiment")
    if kind == "combination" and len(effective_parents) < 2:
        raise ValueError(
            "combination requires at least two distinct parent Directions, "
            "directly or via Experiments"
        )
    if supersedes is not None and (kind != "correction" or supersedes not in effective_parents):
        raise ValueError(
            "supersedes_direction_id requires correction and must name one of its parent Directions"
        )

    pending = list(effective_parents)
    visited: set[str] = set()
    while pending:
        parent = pending.pop()
        if parent == direction_id:
            raise ValueError("Direction genealogy cannot contain self-references or cycles")
        if parent in visited:
            continue
        visited.add(parent)
        prior = directions.get(parent, {})
        pending.extend(prior.get("derived_from_direction_ids") or [])
        for experiment_id in prior.get("derived_from_experiment_ids") or []:
            source = experiments.get(experiment_id)
            if source is not None:
                pending.append(str(source["direction_id"]))
    return {
        "relationship": kind,
        "derived_from_direction_ids": parents,
        "derived_from_experiment_ids": evidence,
        "supersedes_direction_id": supersedes,
    }
