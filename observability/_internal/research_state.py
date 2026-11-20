#!/usr/bin/env python3
"""Build private Research Store and Agent-visible Research Board projections."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.workflow.light.research.store import _apply, _empty  # noqa: E402
from observability._internal.common import (  # noqa: E402
    add_output_arguments,
    discover_scopes,
    emit,
    main_error,
    observed_at,
    read_json,
    table,
)


def _research_state(root: Path) -> dict:
    """Rebuild Research state without creating locks or rewriting snapshots."""
    state = _empty()
    segments = []
    for path in root.glob("events-*.jsonl") if root.is_dir() else ():
        try:
            sequence = int(path.stem.removeprefix("events-"))
        except ValueError:
            continue
        segments.append((sequence, path))
    paths = [path for _sequence, path in sorted(segments)]
    paths.append(root / "events.jsonl")
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("sequence") != state["sequence"] + 1:
                raise ValueError(f"Research event sequence is invalid at {path}")
            _apply(state, event)
            if event.get("type") in {"queue.synchronized", "shared_queue.synchronized"}:
                state["_observability_last_shared_queue_sync"] = {
                    "sequence": event.get("sequence"),
                    "origin": event.get("origin") if isinstance(event.get("origin"), dict) else {},
                    "recorded_at": event.get("recorded_at") or event.get("created_at"),
                }
    return state


def _peer_refs(workspace: Path, *, limit: int = 64) -> dict:
    process = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--sort=-creatordate",
            "--format=%(refname)%09%(objectname)",
            "refs/remotes/atrex",
        ],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return {"available": False, "count": None, "returned": 0, "truncated": False, "refs": []}
    rows = []
    for line in process.stdout.splitlines():
        ref, _, commit = line.partition("\t")
        if ref and commit:
            rows.append({"ref": ref, "kernel_commit_id": commit})
    return {
        "available": True,
        "count": len(rows),
        "returned": min(len(rows), limit),
        "truncated": len(rows) > limit,
        "refs": rows[:limit],
    }


def _latest_modules(scope) -> list[str] | None:
    value = read_json(scope.audit_root / "state.json", default={})
    if not isinstance(value, dict):
        return None
    modules = value.get("tool_modules")
    if isinstance(modules, list):
        return [item for item in modules if item in {"directions", "experiments"}]
    if value.get("research_records") is False:
        return []
    return None


def _claim_review_summary(scope) -> dict:
    qualities = Counter()
    statuses = Counter()
    total = 0
    from aka.workflow.light.support.paths import supervisor_campaign_base_root

    library = supervisor_campaign_base_root(scope.workspace) / "measurements" / "reviews"
    paths = [
        *library.glob("direction_*/v*/result.json"),
        *library.glob("direction_*/attempts/*/result.json"),
        *library.glob("experiment_*/v*/result.json"),
        *library.glob("experiment_*/attempts/*/result.json"),
    ]
    for path in sorted(paths):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        origin = value.get("origin")
        trajectory = origin.get("trajectory") if isinstance(origin, dict) else None
        if scope.trajectory is not None and trajectory != scope.trajectory:
            continue
        total += 1
        statuses[str(value.get("status") or "unknown")] += 1
        review = value.get("review")
        quality = review.get("claim_review_quality") if isinstance(review, dict) else None
        qualities[str(quality or "unavailable")] += 1
    # Read-only compatibility for pre-object-bound Campaigns.
    from aka.workflow.light.support.paths import supervisor_research_catalog_root

    legacy_paths = [
        *(scope.audit_root / "claim-audit").glob("*/outcome.json"),
        *(
            supervisor_research_catalog_root(scope.workspace)
            / "reviews"
            / (
                f"T{scope.trajectory}"
                if scope.trajectory is not None else "T1"
            )
        ).glob("e*/claim-audit/*/outcome.json"),
    ]
    for path in sorted(legacy_paths):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        total += 1
        statuses[str(value.get("status") or "unknown")] += 1
        result = value.get("result")
        review = result.get("review") if isinstance(result, dict) else None
        if not isinstance(review, dict):
            review = value.get("private_review")
        quality = review.get("claim_review_quality") if isinstance(review, dict) else None
        qualities[str(quality or "unavailable")] += 1
    return {
        "count": total,
        "status_counts": dict(sorted(statuses.items())),
        "quality_counts": dict(sorted(qualities.items())),
    }


def _direction_row(item: dict, *, local_rank: int | None, shared_rank: int | None) -> dict:
    return {
        key: value
        for key, value in {
            "direction_id": item.get("direction_id"),
            "name": item.get("name"),
            "status": item.get("status"),
            "access": item.get("access"),
            "owner": item.get("owner"),
            "proposed_by": item.get("proposed_by"),
            "change_scope": item.get("change_scope"),
            "claim_review_quality": item.get(
                "claim_review_quality", item.get("review_quality")
            ),
            "local_rank": local_rank,
            "shared_rank": shared_rank,
        }.items()
        if value is not None
    }


def snapshot(workspace: Path) -> dict:
    scopes = []
    for scope in discover_scopes(workspace):
        state = _research_state(scope.audit_root / "research")
        local_rank = {value: index for index, value in enumerate(state["queue"], 1)}
        shared_rank = {
            value: index for index, value in enumerate(state["shared_queue"], 1)
        }
        directions = [
            _direction_row(
                item,
                local_rank=local_rank.get(direction_id),
                shared_rank=shared_rank.get(direction_id),
            )
            for direction_id, item in state["directions"].items()
        ]
        directions.sort(
            key=lambda item: (
                item.get("local_rank", 1 << 30),
                item.get("shared_rank", 1 << 30),
                str(item.get("direction_id")),
            )
        )
        experiments = []
        for item in state["experiments"].values():
            experiments.append(
                {
                    key: value
                    for key, value in {
                        "experiment_id": item.get("experiment_id"),
                        "direction_id": item.get("direction_id"),
                        "name": item.get("name"),
                        "access": item.get("access"),
                        "recorded_by": item.get("recorded_by"),
                        "claim_review_quality": item.get(
                            "claim_review_quality", item.get("review_quality")
                        ),
                        "gateway_record_ids": item.get("gateway_record_ids"),
                        "candidate_kernel_commit_id": item.get(
                            "candidate_kernel_commit_id"
                        ),
                    }.items()
                    if value not in (None, [], {})
                }
            )
        progress = scope.agent_workspace / "PROGRESS.md"
        enabled_modules = _latest_modules(scope)
        current_lineage = f"T{scope.trajectory}" if scope.trajectory is not None else None
        current_lineage_state = (
            state["lineages"].get(current_lineage, {})
            if current_lineage is not None
            else {}
        )
        scopes.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "agent_workspace": str(scope.agent_workspace),
                "revision": state["sequence"],
                "enabled_modules": enabled_modules,
                "module_configuration_available": enabled_modules is not None,
                "direction_counts": dict(
                    sorted(Counter(str(row.get("status") or "unknown") for row in directions).items())
                ),
                "experiment_count": len(experiments),
                "directions": directions,
                "experiments": experiments,
                "lineages": state["lineages"],
                "focus_direction_id": current_lineage_state.get("focus_direction_id"),
                "active_direction_ids": current_lineage_state.get("active_direction_ids", []),
                "local_direction_queue": list(state["queue"]),
                "shared_direction_queue": list(state["shared_queue"]),
                "last_shared_queue_sync": state.get(
                    "_observability_last_shared_queue_sync"
                ),
                "peer_git_refs": _peer_refs(scope.agent_workspace),
                "claim_reviews": _claim_review_summary(scope),
                "progress_projection": {
                    "path": str(progress),
                    "exists": progress.is_file(),
                    "size_bytes": progress.stat().st_size if progress.is_file() else None,
                },
            }
        )
    return {
        "schema_version": 2,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "scopes": scopes,
    }


def render(value: dict) -> str:
    rows = []
    for scope in value["scopes"]:
        counts = scope["direction_counts"]
        modules = scope["enabled_modules"]
        rows.append(
            [
                scope["scope"],
                "unavailable" if modules is None else ",".join(modules) or "off",
                scope["revision"],
                sum(counts.values()),
                counts.get("in_progress", 0),
                counts.get("deferred", 0) + counts.get("blocked", 0),
                scope["experiment_count"],
                scope["claim_reviews"]["count"],
                scope["progress_projection"]["exists"],
            ]
        )
    return "\n".join(
        [
            "AKA Light Research Board",
            "",
            table(
                [
                    "Scope", "Modules", "Revision", "Directions", "Active",
                    "Paused", "Experiments", "Reviews", "PROGRESS",
                ],
                rows,
            ),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-research-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
