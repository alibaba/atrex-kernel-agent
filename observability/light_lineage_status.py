#!/usr/bin/env python3
"""Report one factual status row per Light lineage/trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import add_output_arguments, emit, main_error, observed_at, table
from observability._internal.research_state import snapshot as research_snapshot
from observability._internal.runtime_state import snapshot as runtime_snapshot
from observability._internal.token_usage import snapshot as token_snapshot
from observability._internal.view_helpers import usage_sum


def _scope_tokens(tokens: dict, scope: str) -> dict:
    episode_usage = [
        row.get("usage", {})
        for row in tokens.get("episodes", [])
        if row.get("scope") == scope
    ]
    startup_usage = [
        row.get("usage", {})
        for row in tokens.get("sessions", [])
        if row.get("scope") == scope and row.get("stage") != "episode"
    ]
    review_usage = [
        row.get("usage", {})
        for row in tokens.get("reviews", [])
        if row.get("scope") == scope
    ]
    main = usage_sum([*episode_usage, *startup_usage])
    reviews = usage_sum(review_usage)
    all_paths = usage_sum([*episode_usage, *startup_usage, *review_usage])
    all_paths["main_agent_total_tokens"] = main.get("total_tokens")
    all_paths["review_total_tokens"] = reviews.get("total_tokens")
    return {"main_agent": main, "reviews": reviews, "all_paths": all_paths}


def snapshot(
    workspace: Path,
    *,
    stale_after: float = 600.0,
    lineages: tuple[str, ...] = (),
    research_limit: int | None = 50,
) -> dict:
    if research_limit is not None and (
        type(research_limit) is not int or research_limit <= 0
    ):
        raise ValueError("--research-limit must be a positive integer")
    runtime = runtime_snapshot(workspace, stale_after=stale_after)
    research = research_snapshot(workspace)
    tokens = token_snapshot(workspace)
    research_by_scope = {row["scope"]: row for row in research.get("scopes", [])}
    scheduler = runtime.get("scheduler_checkpoint") or {}
    scheduler_lineages = scheduler.get("lineages") if isinstance(scheduler.get("lineages"), dict) else {}
    runtime_scopes = runtime.get("scopes", [])
    campaign_runtime = next(
        (row for row in runtime_scopes if row.get("scope") == "campaign"), None
    )
    lineage_scopes = [row for row in runtime_scopes if row.get("trajectory") is not None]
    if not lineage_scopes:
        lineage_scopes = runtime_scopes
    selected_lineages = set(lineages)
    lineage_rows = []
    for row in lineage_scopes:
        scope = row["scope"]
        if selected_lineages and scope not in selected_lineages:
            continue
        board = research_by_scope.get(scope, {})
        directions = board.get("directions", [])
        experiments = board.get("experiments", [])
        visible_directions = directions if research_limit is None else directions[:research_limit]
        visible_experiments = experiments if research_limit is None else experiments[-research_limit:]
        lineage_rows.append(
            {
                "lineage": scope,
                "trajectory": row.get("trajectory"),
                "workspace": row.get("workspace"),
                "runtime": {
                    key: row.get(key)
                    for key in (
                        "phase",
                        "writer_active",
                        "liveness",
                        "current_episode",
                        "completed_episodes",
                        "latest_activity_at",
                        "stagnation_seconds",
                        "main_session_silence_seconds",
                        "active_session_count",
                        "active_sessions",
                        "active_supervisor_activity_count",
                        "supervisor_activity",
                    )
                },
                "runtime_configuration": row.get("runtime_configuration"),
                "research": {
                    "enabled_modules": board.get("enabled_modules"),
                    "revision": board.get("revision", 0),
                    "direction_counts": board.get("direction_counts", {}),
                    "directions": visible_directions,
                    "direction_total": len(directions),
                    "experiment_count": board.get("experiment_count", 0),
                    "experiments": visible_experiments,
                    "details_truncated": {
                        "directions": len(visible_directions) < len(directions),
                        "experiments": len(visible_experiments) < len(experiments),
                    },
                    "focus_direction_id": board.get("focus_direction_id"),
                    "active_direction_ids": board.get("active_direction_ids", []),
                    "local_direction_queue": board.get("local_direction_queue", []),
                    "shared_direction_queue": board.get("shared_direction_queue", []),
                    "last_shared_queue_sync": board.get("last_shared_queue_sync"),
                    "peer_git_refs": board.get("peer_git_refs"),
                    "claim_reviews": board.get(
                        "claim_reviews",
                        {"count": 0, "status_counts": {}, "quality_counts": {}},
                    ),
                },
                "tokens": _scope_tokens(tokens, scope),
                "search_lineage": scheduler_lineages.get(str(row.get("trajectory"))),
            }
        )
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": runtime.get("campaign_workspace"),
        "campaign_runtime": campaign_runtime,
        "lineages": lineage_rows,
        "campaign_configuration": {
            "search": scheduler.get("options"),
            "search_binding_digest": scheduler.get("binding_digest"),
            "token_budget": scheduler.get("token_budget"),
        },
    }


def render(value: dict) -> str:
    rows = []
    for item in value["lineages"]:
        runtime, research, tokens = item["runtime"], item["research"], item["tokens"]
        directions = research["direction_counts"]
        rows.append(
            [
                item["lineage"],
                runtime.get("liveness"),
                runtime.get("current_episode"),
                runtime.get("active_session_count"),
                (
                    runtime.get("supervisor_activity", {}).get("phase")
                    if isinstance(runtime.get("supervisor_activity"), dict)
                    else None
                ),
                directions.get("proposed", 0),
                directions.get("in_progress", 0),
                directions.get("deferred", 0) + directions.get("blocked", 0),
                directions.get("completed", 0) + directions.get("abandoned", 0),
                research.get("experiment_count"),
                tokens["main_agent"].get("total_tokens"),
                tokens["reviews"].get("total_tokens"),
                tokens["all_paths"].get("total_tokens"),
            ]
        )
    return "\n".join(
        [
            "AKA Light lineage status",
            "",
            table(
                [
                    "Lineage",
                    "Liveness",
                    "Episode",
                    "Sessions",
                    "Supervisor",
                    "Proposed",
                    "Active Dir",
                    "Paused Dir",
                    "Closed Dir",
                    "Experiments",
                    "Agent Tokens",
                    "Review Tokens",
                    "All Tokens",
                ],
                rows,
            ),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--stale-after", type=float, default=600.0)
    parser.add_argument("--lineage", action="append", default=[], help="lineage scope; repeatable")
    parser.add_argument("--research-limit", type=int, default=50, help="maximum Direction and Experiment detail rows per lineage")
    parser.add_argument("--all-research", action="store_true", help="return every Direction and Experiment row")
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(
        options.workspace,
        stale_after=options.stale_after,
        lineages=tuple(options.lineage),
        research_limit=None if options.all_research else options.research_limit,
    )
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-lineage-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
