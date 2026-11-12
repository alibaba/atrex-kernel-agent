#!/usr/bin/env python3
"""Report GPU Wiki queries, results, latency, failures and Journal adoption."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments,
    discover_scopes,
    emit,
    main_error,
    observed_at,
    read_json,
    table,
)


def _first_text(value: object, keys: set[str]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str) and item.strip():
                return item.strip()
        for item in value.values():
            if found := _first_text(item, keys):
                return found
    elif isinstance(value, list):
        for item in value:
            if found := _first_text(item, keys):
                return found
    return None


def _knowledge_ids(value: object) -> set[str]:
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"wiki_id", "knowledge_id", "record_id"} and isinstance(item, str) and item:
                result.add(item)
            elif key == "records" and isinstance(item, dict):
                result.update(str(name) for name in item)
            result.update(_knowledge_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_knowledge_ids(item))
    return result


def _journal_adoption(scope) -> tuple[set[str], set[str], list[dict]]:
    query_ids, wiki_ids, usages = set(), set(), []
    for path in sorted((scope.workspace / ".atrex_long_horizon" / "episodes").glob("e*/episode_runtime/journal.json")):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        for experiment in value.get("experiments") or []:
            if not isinstance(experiment, dict):
                continue
            for item in experiment.get("wiki_query_ids") or []:
                if isinstance(item, str):
                    query_ids.add(item)
            for item in experiment.get("wiki_usage") or []:
                if not isinstance(item, dict):
                    continue
                usages.append({"scope": scope.label, "experiment_id": experiment.get("experiment_id"), **item})
                if isinstance(item.get("query_id"), str):
                    query_ids.add(item["query_id"])
                if isinstance(item.get("wiki_id"), str):
                    wiki_ids.add(item["wiki_id"])
    return query_ids, wiki_ids, usages


def snapshot(workspace: Path) -> dict:
    queries = []
    usages = []
    adopted_query_ids = set()
    adopted_wiki_ids = set()
    status_counts = Counter()
    for scope in discover_scopes(workspace):
        scope_query_ids, scope_wiki_ids, scope_usages = _journal_adoption(scope)
        adopted_query_ids.update(scope_query_ids)
        adopted_wiki_ids.update(scope_wiki_ids)
        usages.extend(scope_usages)
        events_by_query: dict[str, dict] = {}
        for path in sorted((scope.workspace / ".gpu_wiki_profile" / "raw" / "query_events").glob("*/*.json")):
            value = read_json(path, default={})
            if not isinstance(value, dict):
                continue
            query_id = value.get("query_id")
            if isinstance(query_id, str):
                events_by_query[query_id] = value
        for path in sorted((scope.audit_root / "wiki" / "requests").glob("*.json")):
            receipt = read_json(path, default={})
            if not isinstance(receipt, dict):
                continue
            query_id = receipt.get("query_id")
            event = events_by_query.get(query_id, {}) if isinstance(query_id, str) else {}
            status = str(receipt.get("status") or "unknown")
            status_counts[status] += 1
            knowledge = sorted(_knowledge_ids(event))
            queries.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "request_id": receipt.get("request_id") or path.stem,
                    "query_id": query_id,
                    "tool": receipt.get("tool"),
                    "query": _first_text(event, {"query", "request", "question", "prompt"}),
                    "status": status,
                    "cache_hit": receipt.get("cache_hit") is True,
                    "queue_latency_ms": receipt.get("queue_latency_ms"),
                    "bridge_latency_ms": receipt.get("bridge_latency_ms"),
                    "retrieval_latency_ms": receipt.get("retrieval_latency_ms"),
                    "total_latency_ms": receipt.get("total_latency_ms"),
                    "knowledge_ids": knowledge,
                    "adopted": bool(
                        (isinstance(query_id, str) and query_id in adopted_query_ids)
                        or set(knowledge) & adopted_wiki_ids
                    ),
                    "provider_error_code": receipt.get("provider_error_code"),
                    "provider_error_message": receipt.get("provider_error_message"),
                    "finished_at": receipt.get("finished_at"),
                }
            )
        # Older runs can have query events but no Runtime receipt.
        known = {row["query_id"] for row in queries if row["scope"] == scope.label}
        for query_id, event in sorted(events_by_query.items()):
            if query_id in known:
                continue
            knowledge = sorted(_knowledge_ids(event))
            queries.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "request_id": None,
                    "query_id": query_id,
                    "tool": event.get("tool"),
                    "query": _first_text(event, {"query", "request", "question", "prompt"}),
                    "status": event.get("status", "event_only"),
                    "cache_hit": None,
                    "queue_latency_ms": None,
                    "bridge_latency_ms": None,
                    "retrieval_latency_ms": None,
                    "total_latency_ms": None,
                    "knowledge_ids": knowledge,
                    "adopted": query_id in adopted_query_ids or bool(set(knowledge) & adopted_wiki_ids),
                    "provider_error_code": None,
                    "provider_error_message": None,
                    "finished_at": event.get("finished_at") or event.get("timestamp"),
                }
            )
    failures = sum(row["status"] not in {"completed", "reused", "event_only"} for row in queries)
    no_results = sum(not row["knowledge_ids"] and row["status"] in {"completed", "reused", "event_only"} for row in queries)
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "summary": {
            "query_count": len(queries),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_count": failures,
            "no_result_count": no_results,
            "adopted_query_count": sum(row["adopted"] for row in queries),
        },
        "queries": queries,
        "journal_usage": usages,
    }


def render(value: dict) -> str:
    lines = ["AKA Light GPU Wiki", ""]
    lines.append(table(
        ["Scope", "Tool", "Status", "Cache", "Total(ms)", "Knowledge", "Adopted", "Query"],
        [[row["scope"], row["tool"], row["status"], row["cache_hit"], row["total_latency_ms"],
          len(row["knowledge_ids"]), row["adopted"], (row["query"] or "")[:80]]
         for row in value["queries"]],
    ))
    lines += ["", f"Summary: {value['summary']}"]
    return "\n".join(lines)


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
    return main_error("light-wiki-usage", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
