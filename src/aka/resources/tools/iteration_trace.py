#!/usr/bin/env python3
"""Append explicit, metadata-only events to the current local iteration trace."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PHASES = {
    "profile",
    "research",
    "planning",
    "implementation",
    "correctness",
    "benchmark",
    "recording",
}
SOURCE_KINDS = {"gpu_wiki", "reference_projects", "workspace", "public_web", "unknown"}


def _safe_ref(value: str, *, source_kind: str) -> str:
    value = value.strip().replace("\\", "/")
    if source_kind == "public_web":
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("public source reference must be a credential-free HTTP URL")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:500]
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("source reference must be a safe relative reference")
    return value[:500]


def append_event(event: str, **fields: object) -> bool:
    trace = os.environ.get("ATREX_TELEMETRY_TRACE")
    if not trace:
        return False
    payload = {
        "schema_version": "atrex_iteration_event_v1",
        "campaign_id": os.environ.get("ATREX_TELEMETRY_CAMPAIGN_ID", "campaign"),
        "iteration_id": os.environ.get("ATREX_TELEMETRY_ITERATION_ID", "unknown"),
        "attempt_id": os.environ.get("ATREX_TELEMETRY_ATTEMPT_ID", "attempt"),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "source": "agent_marker",
        "measurement": "explicit",
        **fields,
    }
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase-start", "phase-end"):
        command = sub.add_parser(name)
        command.add_argument("phase", choices=sorted(PHASES))
    source = sub.add_parser("source-read")
    source.add_argument("source_kind", choices=sorted(SOURCE_KINDS))
    source.add_argument("reference")
    args = parser.parse_args(argv)
    if args.command.startswith("phase-"):
        action = "start" if args.command == "phase-start" else "end"
        event = "phase_started" if action == "start" else "phase_completed"
        marker_id = uuid.uuid4().hex
        if append_event(
            event,
            phase=args.phase,
            action=action,
            marker_id=marker_id,
        ):
            receipt = {
                "schema": "atrex.iteration_trace.v1",
                "kind": "phase_marker",
                "action": action,
                "phase": args.phase,
                "marker_id": marker_id,
            }
            print(
                "ATREX_TRACE_EVENT="
                + json.dumps(receipt, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
    else:
        append_event(
            "source_read",
            source_kind=args.source_kind,
            reference=_safe_ref(args.reference, source_kind=args.source_kind),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
