#!/usr/bin/env python3
"""Write compact, consumer-owned Wiki query evidence.

The Wiki remains usable without telemetry.  A consumer enables immutable event
files by setting ``ATREX_WIKI_PROFILE_ROOT`` to a directory it owns.  Events
contain query scope and returned record identities, never returned payloads or
coding-agent transcripts.  ``run.json`` identifies the lifetime of that profile
root; each event independently records the current ``ATREX_WIKI_TASK_ID`` so a
resumed workspace can be attributed to a new task without changing its run id.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_ROOT_ENV = "ATREX_WIKI_PROFILE_ROOT"
TASK_ID_ENV = "ATREX_WIKI_TASK_ID"
EVENT_SCHEMA = "atrex-wiki-query-event-v2"
RUN_SCHEMA = "atrex-wiki-profile-run-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _publish_create_only(temporary: Path, target: Path) -> bool:
    """Publish a prepared file without requiring hardlink support.

    Hardlinks preserve create-only behavior where available.  Some workspace
    filesystems do not support them, so fall back to an exclusive create.  The
    fallback may be observed while it is being copied, so callers re-read the
    single run identity after publication.
    """
    try:
        os.link(temporary, target)
        return True
    except FileExistsError:
        return False
    except OSError:
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as output:
            output.write(temporary.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        return True


def _run_identity(root: Path) -> dict[str, Any]:
    path = root / "run.json"
    existing = _read_json(path)
    if existing and existing.get("schema_version") == RUN_SCHEMA:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": RUN_SCHEMA,
        "run_id": str(uuid.uuid4()),
        "created_at": utc_now(),
    }
    task_id = os.environ.get(TASK_ID_ENV, "").strip()
    if task_id:
        value["task_id"] = task_id
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}-{uuid.uuid4()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # A process can die while the no-hardlink O_EXCL fallback is copying
        # the first identity.  Repair only a file that still fails validation;
        # a concurrently completed valid identity always wins.
        if path.exists():
            existing = _read_json(path)
            if existing and existing.get("schema_version") == RUN_SCHEMA:
                return existing
            temporary.replace(path)
            published = _read_json(path)
            if published and published.get("schema_version") == RUN_SCHEMA:
                return published
            raise RuntimeError(f"Wiki trace run identity repair failed: {path}")
        if _publish_create_only(temporary, path):
            published = _read_json(path)
            return published or value
        existing = _read_json(path)
        if existing and existing.get("schema_version") == RUN_SCHEMA:
            return existing
        raise RuntimeError(f"Wiki trace run identity is unreadable: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def store_revision(root: Path) -> str:
    """Hash every available index without opening individual records."""
    candidates = (
        root / "search_index" / "index.json",
        root / "kernel_wiki" / "records" / "index.json",
    )
    digest = hashlib.sha256()
    found = False
    for path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        found = True
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return "sha256:" + digest.hexdigest() if found else "unavailable"


def write_query_event(
    *,
    query_id: str,
    request: str,
    status: str,
    bridge_intent: dict[str, Any] | None,
    normalized_intents: list[dict[str, Any]],
    returned_records: list[dict[str, Any]],
    wiki_stores: list[dict[str, Any]],
    retrieval_failures: list[dict[str, str]],
    metric: dict[str, Any],
) -> Path | None:
    raw_root = os.environ.get(PROFILE_ROOT_ENV, "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).expanduser().resolve()
    try:
        identity = _run_identity(root)
        current_task_id = os.environ.get(TASK_ID_ENV, "").strip()
        timestamp = utc_now()
        target_dir = root / "raw" / "query_events" / timestamp[:10]
        target_dir.mkdir(parents=True, exist_ok=True)
        event_id = str(uuid.uuid4())
        # Keep query timing/token counters at the top level.  The offline
        # profile builder already consumes this shape, so the producer does
        # not need a second adapter or a duplicate metrics file.
        event = {
            **metric,
            "schema_version": EVENT_SCHEMA,
            "event_type": "wiki_query",
            "event_id": event_id,
            "query_id": query_id,
            "run_id": identity["run_id"],
            "task_id": current_task_id or identity.get("task_id"),
            "entrypoint": "query_nl",
            "timestamp": timestamp,
            "status": status,
            "request": request,
            "bridge_intent": bridge_intent,
            "normalized_intents": normalized_intents,
            "normalized_keywords": normalized_keywords(normalized_intents),
            "result": {
                "kind": "matches" if returned_records else "empty",
                "served": len(returned_records),
            },
            "wiki_stores": wiki_stores,
            "retrieval_failures": retrieval_failures,
            "returned_records": returned_records,
        }
        target = target_dir / f"{event_id}.json"
        temporary = target.with_suffix(
            f".json.tmp-{os.getpid()}-{uuid.uuid4()}"
        )
        try:
            temporary.write_text(
                json.dumps(event, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            if not _publish_create_only(temporary, target):
                raise FileExistsError(f"Wiki trace event already exists: {target}")
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except Exception as exc:
        # Telemetry must never change a query result.
        import sys

        print(f"WARNING Wiki query trace could not be written: {exc}", file=sys.stderr)
        return None


def normalized_keywords(intents: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for intent in intents:
        for key in (
            "arch", "product", "vendor", "dsl", "operator", "family",
            "symptom",
        ):
            value = intent.get(key)
            if isinstance(value, str) and value.strip():
                values.add(value.strip().casefold())
        for key in ("terms", "operator_terms", "component_terms", "intents"):
            rows = intent.get(key)
            if isinstance(rows, list):
                values.update(
                    str(value).strip().casefold()
                    for value in rows
                    if str(value).strip()
                )
    return sorted(values)
