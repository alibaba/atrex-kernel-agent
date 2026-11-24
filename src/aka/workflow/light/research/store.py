"""Append-only research state shared by Light Episodes.

Measurement Records and exact Git-identified Kernel source snapshots remain
authoritative for execution facts.  This store owns only research intent,
queue order, trajectory leases, and the compact projection used to start an
Episode.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aka.core.filesystem import durable_write_json
from aka.agent.execution.home import open_private_directory
from aka.agent.session_tail import read_regular_bytes
from aka.workflow.light.long_horizon.claim_review import canonical_review_quality

SCHEMA_VERSION = 1
MAX_EVENTS_BYTES = 64 * 1024 * 1024
MAX_EVENT_SEGMENT_BYTES = 8 * 1024 * 1024
MAX_EVENT_SEGMENTS = 4096
MAX_BOARD_DIRECTIONS = 12
MAX_BOARD_EXPERIMENTS = 8
RUNNABLE_STATUSES = {"proposed", "in_progress"}
PAUSED_STATUSES = {"deferred", "blocked"}
OPEN_STATUSES = RUNNABLE_STATUSES | PAUSED_STATUSES
CLOSED_STATUSES = {"completed", "abandoned"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _origin(value: Mapping[str, object] | None) -> dict[str, int]:
    value = value or {}
    result: dict[str, int] = {}
    for name in ("epoch", "trajectory", "episode"):
        item = value.get(name)
        # Episode/Epoch numbering may start at zero; trajectory numbering starts
        # at one. Preserve the origin for later audit segments.
        minimum = 1 if name == "trajectory" else 0
        if type(item) is int and item >= minimum:
            result[name] = item
    result.setdefault("trajectory", 1)
    return result


def _empty() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": 0,
        "directions": {},
        "experiments": {},
        "queue": [],
        "shared_queue": [],
        "lineages": {},
        "promotions": [],
        # Private idempotency index.  Keeping this in the compact projection
        # avoids rescanning the complete event history for every append.
    }


def _lineage_key(origin: Mapping[str, object] | None) -> str:
    return f"T{_origin(origin)['trajectory']}"


def _remove_retired_plan_progress(state: Mapping[str, object]) -> None:
    """Remove retired Runtime-owned fields from historical snapshots/events."""
    directions = state.get("directions")
    if isinstance(directions, Mapping):
        for direction in directions.values():
            if isinstance(direction, dict):
                if "owner_lineage" not in direction and isinstance(
                    direction.get("proposed_by"), str
                ):
                    direction["owner_lineage"] = direction["proposed_by"]
                direction.pop("proposed_by", None)
                direction.pop("completed_plan_steps", None)
                direction.pop("next_step", None)
                if "objective" not in direction and isinstance(
                    direction.get("hypothesis"), str
                ):
                    direction["objective"] = direction["hypothesis"]
                direction.pop("hypothesis", None)
                direction.pop("hypothesis_status", None)
                quality = canonical_review_quality(
                    direction.get("claim_review_quality", direction.get("review_quality"))
                )
                if quality is not None:
                    direction["claim_review_quality"] = quality
    experiments = state.get("experiments")
    if isinstance(experiments, Mapping):
        for experiment in experiments.values():
            if isinstance(experiment, dict):
                if "owner_lineage" not in experiment and isinstance(
                    experiment.get("recorded_by"), str
                ):
                    experiment["owner_lineage"] = experiment["recorded_by"]
                experiment.pop("recorded_by", None)
                quality = canonical_review_quality(
                    experiment.get(
                        "claim_review_quality", experiment.get("review_quality")
                    )
                )
                if quality is not None:
                    experiment["claim_review_quality"] = quality
    lineages = state.get("lineages")
    if isinstance(lineages, Mapping):
        for lineage in lineages.values():
            if isinstance(lineage, dict):
                # A passing full Evaluate is measurement evidence, not a winner.
                # Winners are derived separately from accepted submit-kernel commits.
                lineage.pop("lineage_best_kernel_id", None)
                lineage.pop("lineage_best_latency_us", None)


def _apply(state: dict[str, Any], event: Mapping[str, object]) -> None:
    kind = event["type"]
    payload = event.get("payload")
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    origin = event.get("origin")
    origin = dict(origin) if isinstance(origin, Mapping) else {}
    if kind == "direction.observed":
        direction = payload.get("direction")
        if not isinstance(direction, Mapping):
            raise ValueError("direction.observed requires a direction object")
        item = deepcopy(dict(direction))
        item.pop("completed_plan_steps", None)
        item.pop("next_step", None)
        if "objective" not in item and isinstance(item.get("hypothesis"), str):
            item["objective"] = item["hypothesis"]
        item.pop("hypothesis", None)
        item.pop("hypothesis_status", None)
        direction_id = item.get("direction_id")
        if not isinstance(direction_id, str) or not direction_id:
            raise ValueError("observed Direction has no ID")
        previous = state["directions"].get(direction_id, {})
        previous_revision = previous.get("revision", 0)
        item_revision = item.get("revision", 0)
        if (
            type(previous_revision) is not int
            or previous_revision < 0
            or type(item_revision) is not int
            or item_revision < 0
        ):
            raise ValueError("Direction revision is invalid")
        if item_revision < previous_revision:
            raise ValueError("Direction revision regressed")
        item.setdefault("access", previous.get("access", "local_writable"))
        owner_lineage = (
            item.get("owner_lineage")
            or item.get("proposed_by")
            or previous.get("owner_lineage")
            or previous.get("proposed_by")
        )
        if not isinstance(owner_lineage, str) or not owner_lineage:
            owner_lineage = _lineage_key(origin)
        item["owner_lineage"] = owner_lineage
        item.pop("proposed_by", None)
        previous_owner = previous.get("owner")
        if item.get("status") == "in_progress" and not item.get("owner") and previous_owner:
            item["owner"] = previous_owner
        item["origin"] = _origin(item.get("origin") if isinstance(item.get("origin"), Mapping) else origin)
        state["directions"][direction_id] = item
        if item.get("status") in OPEN_STATUSES:
            if item.get("access") == "peer_read_only":
                if direction_id in state["queue"]:
                    state["queue"].remove(direction_id)
            elif direction_id not in state["queue"]:
                state["queue"].append(direction_id)
        if item.get("status") in CLOSED_STATUSES:
            if direction_id in state["queue"]:
                state["queue"].remove(direction_id)
            if direction_id in state["shared_queue"]:
                state["shared_queue"].remove(direction_id)
        owner = item.get("owner") or previous_owner
        if isinstance(owner, str) and owner:
            active_ids = [
                candidate_id
                for candidate_id in state["queue"]
                if state["directions"].get(candidate_id, {}).get("status") == "in_progress"
                and state["directions"].get(candidate_id, {}).get("owner") == owner
            ]
            lineage = state["lineages"].setdefault(owner, {})
            lineage["active_direction_ids"] = active_ids
            # Retain the singular key for old snapshots and consumers that display one focus.
            lineage["active_direction_id"] = active_ids[0] if active_ids else None
            if lineage.get("focus_direction_id") not in active_ids:
                lineage["focus_direction_id"] = active_ids[0] if active_ids else None
    elif kind == "direction.progressed":
        # Historical compatibility only. Direction plans are Agent-owned:
        # Runtime does not track completed steps or derive the next action.
        pass
    elif kind == "direction.reordered":
        direction_id = payload.get("direction_id")
        if direction_id not in state["queue"]:
            raise ValueError("Only an open queued Direction can be reordered")
        before = payload.get("before_direction_id")
        after = payload.get("after_direction_id")
        if before is not None and after is not None:
            raise ValueError("Direction reorder accepts before or after, not both")
        queue = [item for item in state["queue"] if item != direction_id]
        anchor = before if before is not None else after
        if anchor is None:
            queue.append(direction_id)
        else:
            if anchor not in queue:
                raise ValueError("Direction reorder anchor is outside the open queue")
            offset = queue.index(anchor) + (1 if after is not None else 0)
            queue.insert(offset, direction_id)
        state["queue"] = queue
    elif kind == "direction.focused":
        direction_id = payload.get("direction_id")
        direction = state["directions"].get(direction_id)
        if (
            not isinstance(direction, Mapping)
            or direction.get("access") != "local_writable"
            or direction.get("status") != "in_progress"
        ):
            raise ValueError("Only a local in-progress Direction can be focused")
        owner = direction.get("owner") or _lineage_key(origin)
        lineage = state["lineages"].setdefault(owner, {})
        lineage["focus_direction_id"] = direction_id
    elif kind in {"queue.synchronized", "shared_queue.synchronized"}:
        requested = payload.get("direction_ids")
        if not isinstance(requested, list) or len(set(requested)) != len(requested):
            raise ValueError("Shared Direction queue must contain unique IDs")
        unknown = [item for item in requested if item not in state["directions"]]
        if unknown:
            raise ValueError("Shared Direction queue references an unknown Direction")
        open_ids = [
            item for item in requested
            if state["directions"][item].get("status") in OPEN_STATUSES
            and state["directions"][item].get("access") == "peer_read_only"
        ]
        for direction_id, direction in state["directions"].items():
            if (
                direction_id not in open_ids
                and direction.get("status") in OPEN_STATUSES
                and direction.get("access") == "peer_read_only"
            ):
                open_ids.append(direction_id)
        state["shared_queue"] = open_ids
    elif kind == "experiment.observed":
        experiment = payload.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ValueError("experiment.observed requires an experiment object")
        item = deepcopy(dict(experiment))
        experiment_id = item.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("observed Experiment has no ID")
        item["origin"] = _origin(item.get("origin") if isinstance(item.get("origin"), Mapping) else origin)
        previous = state["experiments"].get(experiment_id)
        if previous is not None:
            derived = {"origin", "owner_lineage", "recorded_by", "access"}
            previous_value = {
                key: value for key, value in previous.items() if key not in derived
            }
            item_value = {key: value for key, value in item.items() if key not in derived}
            previous_revision = previous.get("revision", 1)
            item_revision = item.get("revision", 1)
            if (
                type(previous_revision) is not int
                or previous_revision < 1
                or type(item_revision) is not int
                or item_revision < 1
            ):
                raise ValueError("Experiment revision is invalid")
            if item_revision < previous_revision:
                item = previous
            elif item_revision == previous_revision:
                if previous_value != item_value:
                    raise ValueError(
                        "Experiment revision was reused for different content"
                    )
                # Source lineage and local access are derived independently in
                # each trajectory and may be added to an existing revision by
                # a newer Runtime projection.
                for field in ("owner_lineage", "recorded_by", "access"):
                    if field not in item and field in previous:
                        item[field] = previous[field]
        owner_lineage = (
            item.get("owner_lineage")
            or item.get("recorded_by")
            or (previous or {}).get("owner_lineage")
            or (previous or {}).get("recorded_by")
            or _lineage_key(origin)
        )
        item["owner_lineage"] = owner_lineage
        item.pop("recorded_by", None)
        state["experiments"][experiment_id] = item
    elif kind == "lineage.checkpointed":
        lineage = _lineage_key(origin)
        current = state["lineages"].setdefault(lineage, {})
        for key in (
            "working_kernel_id", "wip_checkpoint", "last_correct_kernel_id",
            "active_direction_id", "active_direction_ids", "session_id",
            "gateway_record_ids",
        ):
            if key in payload:
                current[key] = (list(dict.fromkeys([*current.get(key, []), *payload[key]]))
                                if key == "gateway_record_ids" else deepcopy(payload[key]))
    elif kind == "kernel.adopted":
        lineage = _lineage_key(origin)
        current = state["lineages"].setdefault(lineage, {})
        # Replay-only compatibility for logs created before the adopt-kernel
        # operation was retired. New Agents use imported Git refs directly.
        # Older events may contain one ID or distinct source/local IDs.
        source_kernel_id = payload.get(
            "source_kernel_commit_id", payload.get("kernel_id")
        )
        local_kernel_id = payload.get(
            "local_kernel_commit_id", payload.get("kernel_id")
        )
        current["working_kernel_id"] = local_kernel_id
        source = deepcopy(payload.get("source"))
        source = source if isinstance(source, dict) else {}
        source["kernel_commit_id"] = source_kernel_id
        current["adopted_from"] = source
    elif kind in {"promotion.accepted", "promotion.rejected"}:
        state["promotions"].append({
            "type": kind,
            "payload": deepcopy(payload),
            "origin": _origin(origin),
            "sequence": event["sequence"],
        })
        state["promotions"] = state["promotions"][-64:]
    else:
        raise ValueError(f"unsupported research event type: {kind}")
    state["sequence"] = event["sequence"]


class ResearchStore:
    """Crash-recoverable event log with a rebuildable compact projection."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.snapshot_path = self.root / "snapshot.json"
        self.lock_path = self.root / "lock"
        with open_private_directory(self.root):
            pass

    _SEGMENT = re.compile(r"events-(\d{20})\.jsonl")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with open_private_directory(self.root) as directory:
            try:
                fd = os.open(
                    "lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600, dir_fd=directory,
                )
            except FileExistsError:
                # Concurrent first readers must not both create the lock.
                # Some filesystems return ENOENT for racing O_CREAT opens.
                fd = os.open("lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _segment_paths(self) -> list[Path]:
        paths = []
        for path in self.root.glob("events-*.jsonl"):
            match = self._SEGMENT.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Invalid Research event segment name: {path.name}")
            paths.append((int(match.group(1)), path))
        paths.sort()
        if len(paths) > MAX_EVENT_SEGMENTS:
            raise ValueError("Research event history exceeds its segment limit")
        return [path for _sequence, path in paths]

    @staticmethod
    def _parse_events(raw: bytes, rows: list[dict[str, Any]]) -> None:
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Research event log line {len(rows) + 1} is invalid"
                ) from error
            if not isinstance(value, dict) or value.get("sequence") != len(rows) + 1:
                raise ValueError("Research event sequence is not contiguous")
            rows.append(value)

    def _events(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        paths = [*self._segment_paths(), self.events_path]
        for path in paths:
            try:
                raw = read_regular_bytes(path, limit=MAX_EVENTS_BYTES + 1)
            except FileNotFoundError:
                continue
            if len(raw) > MAX_EVENTS_BYTES:
                raise ValueError(f"Research event segment exceeds 64 MiB: {path.name}")
            self._parse_events(raw, rows)
            match = self._SEGMENT.fullmatch(path.name)
            if match is not None and rows and rows[-1]["sequence"] != int(match.group(1)):
                raise ValueError("Research event segment boundary is invalid")
        return rows

    def _active_size(self) -> int:
        try:
            value = self.events_path.lstat()
        except FileNotFoundError:
            return 0
        if not stat.S_ISREG(value.st_mode):
            raise ValueError("Research active event log must be a regular file")
        return value.st_size

    def _snapshot_payload(self, state: Mapping[str, object], active_offset: int) -> dict:
        return {
            "snapshot_schema_version": 1,
            "active_offset": active_offset,
            "segments": [path.name for path in self._segment_paths()],
            "state": deepcopy(dict(state)),
        }

    def _write_snapshot(self, state: Mapping[str, object], active_offset: int) -> None:
        durable_write_json(
            self.snapshot_path,
            self._snapshot_payload(state, active_offset),
        )

    def _snapshot_state(self) -> tuple[dict[str, Any], int] | None:
        """Load a compact checkpoint and replay only the active-log tail.

        Segment names and the active byte offset make rotation/recovery
        fail-closed: an interrupted rotation falls back to a complete rebuild.
        """
        try:
            raw = read_regular_bytes(self.snapshot_path, limit=MAX_EVENTS_BYTES + 1)
            value = json.loads(raw)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if (
            not isinstance(value, dict)
            or value.get("snapshot_schema_version") != 1
            or not isinstance(value.get("state"), dict)
            or type(value.get("active_offset")) is not int
            or value["active_offset"] < 0
            or value.get("segments") != [path.name for path in self._segment_paths()]
        ):
            return None
        state = deepcopy(value["state"])
        if state.get("schema_version") != SCHEMA_VERSION or type(state.get("sequence")) is not int:
            return None
        _remove_retired_plan_progress(state)
        state.setdefault("shared_queue", [])
        active_size = self._active_size()
        offset = value["active_offset"]
        if offset > active_size:
            return None
        try:
            descriptor = os.open(self.events_path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if offset == 0:
                return state, 0
            return None
        try:
            os.lseek(descriptor, offset, os.SEEK_SET)
            chunks = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_EVENTS_BYTES:
                    return None
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        # Tail sequences are global, whereas _parse_events starts at one.
        for line in b"".join(chunks).splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None
            if (
                not isinstance(event, dict)
                or event.get("sequence") != state["sequence"] + 1
            ):
                return None
            try:
                _apply(state, event)
            except (KeyError, TypeError, ValueError):
                return None
        return state, active_size

    def _current_state(self) -> tuple[dict[str, Any], int]:
        cached = self._snapshot_state()
        if cached is not None:
            return cached
        state = self.rebuild()
        active_size = self._active_size()
        self._write_snapshot(state, active_size)
        return state, active_size

    def _rotate_active(self, sequence: int) -> int:
        active_size = self._active_size()
        if active_size == 0:
            return 0
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("Cannot rotate Research events without a valid sequence")
        if len(self._segment_paths()) >= MAX_EVENT_SEGMENTS:
            raise ValueError("Research event history exceeds its segment limit")
        target = self.root / f"events-{sequence:020d}.jsonl"
        if target.exists():
            raise ValueError("Research event segment already exists")
        os.replace(self.events_path, target)
        # Snapshot metadata is written by the caller after the next append. If
        # the process stops here, segment-list validation forces a full rebuild.
        return 0

    def rebuild(self) -> dict[str, Any]:
        state = _empty()
        for event in self._events():
            _apply(state, event)
        return state

    def read(self) -> dict[str, Any]:
        with self._locked():
            state, active_size = self._current_state()
            self._write_snapshot(state, active_size)
            return deepcopy(state)

    def events(self) -> list[dict[str, Any]]:
        with self._locked():
            return deepcopy(self._events())

    @staticmethod
    def _through_episode(
        events: list[dict[str, Any]], episode: int
    ) -> list[dict[str, Any]]:
        if type(episode) is not int or episode < 0:
            raise ValueError("Research Episode boundary must be a non-negative integer")
        result = []
        for event in events:
            origin = event.get("origin")
            event_episode = origin.get("episode") if isinstance(origin, Mapping) else None
            if type(event_episode) is int and event_episode > episode:
                continue
            result.append(event)
        return result

    def read_through_episode(self, episode: int) -> dict[str, Any]:
        """Rebuild a terminal projection without observing a newer live Episode."""
        with self._locked():
            state = _empty()
            for event in self._through_episode(self._events(), episode):
                _apply(state, event)
            return deepcopy(state)

    def read_through_sequence(self, sequence: int) -> dict[str, Any]:
        """Rebuild the exact catalog projection visible at one watermark."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Research sequence watermark must be non-negative")
        with self._locked():
            state = _empty()
            for event in self._events():
                if event["sequence"] > sequence:
                    break
                _apply(state, event)
            return deepcopy(state)

    def events_through_sequence(self, sequence: int) -> list[dict[str, Any]]:
        """Return immutable events through an inclusive catalog watermark."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Research sequence watermark must be non-negative")
        with self._locked():
            return deepcopy([
                event for event in self._events() if event["sequence"] <= sequence
            ])

    def events_through_episode(self, episode: int) -> list[dict[str, Any]]:
        """Return only events eligible for a terminal Episode publication."""
        with self._locked():
            return deepcopy(self._through_episode(self._events(), episode))

    def append(
        self,
        event_type: str,
        payload: Mapping[str, object],
        *,
        origin: Mapping[str, object] | None = None,
        skip_if_projection_unchanged: str | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            state, active_size = self._current_state()
            event = {
                "sequence": state["sequence"] + 1,
                "type": event_type,
                "recorded_at": _now(),
                "origin": _origin(origin),
                "payload": deepcopy(dict(payload)),
            }
            projected = deepcopy(state)
            _apply(projected, event)
            if (
                skip_if_projection_unchanged is not None
                and projected.get(skip_if_projection_unchanged)
                == state.get(skip_if_projection_unchanged)
            ):
                return deepcopy(state)
            state = projected
            encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()
            if len(encoded) > MAX_EVENT_SEGMENT_BYTES:
                raise ValueError("Research event exceeds its bounded segment size")
            if active_size + len(encoded) > MAX_EVENT_SEGMENT_BYTES:
                active_size = self._rotate_active(event["sequence"] - 1)
            descriptor = os.open(
                self.events_path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("Research active event log must be a regular file")
                stream = os.fdopen(descriptor, "ab", closefd=False)
                try:
                    stream.write(encoded)
                    stream.flush()
                finally:
                    stream.close()
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            active_size += len(encoded)
            self._write_snapshot(state, active_size)
            return deepcopy(state)

    def observe_direction(
        self, direction: Mapping[str, object], *, origin: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        direction_id = direction.get("direction_id")
        if not isinstance(direction_id, str):
            raise ValueError("Direction has no ID")
        return self.append(
            "direction.observed",
            {"direction": direction},
            origin=origin,
            # Direction state is mutable: the same in-progress projection may
            # legitimately recur after a defer/reopen cycle.  Deduplicate only
            # against the current projection, never against historical events.
            skip_if_projection_unchanged="directions",
        )

    def observe_experiment(
        self, experiment: Mapping[str, object], *, origin: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str):
            raise ValueError("Experiment has no ID")
        return self.append(
            "experiment.observed",
            {"experiment": experiment},
            origin=origin,
            skip_if_projection_unchanged="experiments",
        )

    def prioritize(
        self,
        direction_id: str,
        *,
        before_direction_id: str | None = None,
        after_direction_id: str | None = None,
        reason: str,
        origin: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Direction reorder requires a non-empty reason")
        return self.append(
            "direction.reordered",
            {
                "direction_id": direction_id,
                "before_direction_id": before_direction_id,
                "after_direction_id": after_direction_id,
                "reason": reason.strip(),
            },
            origin=origin,
        )

    def checkpoint_lineage(
        self, values: Mapping[str, object], *, origin: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        return self.append(
            "lineage.checkpointed",
            values,
            origin=origin,
            skip_if_projection_unchanged="lineages",
        )

    def synchronize_shared_queue(
        self,
        direction_ids: list[str],
        *,
        origin: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        return self.append(
            "shared_queue.synchronized",
            {"direction_ids": direction_ids},
            origin=origin,
            skip_if_projection_unchanged="shared_queue",
        )

    def focus_direction(
        self,
        direction_id: str,
        *,
        origin: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        return self.append(
            "direction.focused",
            {"direction_id": direction_id},
            origin=origin,
        )

    def board(
        self,
        *,
        trajectory: int = 1,
        pareto_frontier: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        state = self.read()
        current_lineage = f"T{trajectory}"
        current_state = state.get("lineages", {}).get(current_lineage, {})
        active_ids = {
            value
            for value in current_state.get("active_direction_ids", [])
            if isinstance(value, str)
        }

        def projected_queue(
            direction_ids, *, rank_field, access, statuses, pinned=frozenset()
        ):
            candidates = []
            for rank, direction_id in enumerate(direction_ids, 1):
                item = state["directions"].get(direction_id)
                if not isinstance(item, dict) or item.get("access") != access:
                    continue
                owner = item.get("owner")
                status = item.get("status")
                if status not in statuses:
                    continue
                candidates.append({
                    rank_field: rank,
                    "direction_id": direction_id,
                    "name": item.get("name"),
                    "change_scope": item.get("change_scope", "incremental"),
                    "owner_lineage": item.get("owner_lineage"),
                    "access": item.get("access"),
                    "status": (
                        f"in_progress({owner})"
                        if status == "in_progress" and owner else status
                    ),
                    "objective": item.get("objective"),
                    "applicability": item.get("applicability"),
                    "expected_benefit": item.get("expected_benefit"),
                    "analysis": item.get("analysis"),
                    "reopen_condition": item.get("reopen_condition"),
                    "claim_review_quality": canonical_review_quality(
                        item.get("claim_review_quality", item.get("review_quality"))
                    ),
                })
            rows = candidates[:MAX_BOARD_DIRECTIONS]
            visible_ids = {item.get("direction_id") for item in rows}
            rows.extend(
                item
                for item in candidates
                if item.get("direction_id") in pinned
                and item.get("direction_id") not in visible_ids
            )
            rows.sort(key=lambda item: item.get(rank_field, 1 << 30))
            return rows, len(candidates)

        queue, queue_total = projected_queue(
            state["queue"], rank_field="local_rank", access="local_writable",
            statuses=RUNNABLE_STATUSES, pinned=active_ids,
        )
        paused, paused_total = projected_queue(
            state["queue"], rank_field="local_rank", access="local_writable",
            statuses=PAUSED_STATUSES,
        )
        shared_queue, shared_queue_total = projected_queue(
            state["shared_queue"],
            rank_field="shared_rank",
            access="peer_read_only",
            statuses=RUNNABLE_STATUSES,
        )
        shared_paused, shared_paused_total = projected_queue(
            state["shared_queue"],
            rank_field="shared_rank",
            access="peer_read_only",
            statuses=PAUSED_STATUSES,
        )
        experiments = list(state["experiments"].values())[-MAX_BOARD_EXPERIMENTS:]
        recent = [
            {
                key: item.get(key)
                for key in (
                    "experiment_id", "direction_id", "name", "owner_lineage", "access",
                    "implementation", "expectations", "interpretation", "next_step",
                )
                if item.get(key) is not None
            }
            for item in experiments
        ]
        all_closed = [
            {
                key: item.get(key)
                for key in (
                    "direction_id", "name", "owner_lineage", "access", "status", "objective",
                    "applicability", "expected_benefit", "analysis", "reopen_condition",
                )
                if item.get(key) is not None
            }
            for item in state["directions"].values()
            if item.get("status") in CLOSED_STATUSES
        ]
        closed = all_closed[-MAX_BOARD_DIRECTIONS:]
        promotions = state["promotions"]
        incumbent = next(
            (
                item["payload"]
                for item in reversed(promotions)
                if item["type"] == "promotion.accepted"
            ),
            None,
        )
        public_lineages = {}
        for name, value in state["lineages"].items():
            item = deepcopy(value)
            # Ignore retired snapshot fields even when a pre-migration state was
            # loaded without replaying its complete event history.
            item.pop("lineage_best_kernel_id", None)
            item.pop("lineage_best_latency_us", None)
            for old, new in (
                ("working_kernel_id", "working_kernel_commit_id"),
                ("last_correct_kernel_id", "last_correct_kernel_commit_id"),
            ):
                if old in item:
                    item[new] = item.pop(old)
            public_lineages[name] = item
        public_incumbent = deepcopy(incumbent)
        if isinstance(public_incumbent, dict) and "kernel_id" in public_incumbent:
            public_incumbent["kernel_commit_id"] = public_incumbent.pop("kernel_id")
        frontier = deepcopy(dict(pareto_frontier or {
            "best_by_shape": {},
            "shape_count": 0,
            "winner_kernel_commit_ids": [],
            "evaluation_count": 0,
            "best_full_evaluation": None,
        }))
        return {
            "schema_version": "atrex.research_board.v2",
            "revision": state["sequence"],
            "incumbent": public_incumbent,
            "lineages": public_lineages,
            "current_lineage": current_lineage,
            "pareto_frontier": frontier,
            "open_queue": queue,
            "open_queue_total": queue_total,
            "paused_directions": paused,
            "paused_directions_total": paused_total,
            "shared_direction_pool": shared_queue,
            "shared_direction_pool_total": shared_queue_total,
            "shared_paused_directions": shared_paused,
            "shared_paused_directions_total": shared_paused_total,
            "recent_experiments": recent,
            "closed_directions": closed,
            "closed_directions_total": len(all_closed),
        }
