#!/usr/bin/env python3
"""Report accepted and observed-best latency for a running Light Campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable


# Allow direct execution from a source checkout without installing AKA.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aka.workflow.legacy.export import RetainedWorkflowExport  # noqa: E402
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.evaluation.tool_results import result_from_response  # noqa: E402
from observability.measurement_reader import (  # noqa: E402
    MeasurementStore,
    RECORD_ID,
)
from aka.workflow.light.support.paths import supervisor_campaign_root  # noqa: E402
from aka.workflow.light.search import read_state  # noqa: E402
from aka.task.source.formats import from_git  # noqa: E402


def finite_latency(value: object) -> float | None:
    """Return a finite positive latency without treating bool as a number."""
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _search_accepted_candidates(state: dict) -> Iterable[dict]:
    """Yield published and already-terminal accepted search candidates."""
    seed = state.get("seed")
    if isinstance(seed, dict):
        yield seed

    active = state.get("active")
    if not isinstance(active, dict):
        return
    jobs = active.get("jobs")
    if not isinstance(jobs, list):
        return
    for job in jobs:
        if not isinstance(job, dict) or job.get("status") != "completed":
            continue
        result = job.get("result")
        if not isinstance(result, dict):
            continue
        retained = result.get("retained_kernel")
        if isinstance(retained, dict):
            # retained_kernel is populated only after the Episode's acceptance
            # gate passes. It can exist before the rest of the Epoch finishes.
            yield retained


def _document(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _episode_attempt(workspace: Path, episode: int) -> dict | None:
    root = workspace / ".atrex_long_horizon"
    for path in (
        root / "episodes" / f"e{episode:04d}" / "attempt.json",
        root / "episodes" / f"e{episode:08d}" / "attempt.json",
    ):
        value = _document(path)
        if isinstance(value, dict) and value.get("episode") == episode:
            return value
    state = _document(root / "state.json") or {}
    for value in state.get("attempts") or []:
        if isinstance(value, dict) and value.get("episode") == episode:
            return value
    return None


def _accepted_evidence(owner: Path, candidate: dict) -> dict:
    """Resolve an accepted Candidate to its exact terminal verification Record."""
    unavailable = {
        "gateway_record_id": None,
        "verification_kernel_id": None,
        "verification_operation": None,
    }
    episode = candidate.get("episode")
    if type(episode) is not int or episode <= 0:
        return {**unavailable, "verification_evidence": "bootstrap_has_no_episode_verification"}
    attempt = _episode_attempt(owner, episode)
    if attempt is None:
        return {**unavailable, "verification_evidence": "attempt_not_available"}
    if attempt.get("accepted") is not True:
        return {**unavailable, "verification_evidence": "attempt_not_accepted"}
    version = candidate.get("version")
    if type(version) is int and attempt.get("version") != version:
        return {**unavailable, "verification_evidence": "attempt_version_mismatch"}
    verification = attempt.get("verification")
    if not isinstance(verification, dict) or verification.get("gate") != "PASS":
        return {**unavailable, "verification_evidence": "passing_verification_not_available"}
    record_id = verification.get("gateway_record_id")
    if not isinstance(record_id, str) or RECORD_ID.fullmatch(record_id) is None:
        legacy = verification.get("artifact")
        record_id = legacy if isinstance(legacy, str) and RECORD_ID.fullmatch(legacy) else None
    if record_id is None:
        return {**unavailable, "verification_evidence": "gateway_record_id_not_available"}
    if not MeasurementStore.available(owner):
        return {**unavailable, "gateway_record_id": record_id,
                "verification_evidence": "measurement_store_not_available"}
    try:
        store = MeasurementStore(owner)
        record = store.read(record_id)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return {**unavailable, "gateway_record_id": record_id,
                "verification_evidence": "gateway_record_not_readable"}
    identity = record.get("kernels", {}).get("candidate")
    identity = identity if isinstance(identity, dict) else {}
    revision = candidate.get("git_commit")
    try:
        expected = store.source_identity(from_git(owner, revision), kernel_id=identity.get("kernel_id"))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return {
            **unavailable,
            "gateway_record_id": record_id,
            "verification_kernel_id": identity.get("kernel_id"),
            "verification_operation": record.get("operation"),
            "verification_evidence": "accepted_source_not_readable",
        }
    if identity != expected:
        return {
            **unavailable,
            "gateway_record_id": record_id,
            "verification_kernel_id": identity.get("kernel_id"),
            "verification_operation": record.get("operation"),
            "verification_evidence": "candidate_kernel_identity_mismatch",
        }
    return {
        "gateway_record_id": record_id,
        "verification_kernel_id": identity["kernel_id"],
        "verification_operation": record.get("operation"),
        "verification_evidence": "verified",
    }


def accepted_best(workspace: Path, campaign_root: Path) -> dict | None:
    """Return the lowest-latency candidate already accepted by Light."""
    search_state = campaign_root / "search" / "state.json"
    if search_state.is_file():
        rows = []
        for kernel in _search_accepted_candidates(read_state(search_state)):
            latency = finite_latency(kernel.get("latency_us_geomean"))
            if latency is None:
                continue
            trajectory = kernel.get("trajectory")
            owner = (
                campaign_root / "search" / "trajectories" / f"t{trajectory:04d}"
                if type(trajectory) is int and trajectory > 0
                else workspace
            )
            raw_version = (kernel.get("record") or {}).get("version") if isinstance(kernel.get("record"), dict) else None
            version = int(raw_version[1:]) if isinstance(raw_version, str) and re.fullmatch(r"v\d+", raw_version) else None
            rows.append(
                {
                    "latency_us_geomean": latency,
                    "git_commit": kernel.get("commit"),
                    "trajectory": trajectory,
                    "episode": kernel.get("episode"),
                    "version": version,
                    "source_sha256": kernel.get("sha256"),
                    "_owner": owner,
                }
            )
        selected = min(
            rows,
            key=lambda row: (
                row["latency_us_geomean"],
                str(row.get("git_commit")),
            ),
            default=None,
        )
        if selected is None:
            return None
        owner = selected.pop("_owner")
        return {**selected, **_accepted_evidence(owner, selected)}

    snapshot = RetainedWorkflowExport(candidate_workspace=GitCandidateWorkspace()).read(workspace)
    rows = []
    for candidate in snapshot.candidates:
        latency = finite_latency(candidate.result.latency_us)
        if (
            not candidate.promoted
            or candidate.result.state != "pass"
            or latency is None
        ):
            continue
        rows.append(
            {
                "latency_us_geomean": latency,
                "git_commit": candidate.revision,
                "version": candidate.version,
                "episode": candidate.episode,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["latency_us_geomean"],
            row["version"],
        ),
        default=None,
    )
    return {**selected, **_accepted_evidence(workspace, selected)} if selected else None


def measurement_workspaces(workspace: Path, campaign_root: Path) -> list[Path]:
    """Discover every durable Measurement Store owned by this Campaign."""
    values = [workspace]
    children = campaign_root / "search" / "trajectories"
    if children.is_dir():
        values.extend(sorted(path for path in children.glob("t*") if path.is_dir()))
    return values


def eligible_full_eval(record: dict) -> dict | None:
    """Apply main's ordinary full-Eval eligibility boundary."""
    request = record.get("request")
    if (
        record.get("operation") != "evaluate"
        or record.get("cacheable") is not True
        or not isinstance(request, dict)
        or request.get("operation") != "evaluate"
    ):
        return None

    options = request.get("options")
    if not isinstance(options, dict):
        return None
    policy = options.get("evaluation_policy", {})
    if not isinstance(policy, dict):
        return None
    command = options.get("command", [])
    if not isinstance(command, list):
        return None
    if (
        (options.get("evaluation_mode") or "full") != "full"
        or policy.get("mode", "full") != "full"
        or options.get("evaluation_input_path")
        or options.get("evaluation_shapes_path")
        or options.get("baseline_path")
        or (request.get("standard_full_evaluation") is not True and command[:2] != ["python3", "test_kernel.py"])
        or any(
            isinstance(argument, str)
            and (argument == "--shape-id" or argument.startswith("--shape-id="))
            for argument in command
        )
    ):
        return None

    response = record.get("response")
    if (
        not isinstance(response, dict)
        or response.get("exit_code") != 0
        or response.get("truncated")
    ):
        return None
    result = result_from_response(response, "evaluate")
    if (
        not isinstance(result, dict)
        or result.get("all_pass") is not True
        or result.get("error")
        or result.get("mode", "full") != "full"
    ):
        return None
    return result


def best_passing_in_store(store: MeasurementStore, owner: Path) -> list[dict]:
    """Return eligible full-Eval observations from one validated store."""
    rows = []
    for record_id in store.record_ids():
        if RECORD_ID.fullmatch(record_id) is None:
            continue
        try:
            record = store.read(record_id)
            result = eligible_full_eval(record)
            if result is None:
                continue
            latencies = result.get("latency_us_by_shape")
            if not isinstance(latencies, dict) or not latencies:
                continue
            values = [finite_latency(value) for value in latencies.values()]
            if any(value is None for value in values):
                continue

            latency = finite_latency(result.get("latency_us_geomean"))
            if latency is None:
                latency = math.exp(
                    sum(math.log(value) for value in values) / len(values)
                )

            identity = record["kernels"]["candidate"]
            kernel_id = identity["kernel_id"]
            rows.append(
                {
                    "latency_us_geomean": latency,
                    "kernel_id": kernel_id,
                    "kernel_artifact_digest": identity.get("kernel_artifact_digest"),
                    "gateway_record_id": record_id,
                    "shape_count": len(latencies),
                    "measurement_workspace": str(owner),
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RecursionError,
            OverflowError,
        ):
            # Match the existing frontier query: one corrupt historical record
            # must not make the entire operator status query unavailable.
            continue
    return rows


def best_passing_full_eval(workspace: Path, campaign_root: Path) -> dict | None:
    """Return the best passing standard full Eval across the Campaign."""
    rows = []
    for owner in measurement_workspaces(workspace, campaign_root):
        store_root = supervisor_campaign_root(owner) / "measurements"
        # Avoid initializing a new store from an observability command.
        if not MeasurementStore.available(owner):
            continue
        rows.extend(best_passing_in_store(MeasurementStore(owner), owner))
    return min(
        rows,
        key=lambda row: (
            row["latency_us_geomean"],
            str(row.get("kernel_id")),
            str(row.get("gateway_record_id")),
        ),
        default=None,
    )


def campaign_snapshot(workspace: Path) -> dict:
    """Build one point-in-time, read-only Campaign status snapshot."""
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Campaign workspace does not exist: {workspace}")
    campaign_root = supervisor_campaign_root(workspace)
    return {
        "schema_version": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "campaign_workspace": str(workspace),
        "accepted_best": accepted_best(workspace, campaign_root),
        "best_passing_full_eval": best_passing_full_eval(workspace, campaign_root),
    }


def render(snapshot: dict, *, compact: bool) -> str:
    return json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read accepted-best and passing-full-Eval-best latency from a "
            "running Light Campaign without submitting Agate work."
        )
    )
    parser.add_argument("workspace", type=Path, help="top-level Light Campaign workspace")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="repeat until interrupted; SECONDS must be positive",
    )
    parser.add_argument("--compact", action="store_true", help="emit one JSON line per snapshot")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit formatted JSON (accepted for consistency with other status commands)",
    )
    options = parser.parse_args(argv)
    if options.watch is not None and (
        not math.isfinite(options.watch) or options.watch <= 0
    ):
        parser.error("--watch must be a finite positive number of seconds")
    return options


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    while True:
        try:
            print(render(campaign_snapshot(options.workspace), compact=options.compact), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"light-latency-status: {error}", file=sys.stderr, flush=True)
            if options.watch is None:
                return 1
        if options.watch is None:
            return 0
        time.sleep(options.watch)


if __name__ == "__main__":
    raise SystemExit(main())
