#!/usr/bin/env python3
"""Report accepted and observed-best latency for a running Light Campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time


# Allow direct execution from a source checkout without installing AKA.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aka.light.evaluation.provider import result_from_response  # noqa: E402
from aka.light.experiment.store import (  # noqa: E402
    KERNEL_ID,
    MeasurementStore,
    RECORD_ID,
)
from aka.light.runtime.paths import supervisor_campaign_root  # noqa: E402


def finite_latency(value: object) -> float | None:
    """Return a finite positive latency without treating bool as a number."""
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _document(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _attempts(workspace: Path) -> list[dict]:
    root = workspace / ".atrex_long_horizon"
    rows: dict[int, dict] = {}
    state = _document(root / "state.json") or {}
    for value in state.get("attempts") or []:
        if isinstance(value, dict) and type(value.get("episode")) is int:
            rows[value["episode"]] = value
    for path in sorted((root / "episodes").glob("e*/attempt.json")):
        value = _document(path)
        if isinstance(value, dict) and type(value.get("episode")) is int:
            rows[value["episode"]] = value
    return [rows[key] for key in sorted(rows)]


def _attempt_commit(attempt: dict) -> str | None:
    value = attempt.get("candidate_kernel_commit_id") or attempt.get("candidate_commit")
    return value if isinstance(value, str) and KERNEL_ID.fullmatch(value) else None


def _attempt_record_id(attempt: dict) -> str | None:
    value = attempt.get("gateway_record_id")
    verification = attempt.get("verification")
    if not isinstance(value, str) and isinstance(verification, dict):
        value = verification.get("gateway_record_id") or verification.get("artifact")
    return value if isinstance(value, str) and RECORD_ID.fullmatch(value) else None


def _attempt_latency(attempt: dict) -> float | None:
    value = finite_latency(attempt.get("latency_us_geomean"))
    verification = attempt.get("verification")
    if value is None and isinstance(verification, dict):
        value = finite_latency(verification.get("candidate_latency_us"))
    return value


def _accepted_evidence(owner: Path, candidate: dict, attempt: dict) -> dict:
    """Resolve an accepted Candidate to its exact terminal verification Record."""
    unavailable = {
        "gateway_record_id": None,
        "verification_kernel_commit_id": None,
        "verification_kernel_artifact_digest": None,
        "verification_operation": None,
    }
    episode = candidate.get("episode")
    if type(episode) is not int or episode < 0:
        return {**unavailable, "verification_evidence": "episode_identity_not_available"}
    if attempt.get("accepted") is not True:
        return {**unavailable, "verification_evidence": "attempt_not_accepted"}
    record_id = _attempt_record_id(attempt)
    if record_id is None:
        return {**unavailable, "verification_evidence": "gateway_record_id_not_available"}
    store_root = supervisor_campaign_root(owner) / "measurements"
    if not (store_root / "identity.key").is_file():
        return {**unavailable, "gateway_record_id": record_id,
                "verification_evidence": "measurement_store_not_available"}
    try:
        record = MeasurementStore(store_root).read(record_id)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return {**unavailable, "gateway_record_id": record_id,
                "verification_evidence": "gateway_record_not_readable"}
    identity = record.get("kernels", {}).get("candidate")
    identity = identity if isinstance(identity, dict) else {}
    commit = candidate.get("kernel_commit_id")
    # MeasurementStore.read() already verifies the source snapshot and digest.
    # The remaining public-identity check is the Agent's own Git commit ID;
    # never try to resolve it in the Supervisor/controller repository.
    if identity.get("kernel_id") != commit:
        return {
            **unavailable,
            "gateway_record_id": record_id,
            "verification_kernel_commit_id": identity.get("kernel_id"),
            "verification_kernel_artifact_digest": identity.get(
                "kernel_artifact_digest"
            ),
            "verification_operation": record.get("operation"),
            "verification_evidence": "candidate_kernel_identity_mismatch",
        }
    return {
        "gateway_record_id": record_id,
        "verification_kernel_commit_id": identity["kernel_id"],
        "verification_kernel_artifact_digest": identity.get(
            "kernel_artifact_digest"
        ),
        "verification_operation": record.get("operation"),
        "verification_evidence": "verified",
    }


def accepted_best(workspace: Path, campaign_root: Path) -> dict | None:
    """Return the lowest-latency candidate already accepted by Light."""
    rows = []
    for owner in measurement_workspaces(workspace, campaign_root):
        trajectory = None
        if owner != workspace:
            match = re.fullmatch(r"t0*([1-9][0-9]*)", owner.name)
            trajectory = int(match.group(1)) if match else None
        for attempt in _attempts(owner):
            commit = _attempt_commit(attempt)
            latency = _attempt_latency(attempt)
            if attempt.get("accepted") is not True or commit is None or latency is None:
                continue
            row = {
                "latency_us_geomean": latency,
                "kernel_commit_id": commit,
                "trajectory": trajectory,
                "episode": attempt.get("episode"),
                "version": attempt.get("version", attempt.get("memory_version")),
                "_owner": owner,
                "_attempt": attempt,
            }
            rows.append(row)
    selected = min(
        rows,
        key=lambda row: (
            row["latency_us_geomean"],
            str(row["kernel_commit_id"]),
        ),
        default=None,
    )
    if selected is None:
        return None
    owner = selected.pop("_owner")
    attempt = selected.pop("_attempt")
    return {**selected, **_accepted_evidence(owner, selected, attempt)}


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
        or command[:2] != ["python3", "test_kernel.py"]
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
    for record_path in (store.root / "records").glob("gateway-*/record.json"):
        record_id = record_path.parent.name
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
            kernel_commit_id = identity["kernel_id"]
            rows.append(
                {
                    "latency_us_geomean": latency,
                    "kernel_commit_id": kernel_commit_id,
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
        if not (store_root / "identity.key").is_file():
            continue
        rows.extend(best_passing_in_store(MeasurementStore(store_root), owner))
    return min(
        rows,
        key=lambda row: (
            row["latency_us_geomean"],
            str(row.get("kernel_commit_id")),
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
        "schema_version": 2,
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
