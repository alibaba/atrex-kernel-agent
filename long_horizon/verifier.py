from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from threading import Event
from typing import Any

from orchestrator._bootstrap import task_modules

task_modules()
from aka.legacy.evaluation.abba import (
    verification_schedule, geomean as _geomean, merge_batch_results as _merge_batch_results,
    result_latency as _result_latency, result_performance_score as _result_performance_score,
    score_verification_payload,
)
from aka.legacy.evaluation.resources import remote_abba_script

from orchestrator.infrastructure_retry import InfrastructureUnavailable, check_transport, retry_infrastructure

from . import main_adapter
from .models import VerificationResult, VerificationRun
from .protocol import atomic_write_json
from .store import VERIFY_DIR

ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
DEFAULT_SHAPE_BATCH_SIZE = 4
DEFAULT_SHAPE_BATCH_WORKERS = 4


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the long-horizon ABBA payload from ordinary sandbox stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(ABBA_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ValueError("malformed ABBA result sentinel") from exc
        if not isinstance(payload, dict):
            raise ValueError("ABBA result sentinel must contain a JSON object")
        return payload
    raise ValueError("missing ABBA result sentinel")


def _verification_shape_batches(
    workspace: Path, private_reference_dir: Path | None, batch_size: int
) -> tuple[list[list[str] | None], list[str]]:
    harness = workspace / "test_kernel.py"
    if not harness.is_file() or "--shape-id" not in harness.read_text(encoding="utf-8"):
        return [None], []
    shapes_path = (private_reference_dir or workspace) / "shapes.json"
    if not shapes_path.is_file():
        return [None], []
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    shape_ids = sorted(
        (str(shape_id) for shape_id in shapes),
        key=lambda shape_id: (
            0,
            int(shape_id),
        )
        if shape_id.isdigit()
        else (1, shape_id),
    )
    return (
        [
            shape_ids[offset : offset + batch_size]
            for offset in range(0, len(shape_ids), batch_size)
        ],
        shape_ids,
    )


def _merge_batch_payloads(
    payloads: list[dict[str, Any]],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
) -> dict[str, Any]:
    if len(payloads) == 1:
        return payloads[0]
    for payload in payloads:
        rows = payload.get("runs")
        if (
            payload.get("schema_version") != 1
            or payload.get("error")
            or not isinstance(rows, list)
            or len(rows) != len(schedule)
        ):
            raise ValueError(str(payload.get("error") or "invalid shape batch result"))
        if any(
            not isinstance(row, dict) or not isinstance(row.get("result"), dict)
            for row in rows
        ):
            raise ValueError("shape batch has no valid run result")
        actual = [
            {"revision": row.get("revision"), "repeat": row.get("repeat")}
            for row in rows
        ]
        if actual != schedule:
            raise ValueError("shape batch returned an invalid ABBA schedule")
    runs = []
    for index, step in enumerate(schedule):
        rows = [payload["runs"][index] for payload in payloads]
        result = _merge_batch_results([row["result"] for row in rows], shape_ids)
        runs.append(
            {
                **step,
                "exit_code": 0
                if result["all_pass"] and all(row["exit_code"] == 0 for row in rows)
                else 1,
                "result": result,
                "stdout_tail": "\n".join(
                    str(row.get("stdout_tail", "")) for row in rows
                )[-3000:],
                "stderr_tail": "\n".join(
                    str(row.get("stderr_tail", "")) for row in rows
                )[-3000:],
            }
        )
    return {
        "schema_version": 1,
        "runs": runs,
        "error": "; ".join(
            str(payload["error"]) for payload in payloads if payload.get("error")
        )
        or None,
        "shape_batch_count": len(payloads),
    }


def _snapshot_revision(workspace, revision, changed_paths, destination, label, *, provider=None):
    from aka.legacy.application.workspace import candidate_workspace
    with candidate_workspace(provider=provider) as provider:
        return provider.write_revision(workspace, revision, changed_paths, destination, label)


class GatewayABBAValidator:
    def __init__(
        self,
        *,
        hardware: str,
        profile: str = "",
        url: str = "",
        ssh: str = "",
        ssh_init: str = "",
        health_command: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        min_improvement_pct: float = 0.0,
        queue_wait_grace: int = 14_400,
        private_reference_dir: Path | None = None,
        shape_batch_size: int = DEFAULT_SHAPE_BATCH_SIZE,
        workspace_provider=None,
    ):
        self.workspace_provider = workspace_provider
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.ssh = ssh
        self.ssh_init = ssh_init
        self.health_command = health_command
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace
        self.private_reference_dir = private_reference_dir
        self.shape_batch_size = shape_batch_size

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        schedule = verification_schedule(self.repeats)
        if self.per_run_timeout * len(schedule) + 30 > self.timeout:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error="ABBA schedule cannot fit in one gateway allocation timeout",
            )
        shape_batches, expected_shape_ids = _verification_shape_batches(
            workspace, self.private_reference_dir, self.shape_batch_size
        )
        verification_id = uuid.uuid4().hex
        relative_dir = f"{VERIFY_DIR}/{verification_id}"
        directory = workspace / relative_dir
        directory.mkdir(parents=True, exist_ok=False)
        # Naming the driver test_kernel.py deliberately selects the sandbox's
        # immutable-evaluator payload route. The runtime bundler includes the verification
        # driver and snapshots from their dedicated artifact directory.
        driver = directory / "test_kernel.py"
        shutil.copy2(remote_abba_script(), driver)
        manifests = {
            "incumbent": _snapshot_revision(
                workspace, base_commit, changed_paths, directory, "incumbent", provider=self.workspace_provider
            ),
            "candidate": _snapshot_revision(
                workspace, candidate_commit, changed_paths, directory, "candidate", provider=self.workspace_provider
            ),
        }
        batch_specs = []
        for index, shape_ids in enumerate(shape_batches):
            request_relative = f"{relative_dir}/request-{index:04d}.json"
            result_relative = f"{relative_dir}/result-{index:04d}.json"
            command = [
                "python3",
                "test_kernel.py",
                "--version",
                "vlong",
                "--no-memory",
            ]
            for shape_id in shape_ids or []:
                command += ["--shape-id", shape_id]
            atomic_write_json(
                workspace / request_relative,
                {
                    "schema_version": 1,
                    "schedule": schedule,
                    "manifests": manifests,
                    "command": command,
                    "run_timeout_seconds": self.per_run_timeout,
                },
            )
            batch_specs.append((request_relative, result_relative))

        cancel = Event()

        def run_batch(spec: tuple[str, str]) -> dict[str, Any]:
            request_relative, result_relative = spec
            def evaluate_batch():
                try:
                    process = main_adapter.run_sandbox(
                        workspace,
                        self.hardware,
                        self.profile,
                        self.url,
                        self.timeout,
                        [
                            "python3",
                            f"{relative_dir}/test_kernel.py",
                            request_relative,
                            result_relative,
                        ],
                        ssh=self.ssh,
                        ssh_init=self.ssh_init,
                        health_command=self.health_command,
                        sync=(),
                        wall_timeout=self.timeout + self.queue_wait_grace + 120,
                        gateway_kind="dev",
                        private_reference_dir=self.private_reference_dir,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise InfrastructureUnavailable("GPU transport wait deadline exceeded") from exc
                output = process.stdout + "\n" + process.stderr
                if process.returncode == 0:
                    payload = _payload_from_stdout(process.stdout)
                    atomic_write_json(workspace / result_relative, payload)
                    return payload
                check_transport(process)
                raise RuntimeError(
                    f"gateway ABBA command exited {process.returncode}: "
                    + output[-3000:]
                )
            return retry_infrastructure(
                workspace,
                f"abba:{base_commit}:{candidate_commit}:{self.repeats}:{self.shape_batch_size}:"
                f"{Path(request_relative).name}",
                evaluate_batch, cancel=cancel)

        try:
            # One explicitly assigned SSH GPU must never run multiple timing batches
            # concurrently. Gateway allocations remain parallel. Outages wait inside
            # their batch; a definitive failure cancels the remaining batch waits.
            max_workers = (
                1
                if self.ssh
                else min(DEFAULT_SHAPE_BATCH_WORKERS, len(batch_specs))
            )
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = [executor.submit(copy_context().run, run_batch, spec) for spec in batch_specs]
            try:
                completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in completed:
                    future.result()
                payloads = [future.result() for future in futures]
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            payload = _merge_batch_payloads(payloads, schedule, expected_shape_ids)
        except (
            subprocess.SubprocessError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"gateway ABBA verification failed: {exc}",
            )

        result_path = directory / "result.json"
        try:
            atomic_write_json(result_path, payload)
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"cannot persist gateway ABBA result: {type(exc).__name__}: {exc}",
            )
        return score_verification_payload(
            payload,
            schedule=schedule,
            repeats=self.repeats,
            min_improvement_pct=self.min_improvement_pct,
            artifact=str(result_path),
        )
