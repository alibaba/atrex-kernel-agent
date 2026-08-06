from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from long_horizon.verifier import ABBA_RESULT_PREFIX, verification_schedule
from orchestrator import optimize

from .benchmark import MaterializedTeacherWorkspace
from .models import AbbaStatus


SandboxRunner = Callable[[Path, list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class TeacherABBAResult:
    status: AbbaStatus
    candidate_latency_us: float | None
    teacher_latency_us: float | None
    candidate_to_teacher_ratio: float | None
    worst_shape_ratio: float | None
    worst_shape_key: str | None
    runs: tuple[dict[str, Any], ...] = ()
    error: str = ""
    artifact: str = ""


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _error(status: AbbaStatus, message: str, runs: list[dict[str, Any]] | None = None) -> TeacherABBAResult:
    return TeacherABBAResult(status, None, None, None, None, None, tuple(runs or ()), message)


def score_teacher_abba_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    expected_shape_keys: tuple[str, ...],
    geomean_ratio: float,
    shape_ratio: float,
    artifact: str = "",
) -> TeacherABBAResult:
    """Score one exact same-allocation Teacher(A)/Candidate(B) ABBA payload."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return _error(AbbaStatus.INFRA_ERROR, "unsupported ABBA result schema")
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return _error(AbbaStatus.INFRA_ERROR, "ABBA runs must be a list of objects")
    if payload.get("error"):
        return _error(AbbaStatus.INFRA_ERROR, str(payload["error"]), rows)
    actual = [
        {"revision": row.get("revision"), "repeat": row.get("repeat")}
        for row in rows
    ]
    if actual != schedule:
        return _error(AbbaStatus.INFRA_ERROR, "remote verifier did not execute the exact ABBA schedule", rows)

    expected = set(expected_shape_keys)
    geomeans: dict[str, list[float]] = {"incumbent": [], "candidate": []}
    shape_values: dict[str, dict[str, list[float]]] = {
        "incumbent": {key: [] for key in expected_shape_keys},
        "candidate": {key: [] for key in expected_shape_keys},
    }
    for row in rows:
        revision = row.get("revision")
        result = row.get("result")
        if revision not in geomeans:
            return _error(AbbaStatus.INFRA_ERROR, "ABBA result has an unknown revision", rows)
        exit_code = row.get("exit_code")
        if exit_code == -1 or not isinstance(result, dict):
            return _error(
                AbbaStatus.INFRA_ERROR,
                "authoritative ABBA run did not execute to a structured result",
                rows,
            )
        if exit_code != 0 or not result.get("all_pass"):
            return _error(AbbaStatus.FAIL, "not every authoritative ABBA run passed correctness", rows)
        geomean = result.get("latency_us_geomean")
        by_shape = result.get("latency_us_by_shape")
        if (
            isinstance(geomean, bool)
            or not isinstance(geomean, (int, float))
            or not math.isfinite(float(geomean))
            or float(geomean) <= 0.0
        ):
            return _error(AbbaStatus.FAIL, "ABBA run reported an invalid geomean", rows)
        if not isinstance(by_shape, dict) or set(by_shape) != expected:
            return _error(AbbaStatus.FAIL, "ABBA run did not cover the exact shape set", rows)
        geomeans[revision].append(float(geomean))
        for key, value in by_shape.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                return _error(AbbaStatus.FAIL, "ABBA run reported an invalid shape latency", rows)
            shape_values[revision][key].append(float(value))

    if any(len(values) != repeats for values in geomeans.values()):
        return _error(AbbaStatus.INFRA_ERROR, "ABBA repeat count does not match the request", rows)
    teacher_latency = _geomean(geomeans["incumbent"])
    candidate_latency = _geomean(geomeans["candidate"])
    if teacher_latency is None or candidate_latency is None:
        return _error(AbbaStatus.FAIL, "ABBA aggregate latency is invalid", rows)
    candidate_ratio = candidate_latency / teacher_latency

    ratios: dict[str, float] = {}
    for key in expected_shape_keys:
        teacher_shape = _geomean(shape_values["incumbent"][key])
        candidate_shape = _geomean(shape_values["candidate"][key])
        if teacher_shape is None or candidate_shape is None:
            return _error(AbbaStatus.FAIL, "ABBA shape aggregate latency is invalid", rows)
        ratios[key] = candidate_shape / teacher_shape
    worst_key = max(expected_shape_keys, key=lambda key: ratios[key])
    worst_ratio = ratios[worst_key]

    status = AbbaStatus.PASS
    error = ""
    if candidate_ratio > geomean_ratio:
        status = AbbaStatus.FAIL
        error = "candidate geomean ratio %.6f exceeds %.6f" % (candidate_ratio, geomean_ratio)
    elif worst_ratio > shape_ratio:
        status = AbbaStatus.FAIL
        error = "candidate shape ratio %.6f exceeds %.6f for %s" % (
            worst_ratio,
            shape_ratio,
            worst_key,
        )
    return TeacherABBAResult(
        status=status,
        candidate_latency_us=candidate_latency,
        teacher_latency_us=teacher_latency,
        candidate_to_teacher_ratio=candidate_ratio,
        worst_shape_ratio=worst_ratio,
        worst_shape_key=worst_key,
        runs=tuple(rows),
        error=error,
        artifact=artifact,
    )


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("unsafe solution source path: %s" % value)
    return path.as_posix()


def _solution_sources(content: bytes | None) -> set[str]:
    if content is None:
        return set()
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("solution.json is not valid UTF-8 JSON") from exc
    sources = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(sources, list):
        return set()
    result: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path"}:
            raise ValueError("solution sources must contain only path entries")
        result.add(_safe_relative(source["path"]))
    return result


def _git_blob(workspace: Path, commit: str, relative: str) -> bytes | None:
    process = subprocess.run(
        ["git", "show", "%s:%s" % (commit, relative)],
        cwd=workspace,
        capture_output=True,
        check=False,
    )
    return process.stdout if process.returncode == 0 else None


def _snapshot(
    verification_dir: Path,
    label: str,
    contents: Mapping[str, bytes | None],
) -> dict[str, str | None]:
    manifest: dict[str, str | None] = {}
    for index, relative in enumerate(sorted(contents)):
        content = contents[relative]
        if content is None:
            manifest[relative] = None
            continue
        snapshot_relative = "snapshots/%s/%04d.bin" % (label, index)
        target = verification_dir / snapshot_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative] = snapshot_relative
    return manifest


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(ABBA_RESULT_PREFIX):
            try:
                value = json.loads(line[len(ABBA_RESULT_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise ValueError("malformed ABBA result sentinel") from exc
            if isinstance(value, dict):
                return value
    raise ValueError("missing ABBA result sentinel")


class TeacherABBAValidator:
    def __init__(
        self,
        *,
        sandbox_runner: SandboxRunner | None = None,
        hardware: str = "",
        profile: str = "",
        url: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        geomean_ratio: float = 1.05,
        shape_ratio: float = 1.10,
    ) -> None:
        self.sandbox_runner = sandbox_runner
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.geomean_ratio = geomean_ratio
        self.shape_ratio = shape_ratio

    def _run(self, workspace: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        if self.sandbox_runner is not None:
            return self.sandbox_runner(workspace, command)
        if not self.hardware:
            raise ValueError("Teacher ABBA requires sandbox hardware")
        return optimize._sandbox_command(
            workspace,
            self.hardware,
            self.profile,
            self.url,
            self.timeout,
            command,
            gateway_kind="dev",
            wall_timeout=self.timeout + 14_520,
        )

    def verify(
        self,
        *,
        candidate_workspace: Path,
        candidate_commit: str,
        teacher: MaterializedTeacherWorkspace,
    ) -> TeacherABBAResult:
        schedule = verification_schedule(self.repeats)
        if self.per_run_timeout * len(schedule) + 30 > self.timeout:
            return _error(AbbaStatus.INFRA_ERROR, "ABBA schedule cannot fit in one allocation")

        private_workspace = teacher.workspace
        relative_dir = "aggregate_kernels/.atrex_teacher_verify/%s" % uuid.uuid4().hex
        verification_dir = private_workspace / relative_dir
        verification_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "long_horizon" / "remote_abba.py",
            verification_dir / "test_kernel.py",
        )

        teacher_solution = (private_workspace / "solution.json").read_bytes() if (private_workspace / "solution.json").is_file() else None
        candidate_solution = _git_blob(candidate_workspace, candidate_commit, "solution.json")
        paths = {"kernel.py", "solution.json"}
        paths.update(_solution_sources(teacher_solution))
        paths.update(_solution_sources(candidate_solution))
        teacher_contents = {
            path: (private_workspace / path).read_bytes() if (private_workspace / path).is_file() else None
            for path in paths
        }
        candidate_contents = {
            path: _git_blob(candidate_workspace, candidate_commit, path)
            for path in paths
        }
        manifests = {
            "incumbent": _snapshot(verification_dir, "teacher", teacher_contents),
            "candidate": _snapshot(verification_dir, "candidate", candidate_contents),
        }
        request_relative = "%s/request.json" % relative_dir
        result_relative = "%s/result.json" % relative_dir
        request = {
            "schema_version": 1,
            "schedule": schedule,
            "manifests": manifests,
            "command": [
                "python3",
                "test_kernel.py",
                "--version",
                "vteacher-abba",
                "--no-memory",
                "--multi-seed",
                "5",
            ],
            "run_timeout_seconds": self.per_run_timeout,
        }
        (private_workspace / request_relative).write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            "python3",
            "%s/test_kernel.py" % relative_dir,
            request_relative,
            result_relative,
        ]
        try:
            process = self._run(private_workspace, command)
        except (OSError, subprocess.SubprocessError) as exc:
            return _error(AbbaStatus.INFRA_ERROR, "Teacher ABBA failed to run: %s" % exc)
        if process.returncode != 0:
            return _error(
                AbbaStatus.INFRA_ERROR,
                "Teacher ABBA command exited %d: %s" % (
                    process.returncode,
                    (process.stdout + "\n" + process.stderr)[-3000:],
                ),
            )
        try:
            payload = _payload_from_stdout(process.stdout)
        except ValueError as exc:
            return _error(AbbaStatus.INFRA_ERROR, str(exc))
        artifact = private_workspace / result_relative
        result = score_teacher_abba_payload(
            payload,
            schedule=schedule,
            repeats=self.repeats,
            expected_shape_keys=teacher.expected_shape_keys,
            geomean_ratio=self.geomean_ratio,
            shape_ratio=self.shape_ratio,
            artifact=str(artifact),
        )
        artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification_status": result.status.value,
                    "candidate_to_teacher_ratio": result.candidate_to_teacher_ratio,
                    "worst_shape_ratio": result.worst_shape_ratio,
                    "worst_shape_key": result.worst_shape_key,
                    "error": result.error,
                    "payload": payload,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result
