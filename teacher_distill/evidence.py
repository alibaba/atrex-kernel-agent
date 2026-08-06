from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator import optimize

from .state import write_json_atomic


@dataclass(frozen=True)
class EvidenceBundle:
    root: Path
    manifest_path: Path
    trajectory_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_number(path: Path) -> int:
    match = re.fullmatch(r"v(\d+)\.json", path.name)
    return int(match.group(1)) if match else 1 << 60


def _abba_classification(path: Path) -> tuple[str, bool]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid-run", False
    status = value.get("verification_status") if isinstance(value, dict) else None
    if status == "PASS":
        return "verified-pass", True
    if status == "FAIL":
        return "verified-fail", False
    if status == "INFRA_ERROR":
        return "infrastructure-error", False
    return "unscored-run", False


def _classification(memory: dict) -> tuple[str, bool]:
    if memory.get("masked", False):
        return "masked", False
    gate = (memory.get("quality_gate") or {}).get("result")
    correctness = (memory.get("correctness") or {}).get("status")
    commit = memory.get("git_commit_hash")
    if gate in {"FAIL", "TIMEOUT_FAIL"} or correctness in {"FAIL", "TIMEOUT_FAIL"}:
        return "reverted", False
    if gate == "PASS" or correctness == "PASS":
        return "accepted", bool(commit)
    return "exploratory", False


def _git_changed_paths(workspace: Path, commit: str) -> list[str]:
    process = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(line for line in process.stdout.splitlines() if line) if process.returncode == 0 else []


def _evidence_entry(
    *,
    evidence_id: str,
    kind: str,
    path: Path,
    scope: str,
    relative: str,
    classification: str,
    citable: bool,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "evidence_id": evidence_id,
        "kind": kind,
        "scope": scope,
        "path": relative,
        "sha256": _sha256(path),
        "classification": classification,
        "citable_as_verified": bool(citable),
    }
    if version is not None:
        entry["version"] = version
    if extra:
        entry.update(extra)
    return entry


def build_evidence_bundle(workspace: Path | str, private_dir: Path | str) -> EvidenceBundle:
    candidate = Path(workspace).resolve()
    private = Path(private_dir).resolve()
    output = private / "distillation" / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    evidence: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    classifications: dict[int, tuple[str, bool]] = {}

    memory_files = sorted((candidate / "memory").glob("v*.json"), key=_version_number)
    for path in memory_files:
        match = re.fullmatch(r"v(\d+)\.json", path.name)
        if match is None:
            continue
        number = int(match.group(1))
        version = "v%d" % number
        try:
            memory = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            memory = {}
        if not isinstance(memory, dict):
            memory = {}
        classification, citable = _classification(memory)
        commit = memory.get("git_commit_hash")
        if citable and isinstance(commit, str):
            citable = optimize.commit_changed_kernel(candidate, commit)
        classifications[number] = (classification, citable)
        evidence.append(
            _evidence_entry(
                evidence_id="E-V%d-MEMORY" % number,
                kind="memory",
                path=path,
                scope="candidate",
                relative="memory/%s" % path.name,
                classification=classification,
                citable=citable,
                version=version,
                extra={"git_commit_hash": commit},
            )
        )
        if isinstance(commit, str) and commit:
            evidence.append(
                {
                    "evidence_id": "E-V%d-GIT" % number,
                    "kind": "git",
                    "scope": "candidate",
                    "path": "git:%s" % commit,
                    "sha256": hashlib.sha256(commit.encode("utf-8")).hexdigest(),
                    "classification": classification,
                    "citable_as_verified": citable,
                    "version": version,
                    "git_commit_hash": commit,
                    "changed_paths": _git_changed_paths(candidate, commit),
                }
            )
        performance = memory.get("performance") or {}
        progress = memory.get("teacher_progress") or {}
        trajectory.append(
            {
                "version": version,
                "evidence_id": "E-V%d-MEMORY" % number,
                "classification": classification,
                "masked": bool(memory.get("masked", False)),
                "latency_us": performance.get("latency_us"),
                "latency_us_by_shape": performance.get("latency_us_by_shape") or {},
                "teacher_ratio": progress.get("candidate_to_teacher_geomean_ratio"),
                "worst_shape_ratio": progress.get("worst_shape_ratio"),
                "abba_status": progress.get("abba_status"),
                "action_category": (memory.get("optimization") or {}).get("action_category"),
                "quality_gate": (memory.get("quality_gate") or {}).get("result"),
                "correctness": (memory.get("correctness") or {}).get("status"),
            }
        )

    for kind, pattern, suffix in (
        ("plan", "plans/v*_plan.md", "PLAN"),
        ("profile", "profiles/v*/REPORT.md", "PROFILE"),
    ):
        for path in sorted(candidate.glob(pattern)):
            match = re.search(r"/v(\d+)(?:_|/)", path.as_posix())
            if match is None:
                continue
            number = int(match.group(1))
            classification, citable = classifications.get(number, ("exploratory", False))
            evidence.append(
                _evidence_entry(
                    evidence_id="E-V%d-%s" % (number, suffix),
                    kind=kind,
                    path=path,
                    scope="candidate",
                    relative=path.relative_to(candidate).as_posix(),
                    classification=classification,
                    citable=citable,
                    version="v%d" % number,
                )
            )

    abba_files = sorted(private.glob("**/.atrex_teacher_verify/*/result.json"))
    for index, path in enumerate(abba_files, 1):
        classification, citable = _abba_classification(path)
        evidence.append(
            _evidence_entry(
                evidence_id="E-ABBA-%03d" % index,
                kind="teacher-abba",
                path=path,
                scope="private",
                relative=path.relative_to(private).as_posix(),
                classification=classification,
                citable=citable,
            )
        )

    benchmark_files = sorted(private.glob("teacher_workspace/benchmark_runs/*.json"))
    for index, path in enumerate(benchmark_files, 1):
        evidence.append(
            _evidence_entry(
                evidence_id="E-TEACHER-BENCH-%03d" % index,
                kind="teacher-benchmark",
                path=path,
                scope="private",
                relative=path.relative_to(private).as_posix(),
                classification="verified-run",
                citable=True,
            )
        )

    audit_files = sorted(private.glob("audit/*.jsonl"))
    for index, path in enumerate(audit_files, 1):
        evidence.append(
            _evidence_entry(
                evidence_id="E-AUDIT-%03d" % index,
                kind="access-audit",
                path=path,
                scope="private",
                relative=path.relative_to(private).as_posix(),
                classification="policy-violation",
                citable=False,
            )
        )

    episode_files = sorted(candidate.glob(".atrex_long_horizon/episodes/*/attempt.json"))
    for index, path in enumerate(episode_files, 1):
        evidence.append(
            _evidence_entry(
                evidence_id="E-EPISODE-%03d" % index,
                kind="exploration-episode",
                path=path,
                scope="candidate",
                relative=path.relative_to(candidate).as_posix(),
                classification="exploratory",
                citable=False,
            )
        )

    checkpoint_files = sorted(
        (path, root)
        for root, patterns in (
            (
                candidate,
                (
                    ".atrex_long_horizon/episodes/*/archive/**/*",
                    ".atrex_long_horizon/episodes/*/interrupted_archive/**/*",
                ),
            ),
            (
                private,
                (
                    "long_horizon/episodes/*/archive/**/*",
                    "long_horizon/episodes/*/interrupted_archive/**/*",
                ),
            ),
        )
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    )
    for index, (path, checkpoint_root) in enumerate(checkpoint_files, 1):
        evidence.append(
            _evidence_entry(
                evidence_id="E-EPISODE-CHECKPOINT-%03d" % index,
                kind="episode-checkpoint",
                path=path,
                scope="private",
                relative=path.relative_to(checkpoint_root).as_posix(),
                classification="exploratory",
                citable=False,
            )
        )

    evidence.sort(key=lambda entry: entry["evidence_id"])
    manifest = {
        "schema_version": 1,
        "workspace_head": optimize.git_head(candidate),
        "evidence": evidence,
    }
    performance_trajectory = {
        "schema_version": 1,
        "versions": trajectory,
    }
    manifest_path = output / "evidence_manifest.json"
    trajectory_path = output / "performance_trajectory.json"
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(trajectory_path, performance_trajectory)
    return EvidenceBundle(output, manifest_path, trajectory_path)
