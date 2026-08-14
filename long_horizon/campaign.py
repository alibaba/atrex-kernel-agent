from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import main_adapter
from .git_episode import (
    EpisodeWorktree,
    git_head,
    git_text,
    promote_candidate,
    record_episode_outcome,
    working_changes,
)
from .journal import initialize as initialize_journal
from .journal import load as load_journal
from .journal import sync_live_memory
from .journal import validate_terminal
from .models import EpisodeHandoff, SupervisorState, VerificationResult
from .session import LongSessionRunner
from .store import CampaignStore, RUNTIME_DIR, VERIFY_DIR
from .telemetry import summarize_episode
from .verifier import GatewayABBAValidator


MODULE_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = MODULE_ROOT / "orchestrator" / "prompts" / "episode.md"
EVIDENCE_PREFIXES = ("plans/", "profiles/")
MEMORY_EXPERIMENT_FIELDS = (
    "name",
    "hypothesis",
    "change",
    "evidence",
    "result",
    "decision",
    "timestamp",
)
MAX_MEMORY_EXPERIMENT_FIELD_CHARS = 2_000
EPISODE_EVALUATIONS_PATH = Path(".atrex_long_horizon/evaluations.jsonl")


def _render(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def _iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _conversion_parity_passes(verification: VerificationResult) -> bool:
    candidate = verification.candidate_latency_us
    incumbent = verification.incumbent_latency_us
    if not isinstance(candidate, (int, float)) or not isinstance(
        incumbent, (int, float)
    ):
        return False
    if candidate > incumbent * (1.0 + main_adapter.CONVERT_PERF_TOL):
        return False
    return bool(verification.runs) and all(
        run.exit_code == 0
        and isinstance(run.result, dict)
        and bool(run.result.get("all_pass"))
        for run in verification.runs
    )


def _representative_candidate_result(
    verification: VerificationResult | None,
) -> dict[str, Any]:
    """Return the latest real candidate measurement from authoritative verification."""
    if verification is None:
        return {}
    for run in reversed(verification.runs):
        if run.revision == "candidate" and isinstance(run.result, dict):
            return run.result
    return {}


def _candidate_shape_latencies(
    verification: VerificationResult | None,
) -> tuple[dict[str, float], int]:
    """Aggregate real per-shape candidate latency across authoritative ABBA repeats."""
    values: dict[str, list[float]] = {}
    measured_runs = 0
    if verification is None:
        return {}, measured_runs
    for run in verification.runs:
        if run.revision != "candidate" or not isinstance(run.result, dict):
            continue
        by_shape = run.result.get("latency_us_by_shape")
        if not isinstance(by_shape, dict):
            continue
        measured_runs += 1
        for shape_id, raw_value in by_shape.items():
            if (
                isinstance(raw_value, (int, float))
                and not isinstance(raw_value, bool)
                and raw_value > 0
                and math.isfinite(float(raw_value))
            ):
                values.setdefault(str(shape_id), []).append(float(raw_value))
    return (
        {
            shape_id: (
                samples[0]
                if len(samples) == 1
                else math.exp(
                    sum(math.log(value) for value in samples) / len(samples)
                )
            )
            for shape_id, samples in values.items()
            if samples
        },
        measured_runs,
    )


def _positive_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _latest_complete_canonical_performance(
    workspace: Path,
    *,
    before_version: int,
    expected_shape_ids: set[str] | None,
) -> tuple[dict[str, Any], int] | None:
    """Find the newest complete real-shape incumbent measurement before a round.

    A pivot, blocked session, protocol failure, or supervisor interruption may have no
    candidate verification at all.  Its memory must still carry the real per-shape
    performance of the unchanged canonical incumbent rather than replacing known facts
    with nulls.  Only a complete, finite canonical record is eligible for carry-forward.
    """
    for version in range(before_version - 1, -1, -1):
        memory = main_adapter.read_memory(workspace, version)
        if not isinstance(memory, dict):
            continue
        performance = memory.get("performance")
        if not isinstance(performance, dict):
            continue
        by_shape = performance.get("latency_us_by_shape")
        if not isinstance(by_shape, dict) or not by_shape:
            continue
        normalized: dict[str, float] = {}
        for shape_id, raw_value in by_shape.items():
            value = _positive_finite(raw_value)
            if value is None:
                normalized = {}
                break
            normalized[str(shape_id)] = value
        if not normalized:
            continue
        if expected_shape_ids is not None and set(normalized) != expected_shape_ids:
            continue
        latency = _positive_finite(
            performance.get("latency_us_geomean", performance.get("latency_us"))
        )
        if latency is None:
            continue
        carried = dict(performance)
        carried["latency_us"] = latency
        carried["latency_us_geomean"] = latency
        carried["latency_us_arith_mean"] = _positive_finite(
            performance.get("latency_us_arith_mean")
        ) or (sum(normalized.values()) / len(normalized))
        carried["latency_us_by_shape"] = normalized
        return carried, version
    return None


def _latest_complete_episode_performance(
    episode_workspace: Path | None,
    *,
    expected_shape_ids: set[str] | None,
) -> dict[str, Any] | None:
    """Read the latest complete supervisor-independent measurement from an episode."""
    if episode_workspace is None:
        return None
    path = episode_workspace / EPISODE_EVALUATIONS_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        kernel_path = episode_workspace / "kernel.py"
        kernel_bytes = kernel_path.read_bytes()
        kernel_sha256 = hashlib.sha256(kernel_bytes).hexdigest()
        kernel_mtime = kernel_path.stat().st_mtime
    except (OSError, UnicodeError):
        return None
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or not result.get("all_pass"):
            continue
        schema_version = payload.get("schema_version")
        recorded_kernel_sha256 = payload.get("kernel_sha256")
        if isinstance(schema_version, int) and schema_version >= 2:
            if recorded_kernel_sha256 != kernel_sha256:
                continue
        else:
            # Schema v1 predates code fingerprints.  It is safe only when the
            # current kernel has not been modified since this measurement.
            try:
                measured_at = _iso_timestamp(str(payload["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if measured_at < kernel_mtime:
                continue
        by_shape = result.get("latency_us_by_shape")
        if not isinstance(by_shape, dict) or not by_shape:
            continue
        normalized: dict[str, float] = {}
        for shape_id, raw_value in by_shape.items():
            value = _positive_finite(raw_value)
            if value is None:
                normalized = {}
                break
            normalized[str(shape_id)] = value
        if not normalized:
            continue
        if expected_shape_ids is not None and set(normalized) != expected_shape_ids:
            continue
        latency = _positive_finite(
            result.get("latency_us_geomean", result.get("latency_us"))
        )
        if latency is None:
            continue
        return {
            "all_pass": True,
            "latency_us_geomean": latency,
            "latency_us_arith_mean": _positive_finite(
                result.get("latency_us_arith_mean")
            )
            or (sum(normalized.values()) / len(normalized)),
            "latency_us_by_shape": normalized,
            "speedup_vs_ref_geomean": result.get("speedup_vs_ref_geomean"),
            "max_abs_err": result.get("max_abs_err"),
            "max_rel_err": result.get("max_rel_err"),
            "eval_id": result.get("eval_id"),
            "timestamp": payload.get("timestamp"),
        }
    return None


def _memory_experiment_value(value: object) -> str:
    """Render one journal value into bounded canonical-memory text."""
    if isinstance(value, str):
        rendered = value.strip()
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = repr(value)
    if len(rendered) > MAX_MEMORY_EXPERIMENT_FIELD_CHARS:
        return rendered[:MAX_MEMORY_EXPERIMENT_FIELD_CHARS] + "… [truncated]"
    return rendered


def _memory_experience(journal: dict[str, Any]) -> dict[str, Any]:
    """Preserve every decisive experiment as a compact canonical-memory record."""
    raw_experiments = journal.get("experiments")
    if not isinstance(raw_experiments, list):
        raw_experiments = []
    experiments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_experiments, start=1):
        if not isinstance(raw, dict):
            continue
        compact: dict[str, Any] = {"index": index}
        for field in MEMORY_EXPERIMENT_FIELDS:
            if field not in raw:
                continue
            rendered = _memory_experiment_value(raw[field])
            if rendered:
                compact[field] = rendered
        extra = {
            key: value
            for key, value in raw.items()
            if key not in MEMORY_EXPERIMENT_FIELDS
        }
        if extra:
            compact["details"] = _memory_experiment_value(extra)
        experiments.append(compact)
    return {
        "experiment_count": len(raw_experiments),
        "recorded_experiment_count": len(experiments),
        "experiments": experiments,
    }


@dataclass
class LongHorizonCampaign:
    base_campaign: main_adapter.Campaign
    max_episodes: int = 8
    max_version: int | None = None
    episode_limit: int = 0
    token_budget: int = 0
    handoff_resumes: int = 2
    max_stall: int = 0
    verifier: GatewayABBAValidator | None = None
    session_runner: LongSessionRunner | None = None
    worktree_root: Path | None = None

    @property
    def workspace(self) -> Path:
        return self.base_campaign.workspace

    def _prompt(
        self,
        *,
        episode: int,
        version: int,
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff_path: Path,
        live_memory_path: Path,
        conversion_pending: bool,
    ) -> str:
        directives = main_adapter.episode_directives(self.base_campaign, version)
        journal_command = (
            f"PYTHONPATH={MODULE_ROOT} python -m long_horizon.journal "
            f"--live-path {json.dumps(str(live_memory_path))}"
        )
        return _render(
            PROMPT_PATH.read_text(encoding="utf-8"),
            {
                "EPISODE": episode,
                "VERSION": version,
                "WORKSPACE": worktree.path,
                "PLATFORM": self.base_campaign.platform,
                "FRAMEWORK": self.base_campaign.framework,
                "BASE_COMMIT": worktree.base_commit,
                "EPISODE_BRANCH": worktree.branch,
                "JOURNAL_PATH": journal_path,
                "JOURNAL_PATH_SHELL": json.dumps(str(journal_path)),
                "HANDOFF_PATH": handoff_path,
                "NOTES": self.base_campaign.notes,
                "MODE_POLICY": directives["mode_policy"],
                "EVALUATOR": directives["evaluator"],
                "HARDWARE": directives["hardware"],
                "SANDBOX": directives["sandbox"],
                "AGENT_RUNTIME": directives["agent_runtime"],
                "PLAN_GENERATOR": directives["plan_generator"],
                "JOURNAL_COMMAND": journal_command,
                "CONVERSION_DIRECTIVE": (
                    "This episode is a mandatory Triton-to-Gluon conversion attempt. Do not "
                    "submit another Triton kernel. A candidate must be a committed Gluon kernel, "
                    f"pass correctness, and stay within {main_adapter.CONVERT_PERF_TOL:.0%} of "
                    "the incumbent latency."
                    if conversion_pending
                    else "No mandatory framework conversion is currently latched."
                ),
            },
        )

    def _completion_check(
        self,
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff: EpisodeHandoff,
    ) -> str:
        candidate = (
            handoff.candidate_commit if handoff.status == "candidate_ready" else ""
        )
        diagnosis = validate_terminal(
            journal_path,
            expected_episode=worktree.episode,
            base_commit=worktree.base_commit,
            branch=worktree.branch,
            state=handoff.status,
            candidate_commit=candidate,
        )
        if diagnosis:
            return diagnosis
        if handoff.status != "candidate_ready":
            return ""
        violation, _ = worktree.validate_candidate(candidate)
        if violation:
            return violation
        try:
            journal = load_journal(journal_path)
            finalized_at = _iso_timestamp(str(journal["finalized_at"]))
            committed_at = float(
                git_text(worktree.path, "show", "-s", "--format=%ct", candidate)
            )
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return f"cannot validate terminal journal ordering: {exc}"
        if finalized_at <= committed_at:
            return (
                "candidate journal must be finalized after the exact candidate commit"
            )
        return ""

    def _copy_runtime_artifacts(
        self, worktree: EpisodeWorktree, episode_dir: Path
    ) -> None:
        source = worktree.path / RUNTIME_DIR
        if source.is_dir():
            destination = episode_dir / "episode_runtime"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        verification = worktree.path / VERIFY_DIR
        if verification.is_dir():
            destination = episode_dir / "verification_runtime"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(verification, destination)

    def _memory_record(
        self,
        *,
        version: int,
        candidate_commit: str,
        journal: dict[str, Any],
        verification: VerificationResult,
    ) -> dict[str, Any]:
        representative = _representative_candidate_result(verification)
        by_shape, shape_measurement_repeats = _candidate_shape_latencies(verification)
        if self.base_campaign.private_reference_dir is not None:
            expected_shapes = set(
                json.loads(
                    (self.base_campaign.private_reference_dir / "shapes.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            measured_shapes = set(by_shape) if isinstance(by_shape, dict) else set()
            if measured_shapes != expected_shapes:
                raise RuntimeError(
                    "authoritative candidate memory lacks complete hidden-shape performance "
                    f"coverage ({len(measured_shapes)}/{len(expected_shapes)})"
                )
        outcome = (
            journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
        )
        directions = (
            outcome.get("next_directions", []) if isinstance(outcome, dict) else []
        )
        return {
            "version": f"v{version}",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "latency_us": verification.candidate_latency_us,
                "latency_us_geomean": verification.candidate_latency_us,
                "latency_us_arith_mean": representative.get(
                    "latency_us_arith_mean", verification.candidate_latency_us
                ),
                "latency_us_by_shape": by_shape if isinstance(by_shape, dict) else {},
                "measurement_scope": "real_evaluator_shapes",
                "shape_ids_are_opaque": self.base_campaign.private_reference_dir
                is not None,
                "measurement_status": "complete",
                "measured_shape_count": len(by_shape),
                "expected_shape_count": (
                    len(expected_shapes)
                    if self.base_campaign.private_reference_dir is not None
                    else None
                ),
                "shape_measurement_repeats": shape_measurement_repeats,
                "measurement_subject": "candidate",
                "measurement_source": "authoritative_verification",
                "carried_from_version": None,
                "speedup_vs_ref_geomean": main_adapter.speedup_vs_reference(
                    self.workspace,
                    verification.candidate_latency_us,
                    representative.get("speedup_vs_ref_geomean"),
                ),
                "tflops_peak_utilization_pct": representative.get(
                    "tflops_peak_utilization_pct"
                ),
                "bandwidth_peak_utilization_pct": representative.get(
                    "bandwidth_peak_utilization_pct"
                ),
                "authoritative_improvement_pct": verification.improvement_pct,
            },
            "optimization": {
                "action_category": "long_horizon_episode",
                "action_description": str(
                    outcome.get("summary", "verified long-horizon candidate")
                ),
                "expected_impact": "independently verified incumbent/candidate latency reduction",
                "risks_and_rollback": "candidate retained on isolated episode branch",
            },
            "profile_evidence": {
                "tool_used": "episode-owned profiler evidence plus supervisor ABBA",
                "evidence_summary": f"{len(journal.get('experiments', []))} structured experiments",
                "bottleneck_type": "episode-derived",
                "evidence_chain": "episode evidence -> candidate -> independent ABBA -> promotion",
            },
            "experience": _memory_experience(journal),
            "correctness": {
                "status": "PASS",
                "max_abs_err": representative.get("max_abs_err", 0.0),
                "max_rel_err": representative.get("max_rel_err", 0.0),
            },
            "quality_gate": {"result": "PASS", "failure_reason": None},
            "open_directions": [
                {
                    "direction": value,
                    "rationale": "carried from terminal episode journal",
                }
                for value in directions
                if isinstance(value, str)
            ],
            "git_commit_hash": candidate_commit,
        }

    def _outcome_memory_record(
        self,
        *,
        version: int,
        status: str,
        violation: str,
        journal: dict[str, Any],
        candidate_commit: str,
        verification: VerificationResult | None = None,
        episode_workspace: Path | None = None,
    ) -> dict[str, Any]:
        outcome = (
            journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
        )
        directions = (
            outcome.get("next_directions", []) if isinstance(outcome, dict) else []
        )
        verification_failure = ""
        if verification is not None and not verification.passed:
            verification_failure = verification.error or (
                f"authoritative verification gate {verification.gate} did not pass"
            )
        failure = (
            violation
            or verification_failure
            or str(outcome.get("summary", status))
        )
        representative = _representative_candidate_result(verification)
        by_shape, shape_measurement_repeats = _candidate_shape_latencies(verification)
        expected_shape_ids: set[str] | None = None
        if self.base_campaign.private_reference_dir is not None:
            expected_shape_ids = set(
                json.loads(
                    (self.base_campaign.private_reference_dir / "shapes.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        expected_shape_count = (
            len(expected_shape_ids) if expected_shape_ids is not None else None
        )
        measured_shape_count = len(by_shape)
        if (
            representative.get("all_pass")
            and expected_shape_ids is not None
            and set(by_shape) != expected_shape_ids
        ):
            raise RuntimeError(
                "correct candidate outcome lacks complete hidden-shape performance "
                f"coverage ({measured_shape_count}/{expected_shape_count})"
            )
        measurement_complete = bool(
            representative.get("all_pass")
            and (
                expected_shape_count is None
                or measured_shape_count == expected_shape_count
            )
        )
        measurement_subject = "candidate" if measurement_complete else "unavailable"
        measurement_source = (
            "authoritative_verification" if measurement_complete else "none"
        )
        carried_from_version: int | None = None
        carried: tuple[dict[str, Any], int] | None = None
        if not measurement_complete:
            episode_performance = _latest_complete_episode_performance(
                episode_workspace,
                expected_shape_ids=expected_shape_ids,
            )
            if episode_performance is not None:
                representative = episode_performance
                by_shape = dict(episode_performance["latency_us_by_shape"])
                shape_measurement_repeats = 1
                measured_shape_count = len(by_shape)
                measurement_complete = True
                measurement_subject = "episode_head"
                measurement_source = "episode_evaluator_result"
            else:
                carried = _latest_complete_canonical_performance(
                    self.workspace,
                    before_version=version,
                    expected_shape_ids=expected_shape_ids,
                )
            if not measurement_complete and carried is not None:
                incumbent_performance, carried_from_version = carried
                representative = {
                    "all_pass": True,
                    "latency_us_geomean": incumbent_performance.get(
                        "latency_us_geomean"
                    ),
                    "latency_us_arith_mean": incumbent_performance.get(
                        "latency_us_arith_mean"
                    ),
                    "speedup_vs_ref_geomean": incumbent_performance.get(
                        "speedup_vs_ref_geomean"
                    ),
                }
                by_shape = dict(incumbent_performance["latency_us_by_shape"])
                shape_measurement_repeats = int(
                    incumbent_performance.get("shape_measurement_repeats") or 0
                )
                measured_shape_count = len(by_shape)
                measurement_complete = True
                measurement_subject = "incumbent"
                measurement_source = "canonical_incumbent_carry_forward"
        return {
            "version": f"v{version}",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "latency_us": representative.get("latency_us_geomean"),
                "latency_us_geomean": representative.get("latency_us_geomean"),
                "latency_us_arith_mean": representative.get("latency_us_arith_mean"),
                "latency_us_by_shape": by_shape,
                "measurement_scope": "real_evaluator_shapes",
                "shape_ids_are_opaque": self.base_campaign.private_reference_dir
                is not None,
                "measurement_status": "complete"
                if measurement_complete
                else "not_evaluated_or_incomplete",
                "measured_shape_count": measured_shape_count,
                "expected_shape_count": expected_shape_count,
                "shape_measurement_repeats": shape_measurement_repeats,
                "measurement_subject": measurement_subject,
                "measurement_source": measurement_source,
                "carried_from_version": (
                    f"v{carried_from_version}"
                    if carried_from_version is not None
                    else None
                ),
                "speedup_vs_ref_geomean": main_adapter.speedup_vs_reference(
                    self.workspace,
                    representative.get("latency_us_geomean"),
                    representative.get("speedup_vs_ref_geomean"),
                ),
            },
            "optimization": {
                "action_category": "long_horizon_episode",
                "action_description": str(outcome.get("summary", status)),
                "expected_impact": "episode exploration did not produce a promotable improvement",
                "risks_and_rollback": "incumbent kernel was preserved",
            },
            "profile_evidence": {
                "tool_used": "episode journal",
                "evidence_summary": f"{len(journal.get('experiments', []))} structured experiments",
                "bottleneck_type": "episode-derived",
                "evidence_chain": "episode evidence -> terminal handoff -> no promotion",
            },
            "experience": _memory_experience(journal),
            "correctness": {
                "status": (
                    "PASS" if measurement_complete else ("FAIL" if violation else "UNKNOWN")
                ),
                "max_abs_err": representative.get("max_abs_err"),
                "max_rel_err": representative.get("max_rel_err"),
            },
            "quality_gate": {"result": "FAIL", "failure_reason": failure},
            "open_directions": [
                {
                    "direction": value,
                    "rationale": "carried from terminal episode journal",
                }
                for value in directions
                if isinstance(value, str)
            ],
            "git_commit_hash": None,
            "long_horizon": {
                "status": status,
                "candidate_commit": candidate_commit or None,
            },
        }

    @staticmethod
    def _valid_blocked_attempt(attempt: object) -> bool:
        return (
            isinstance(attempt, dict)
            and attempt.get("status") == "blocked"
            and not attempt.get("violation")
        )

    def _blocked_retry_pending(self, state: SupervisorState) -> bool:
        if not state.attempts:
            return False
        attempt = state.attempts[-1]
        return self._valid_blocked_attempt(attempt) and not bool(
            attempt.get("blocked_terminal")
        )

    def _terminal_blocked_attempt(
        self, state: SupervisorState
    ) -> dict[str, Any] | None:
        if not state.attempts:
            return None
        attempt = state.attempts[-1]
        if (
            self._valid_blocked_attempt(attempt)
            and attempt.get("blocked_terminal") is True
        ):
            return attempt
        return None

    @staticmethod
    def _load_recovery_journal(
        store: CampaignStore, episode: int, worktree_path: Path | None
    ) -> dict[str, Any]:
        candidates: list[Path] = []
        if worktree_path is not None:
            candidates.append(worktree_path / RUNTIME_DIR / "journal.json")
        candidates.append(
            store.episode_dir(episode) / "episode_runtime" / "journal.json"
        )
        for path in candidates:
            try:
                return load_journal(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                continue
        return {}

    def _recover_interrupted(
        self, store: CampaignStore, state: SupervisorState
    ) -> None:
        active = store.load_active()
        if active is None:
            return
        episode = int(active.get("episode", 0))
        base_commit = str(active.get("base_commit", ""))
        branch = str(active.get("episode_branch", ""))
        worktree_value = active.get("worktree")
        worktree_path = (
            Path(worktree_value).resolve()
            if isinstance(worktree_value, str) and worktree_value.strip()
            else None
        )
        phase = str(active.get("phase", ""))
        memory_version = int(active.get("memory_version", 0) or 0)
        terminal_status = str(active.get("terminal_status", ""))
        already_recorded = any(
            attempt.get("episode") == episode
            and attempt.get("episode_branch") == branch
            for attempt in state.attempts
        )
        if git_head(self.workspace) != base_commit:
            message = git_text(self.workspace, "log", "-1", "--format=%s", check=False)
            parent = git_text(self.workspace, "rev-parse", "HEAD^", check=False)
            evidence = git_text(
                self.workspace,
                "show",
                f"HEAD:memory/long_horizon_e{episode:04d}.json",
                check=False,
            )
            promoted = (
                phase in {"promoting", "promoted"}
                and parent == base_commit
                and message
                == f"episode {episode}: promote verified long-horizon candidate"
                and bool(evidence)
            )
            outcome_recorded = (
                phase in {"recording", "recorded"}
                and memory_version > 0
                and bool(terminal_status)
                and parent == base_commit
                and message
                == f"v{memory_version}: long-horizon episode {episode} {terminal_status}"
                and bool(
                    git_text(
                        self.workspace,
                        "show",
                        f"HEAD:memory/v{memory_version}.json",
                        check=False,
                    )
                )
            )
            if not (promoted or outcome_recorded):
                raise RuntimeError(
                    "incumbent advanced during an interrupted episode without proof"
                )
            if not already_recorded:
                state.episodes = max(state.episodes, episode)
                recovered_attempt: dict[str, Any] = {
                    "episode": episode,
                    "version": memory_version,
                    "status": "candidate_ready" if promoted else terminal_status,
                    "accepted": promoted,
                    "violation": None,
                    "base_commit": base_commit,
                    "episode_branch": branch,
                    "recovered_after_supervisor_interruption": True,
                }
                if promoted:
                    state.accepted += 1
                    state.consecutive_without_promotion = 0
                    recovered_attempt["promotion_commit"] = git_head(self.workspace)
                else:
                    state.consecutive_without_promotion += 1
                    recovered_attempt["outcome_commit"] = git_head(self.workspace)
                    if terminal_status == "pivot":
                        state.pivoted += 1
                    elif terminal_status == "blocked":
                        state.blocked += 1
                        retry_of = (
                            state.attempts[-1]
                            if self._blocked_retry_pending(state)
                            else None
                        )
                        recovered_attempt["blocked_retry_scheduled"] = retry_of is None
                        recovered_attempt["blocked_terminal"] = retry_of is not None
                        if retry_of is not None:
                            recovered_attempt["blocked_retry_of_episode"] = (
                                retry_of.get("episode")
                            )
                    elif terminal_status == "interrupted":
                        state.interrupted += 1
                        recovered_attempt["violation"] = "supervisor process interrupted"
                    elif terminal_status == "invalid_handoff":
                        state.protocol_failures += 1
                    else:
                        state.rejected += 1
                state.attempts.append(recovered_attempt)
                store.archive_attempt(episode, recovered_attempt)
        else:
            # A crash during squash promotion can leave the incumbent index/worktree dirty
            # while HEAD still points at the immutable base. The active marker proves these
            # are supervisor-owned partial changes, so roll them back before continuing.
            if working_changes(self.workspace):
                subprocess.run(
                    ["git", "reset", "--hard", base_commit],
                    cwd=str(self.workspace),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            registered = {
                Path(line.split(" ", 1)[1]).resolve()
                for line in git_text(
                    self.workspace, "worktree", "list", "--porcelain"
                ).splitlines()
                if line.startswith("worktree ")
            }
            if (
                worktree_path is not None
                and worktree_path != self.workspace.resolve()
                and worktree_path.is_dir()
                and worktree_path in registered
                and branch
            ):
                worktree = EpisodeWorktree(episode, base_commit, branch, worktree_path)
                episode_dir = store.episode_dir(episode)
                if not already_recorded:
                    worktree.archive(episode_dir / "interrupted_archive")
                    self._copy_runtime_artifacts(worktree, episode_dir)
            if memory_version <= 0:
                raise RuntimeError("interrupted episode has no canonical memory version")
            journal = self._load_recovery_journal(store, episode, worktree_path)
            outcome = (
                journal.get("outcome")
                if isinstance(journal.get("outcome"), dict)
                else {}
            )
            candidate_commit = str(journal.get("candidate_commit") or "")
            active["phase"] = "recording"
            active["terminal_status"] = "interrupted"
            store.save_active(active)
            memory = self._outcome_memory_record(
                version=memory_version,
                status="interrupted",
                violation="supervisor process interrupted",
                journal=journal,
                candidate_commit=candidate_commit,
                episode_workspace=worktree_path,
            )
            outcome_commit = record_episode_outcome(
                self.workspace,
                base_commit=base_commit,
                version=memory_version,
                episode=episode,
                status="interrupted",
                memory_record=memory,
            )
            active["phase"] = "recorded"
            active["outcome_commit"] = outcome_commit
            store.save_active(active)
            attempt = {
                "episode": episode,
                "version": memory_version,
                "status": "interrupted",
                "accepted": False,
                "violation": "supervisor process interrupted",
                "base_commit": base_commit,
                "episode_branch": branch,
                "candidate_commit": candidate_commit or None,
                "summary": outcome.get("summary"),
                "next_directions": outcome.get("next_directions"),
                "outcome_commit": outcome_commit,
                "recovered_after_supervisor_interruption": True,
            }
            if not already_recorded:
                state.episodes = max(state.episodes, episode)
                state.interrupted += 1
                state.consecutive_without_promotion += 1
                state.attempts.append(attempt)
            else:
                for existing in state.attempts:
                    if (
                        existing.get("episode") == episode
                        and existing.get("episode_branch") == branch
                    ):
                        existing.update(attempt)
                        break
            store.archive_attempt(episode, attempt)
            try:
                sync_live_memory(
                    store.live_memory_path,
                    journal,
                    phase="recorded",
                    canonical_memory=f"memory/v{memory_version}.json",
                    accepted=False,
                    memory_version=memory_version,
                    episode=episode,
                )
            except OSError:
                pass
        if worktree_path is not None and worktree_path != self.workspace.resolve():
            registered = {
                Path(line.split(" ", 1)[1]).resolve()
                for line in git_text(
                    self.workspace, "worktree", "list", "--porcelain"
                ).splitlines()
                if line.startswith("worktree ")
            }
            if worktree_path in registered:
                EpisodeWorktree(
                    episode, base_commit, branch or "atrex/recovery", worktree_path
                ).remove(self.workspace)
        main_adapter.save_stall(self.workspace, state.consecutive_without_promotion)
        store.save_state(state)
        store.clear_active()

    def run(self) -> str:
        main_adapter.prepare_campaign(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        if state.episodes == 0 and state.consecutive_without_promotion == 0:
            state.consecutive_without_promotion = main_adapter.restored_stall(
                self.workspace
            )
        self._recover_interrupted(store, state)
        if working_changes(self.workspace):
            raise RuntimeError(
                "long-horizon campaign requires a clean incumbent workspace: "
                + ", ".join(working_changes(self.workspace)[:12])
            )
        terminal_block = self._terminal_blocked_attempt(state)
        if terminal_block is not None:
            reason = "blocked"
            print(
                f"[long-horizon] STOP {reason}; episodes={state.episodes} "
                f"accepted={state.accepted} rejected={state.rejected} "
                f"pivoted={state.pivoted} blocked={state.blocked} "
                f"protocol_failures={state.protocol_failures} tokens={state.tokens}",
                flush=True,
            )
            return reason
        verifier = self.verifier or GatewayABBAValidator(
            hardware=self.base_campaign.sandbox_hardware,
            profile=self.base_campaign.sandbox_profile,
            url=self.base_campaign.sandbox_url,
            timeout=self.base_campaign.sandbox_timeout,
            private_reference_dir=self.base_campaign.private_reference_dir,
        )
        runner = self.session_runner or LongSessionRunner(
            agent_cli=getattr(self.base_campaign, "agent_cli", "claude")
        )
        starting_episodes = state.episodes
        reason = "budget: max-iters" if self.max_version is not None else "max-episodes"

        while True:
            blocked_retry_pending = self._blocked_retry_pending(state)
            conversion_pending = main_adapter.conversion_required(
                self.base_campaign, state.consecutive_without_promotion, self.workspace
            )
            if self.max_version is not None and not blocked_retry_pending:
                if main_adapter.latest_version(self.workspace) >= self.max_version:
                    if conversion_pending:
                        raise RuntimeError(
                            "mandatory Triton->Gluon conversion did not succeed before max-iters"
                        )
                    reason = "budget: max-iters"
                    break
            elif (
                self.max_version is None
                and state.episodes >= self.max_episodes
                and not blocked_retry_pending
            ):
                reason = "max-episodes"
                break
            if (
                self.episode_limit
                and state.episodes - starting_episodes >= self.episode_limit
                and not blocked_retry_pending
            ):
                reason = "episode-limit"
                break
            if (
                self.token_budget
                and state.tokens >= self.token_budget
                and not blocked_retry_pending
            ):
                if conversion_pending:
                    raise RuntimeError(
                        "mandatory Triton->Gluon conversion did not succeed before token-budget"
                    )
                reason = "budget: token-budget"
                break
            episode = state.episodes + 1
            memory_version = main_adapter.latest_version(self.workspace) + 1
            base_commit = git_head(self.workspace)
            worktree = EpisodeWorktree.plan(
                self.workspace, episode, base_commit, root=self.worktree_root
            )
            active = {
                "episode": episode,
                "memory_version": memory_version,
                "base_commit": base_commit,
                "episode_branch": worktree.branch,
                "worktree": str(worktree.path),
                "phase": "preparing",
            }
            store.save_active(active)
            worktree.materialize(self.workspace)
            active.update(
                {
                    "episode_branch": worktree.branch,
                    "worktree": str(worktree.path),
                    "phase": "exploring",
                }
            )
            store.save_active(active)
            main_adapter.link_episode_runtime(self.base_campaign, worktree.path)
            unexpected = working_changes(worktree.path)
            if unexpected:
                raise RuntimeError(
                    "runtime linking dirtied the episode boundary: "
                    + ", ".join(unexpected)
                )
            runtime = worktree.path / RUNTIME_DIR
            journal_path = runtime / "journal.json"
            handoff_path = runtime / "handoff.json"
            initialize_journal(
                journal_path,
                episode=episode,
                memory_version=memory_version,
                base_commit=base_commit,
                branch=worktree.branch,
                live_path=store.live_memory_path,
            )
            prompt = self._prompt(
                episode=episode,
                version=memory_version,
                worktree=worktree,
                journal_path=journal_path,
                handoff_path=handoff_path,
                live_memory_path=store.live_memory_path,
                conversion_pending=conversion_pending,
            )
            store.write_brief(episode, prompt)
            telemetry_environment = {
                "ATREX_TELEMETRY_TRACE": str(runtime / "telemetry.jsonl"),
                "ATREX_TELEMETRY_CAMPAIGN_ID": str(
                    getattr(self.base_campaign, "campaign_name", self.workspace.name)
                ),
                "ATREX_TELEMETRY_ITERATION_ID": f"episode-{episode:04d}",
                "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
            }
            telemetry_environment.update(self.base_campaign.agent_environment())
            result = runner.run(
                worktree.path,
                prompt,
                handoff_path=handoff_path,
                handoff_resumes=self.handoff_resumes,
                completion_check=lambda handoff: self._completion_check(
                    worktree, journal_path, handoff
                ),
                telemetry_environment=telemetry_environment,
            )
            state.episodes = episode
            state.tokens += result.tokens
            handoff = result.handoff
            status = handoff.status if handoff else "invalid_handoff"
            violation = ""
            candidate_commit = handoff.candidate_commit if handoff else ""
            paths: list[str] = []
            verification: VerificationResult | None = None
            accepted = False
            if result.exit_status != 0 or result.timed_out:
                violation = f"session failed: exit={result.exit_status} timeout={result.timed_out}"
            elif handoff is None:
                violation = (
                    result.completion_diagnosis
                    or "session produced no valid terminal handoff"
                )
            elif status == "candidate_ready":
                violation, paths = worktree.validate_candidate(candidate_commit)
                if (
                    not violation
                    and conversion_pending
                    and not main_adapter.candidate_is_gluon(worktree.path)
                ):
                    violation = (
                        "mandatory conversion candidate is not a committed Gluon kernel"
                    )
                if not violation:
                    policy_violations = main_adapter.candidate_policy_violations(
                        self.base_campaign, worktree.path
                    )
                    if policy_violations:
                        violation = (
                            "production policy rejected candidate: "
                            + "; ".join(policy_violations)
                        )
                if not violation:
                    active["phase"] = "verifying"
                    store.save_active(active)
                    verification = verifier.verify(
                        worktree.path,
                        base_commit=base_commit,
                        candidate_commit=candidate_commit,
                        changed_paths=[
                            path
                            for path in paths
                            if not path.startswith(EVIDENCE_PREFIXES)
                        ],
                    )
                    if (
                        conversion_pending
                        and not verification.passed
                        and _conversion_parity_passes(verification)
                    ):
                        verification = VerificationResult(
                            "PASS",
                            verification.candidate_latency_us,
                            verification.incumbent_latency_us,
                            verification.improvement_pct,
                            runs=verification.runs,
                            artifact=verification.artifact,
                        )
                    accepted = verification.passed

            episode_dir = store.episode_dir(episode)
            worktree.archive(episode_dir / "archive", "HEAD")
            self._copy_runtime_artifacts(worktree, episode_dir)
            try:
                journal = load_journal(journal_path)
            except Exception:
                journal = {}
            outcome = (
                journal.get("outcome")
                if isinstance(journal.get("outcome"), dict)
                else {}
            )
            attempt = {
                "episode": episode,
                "version": memory_version,
                "status": status,
                "accepted": accepted,
                "violation": violation or None,
                "base_commit": base_commit,
                "episode_branch": worktree.branch,
                "episode_head": git_head(worktree.path),
                "candidate_commit": candidate_commit or None,
                "changed_paths": paths,
                "session_id": result.session_id,
                "resume_count": result.resume_count,
                "tokens": result.tokens,
                "summary": outcome.get("summary")
                if isinstance(outcome, dict)
                else None,
                "next_directions": outcome.get("next_directions")
                if isinstance(outcome, dict)
                else None,
                "verification": verification.as_dict() if verification else None,
            }
            try:
                telemetry = summarize_episode(
                    episode=episode,
                    version=memory_version,
                    status=status,
                    accepted=accepted,
                    control_tokens=result.tokens,
                    resume_count=result.resume_count,
                    invocations=result.invocations,
                )
                telemetry_path = store.archive_telemetry(episode, telemetry)
                attempt["telemetry"] = {
                    "summary": str(telemetry_path.relative_to(store.workspace)),
                    "measurement": telemetry["measurement"],
                    "reason_codes": telemetry["reason_codes"],
                }
            except Exception as exc:
                reason_code = f"telemetry_finalize_failed:{type(exc).__name__}"
                attempt["telemetry"] = {
                    "summary": None,
                    "measurement": "unavailable",
                    "reason_codes": [reason_code],
                }
                print(
                    f"[long-horizon] WARNING: could not finalize episode {episode} "
                    f"telemetry: {reason_code}",
                    flush=True,
                )
            valid_blocked = status == "blocked" and not violation
            if valid_blocked:
                retry_of = (
                    state.attempts[-1] if self._blocked_retry_pending(state) else None
                )
                if retry_of is None:
                    attempt["blocked_retry_scheduled"] = True
                    attempt["blocked_terminal"] = False
                else:
                    attempt["blocked_retry_scheduled"] = False
                    attempt["blocked_terminal"] = True
                    attempt["blocked_retry_of_episode"] = retry_of.get("episode")
            promotion_commit = ""
            outcome_commit = ""
            memory: dict[str, Any] | None = None
            if accepted and verification is not None:
                active["phase"] = "promoting"
                store.save_active(active)
                evidence = {**attempt, "journal": journal}
                memory = self._memory_record(
                    version=memory_version,
                    candidate_commit=candidate_commit,
                    journal=journal,
                    verification=verification,
                )
                promotion_commit = promote_candidate(
                    self.workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    episode=episode,
                    evidence=evidence,
                    memory_version=memory_version,
                    memory_record=memory,
                )
                attempt["promotion_commit"] = promotion_commit
                state.accepted += 1
                state.consecutive_without_promotion = 0
                main_adapter.save_stall(self.workspace, 0)
                active["phase"] = "promoted"
                active["promotion_commit"] = promotion_commit
                store.save_active(active)
            else:
                active["phase"] = "recording"
                active["terminal_status"] = status
                store.save_active(active)
                memory = self._outcome_memory_record(
                    version=memory_version,
                    status=status,
                    violation=violation,
                    journal=journal,
                    candidate_commit=candidate_commit,
                    verification=verification,
                    episode_workspace=worktree.path,
                )
                outcome_commit = record_episode_outcome(
                    self.workspace,
                    base_commit=base_commit,
                    version=memory_version,
                    episode=episode,
                    status=status,
                    memory_record=memory,
                )
                attempt["outcome_commit"] = outcome_commit
                state.consecutive_without_promotion += 1
                main_adapter.save_stall(
                    self.workspace, state.consecutive_without_promotion
                )
                if status == "pivot" and not violation:
                    state.pivoted += 1
                elif status == "blocked" and not violation:
                    state.blocked += 1
                elif status == "invalid_handoff":
                    state.protocol_failures += 1
                else:
                    state.rejected += 1
            try:
                sync_live_memory(
                    store.live_memory_path,
                    journal,
                    phase="recorded",
                    canonical_memory=f"memory/v{memory_version}.json",
                    accepted=accepted,
                    memory_version=memory_version,
                    episode=episode,
                )
            except OSError as exc:
                print(
                    "[long-horizon] WARNING: could not update memory/live.json: "
                    f"{type(exc).__name__}",
                    flush=True,
                )
            store.archive_attempt(episode, attempt)
            state.attempts.append(attempt)
            store.save_state(state)
            worktree.remove(self.workspace)
            store.clear_active()
            print(
                f"[long-horizon] episode={episode} status={status} accepted={accepted} "
                f"version=v{memory_version} tokens={result.tokens} "
                f"commit={promotion_commit or outcome_commit or '-'}",
                flush=True,
            )
            if accepted and memory is not None:
                target_util = float(getattr(self.base_campaign, "target_util", 0.0))
                if target_util > 0.0 and main_adapter.peak_util(memory) >= target_util:
                    reason = (
                        f"success: peak_util {main_adapter.peak_util(memory):.1f}% "
                        f">= {target_util:.0f}%"
                    )
                    break
            if valid_blocked:
                if not attempt.get("blocked_terminal"):
                    print(
                        f"[long-horizon] blocked at episode={episode}; starting one fresh "
                        "long-horizon episode retry",
                        flush=True,
                    )
                    continue
                reason = "blocked"
                break
            if (
                self.max_stall
                and state.consecutive_without_promotion >= self.max_stall
                and not main_adapter.conversion_required(
                    self.base_campaign,
                    state.consecutive_without_promotion,
                    self.workspace,
                )
            ):
                reason = f"stall: {state.consecutive_without_promotion} episodes without promotion"
                break

        print(
            f"[long-horizon] STOP {reason}; episodes={state.episodes} accepted={state.accepted} "
            f"rejected={state.rejected} pivoted={state.pivoted} blocked={state.blocked} "
            f"protocol_failures={state.protocol_failures} tokens={state.tokens}",
            flush=True,
        )
        return reason
