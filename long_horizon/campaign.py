from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.hardware import hardware_vendor

from . import main_adapter
from .git_episode import (
    EpisodeWorktree,
    git_blob,
    git_head,
    git_text,
    promote_candidate,
    record_episode_outcome,
    working_changes,
)
from .journal import initialize as initialize_journal
from .journal import load as load_journal
from .journal import normalize_accepted_ppu_diagnostics
from .journal import sync_live_memory, validate_accepted_ppu_evidence, validate_terminal
from .models import (
    EpisodeHandoff,
    SupervisorState,
    VerificationResult,
)
from .protocol import read_handoff
from .session import LongSessionRunner
from .store import RUNTIME_DIR, VERIFY_DIR, CampaignStore
from .telemetry import summarize_episode
from .verifier import GatewayABBAValidator

MODULE_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = MODULE_ROOT / "orchestrator" / "prompts"
PROMPT_PATH = PROMPTS_DIR / "episode.md"
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
EPISODE_REASONING_EFFORT = "max"
GOAL_AFTER_EPISODES = 50
GOAL_AFTER_STALLS = 3
GOAL_HANDOFF_RESUMES = 20


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
    required_performance_objective: str | None = None,
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
        if (memory.get("quality_gate") or {}).get("result") != "PASS":
            continue
        performance = memory.get("performance")
        if not isinstance(performance, dict):
            continue
        if (
            required_performance_objective is not None
            and performance.get("performance_objective")
            != required_performance_objective
        ):
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
        performance_score = _positive_finite(
            performance.get(
                "performance_score", performance.get("speedup_vs_ref_geomean")
            )
        )
        if latency is None or performance_score is None:
            continue
        carried = dict(performance)
        carried["latency_us"] = latency
        carried["latency_us_geomean"] = latency
        carried["latency_us_arith_mean"] = _positive_finite(
            performance.get("latency_us_arith_mean")
        ) or (sum(normalized.values()) / len(normalized))
        carried["latency_us_by_shape"] = normalized
        carried["performance_score"] = performance_score
        return carried, version
    return None


def _latest_complete_episode_performance(
    episode_workspace: Path | None,
    *,
    expected_shape_ids: set[str] | None,
    required_performance_objective: str | None = None,
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
        if (
            required_performance_objective is not None
            and result.get("performance_objective")
            != required_performance_objective
        ):
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
        performance_score = _positive_finite(
            result.get("performance_score", result.get("speedup_vs_ref_geomean"))
        )
        if latency is None or performance_score is None:
            continue
        return {
            "all_pass": True,
            "latency_us_geomean": latency,
            "latency_us_arith_mean": _positive_finite(
                result.get("latency_us_arith_mean")
            )
            or (sum(normalized.values()) / len(normalized)),
            "latency_us_by_shape": normalized,
            "speedup_vs_ref_mean": result.get("speedup_vs_ref_mean"),
            "speedup_vs_ref_geomean": result.get("speedup_vs_ref_geomean"),
            "performance_score": performance_score,
            "performance_objective": result.get("performance_objective"),
            "max_abs_err": result.get("max_abs_err"),
            "max_rel_err": result.get("max_rel_err"),
            "eval_id": result.get("eval_id"),
            "timestamp": payload.get("timestamp"),
        }
    return None


def _episode_head_matches_incumbent(
    workspace: Path, episode_workspace: Path | None
) -> bool:
    if episode_workspace is None:
        return False
    try:
        return (episode_workspace / "kernel.py").read_bytes() == (
            workspace / "kernel.py"
        ).read_bytes()
    except OSError:
        return False


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


def _canonical_ppu_diagnostics(
    journal: dict[str, Any], *, version: int, workspace: Path | None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate optional evidence at the canonical-memory boundary, including recovery."""
    outcome = journal.get("outcome")
    if not isinstance(outcome, dict):
        return [], []
    raw = outcome.get("accepted_ppu_diagnostics", [])
    if not isinstance(raw, list):
        return [], ["accepted_ppu_diagnostics must be a list"]
    episode = journal.get("episode")
    diagnostics: list[dict[str, Any]] = []
    rejected: list[str] = []
    for index, row in enumerate(raw):
        try:
            normalized, errors = normalize_accepted_ppu_diagnostics([row])
            if not errors:
                errors = (
                    validate_accepted_ppu_evidence(normalized, workspace)
                    if workspace is not None
                    else ["episode workspace unavailable"]
                )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            errors = [str(error)]
        if errors:
            rejected.append(f"diagnostic {index}: " + "; ".join(errors))
            continue
        canonical = dict(normalized[0])
        canonical["source_memory_version"] = f"v{version}"
        if isinstance(episode, int) and not isinstance(episode, bool):
            canonical["source_episode"] = episode
        canonical["memory_ref"] = (
            f"memory/v{version}.json#profile_evidence."
            f"accepted_ppu_diagnostics/{len(diagnostics)}"
        )
        evidence = canonical["evidence"]
        canonical["evidence_ref"] = (
            f"{evidence['artifact']}#sha256={evidence['sha256']}"
        )
        diagnostics.append(canonical)
    return diagnostics, rejected


def _memory_profile_evidence(
    journal: dict[str, Any],
    *,
    version: int,
    is_ppu: bool,
    promoted: bool,
    episode_workspace: Path | None,
) -> dict[str, Any]:
    """Build canonical profile memory without changing non-PPU behavior."""
    experiment_count = len(journal.get("experiments", []))
    if not is_ppu:
        return {
            "tool_used": (
                "episode-owned profiler evidence plus supervisor ABBA"
                if promoted
                else "episode journal"
            ),
            "evidence_summary": f"{experiment_count} structured experiments",
            "bottleneck_type": "episode-derived",
            "evidence_chain": (
                "episode evidence -> candidate -> independent ABBA -> promotion"
                if promoted
                else "episode evidence -> terminal handoff -> no promotion"
            ),
        }

    diagnostics, rejected = _canonical_ppu_diagnostics(
        journal, version=version, workspace=episode_workspace
    )
    routes = list(dict.fromkeys(item["route"] for item in diagnostics))
    return {
        "tool_used": (
            f"accepted PPU {' + '.join(routes)} evidence"
            if routes
            else "none recorded (PPU profiling is optional)"
        ),
        "evidence_summary": (
            f"{experiment_count} structured experiments; "
            f"{len(diagnostics)} terminal-reusable PPU diagnostics"
            + ("; excluded: " + "; ".join(rejected) if rejected else "")
        ),
        "bottleneck_type": "episode-derived",
        "evidence_chain": (
            "episode evidence -> terminal handoff; no reusable PPU diagnostics"
            if not diagnostics
            else (
                "terminal-valid PPU evidence -> candidate -> independent ABBA -> promotion"
                if promoted
                else "terminal-valid PPU evidence -> terminal handoff -> no promotion"
            )
        ),
        "accepted_ppu_diagnostics": diagnostics,
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

    @staticmethod
    def _episode_mode(state: SupervisorState, active: dict[str, Any] | None = None) -> str:
        """Goal widens search, never changes the Journal or acceptance contract."""
        if active is not None:
            # Do not widen a running or legacy Episode after a restart.
            return "goal" if active.get("mode") == "goal" else "episode"
        return "goal" if (
            state.episodes >= GOAL_AFTER_EPISODES
            and state.consecutive_without_promotion > GOAL_AFTER_STALLS
        ) else "episode"

    def _expected_shape_ids(self) -> set[str] | None:
        private_reference_dir = self.base_campaign.private_reference_dir
        if private_reference_dir is None:
            return None
        return set(
            json.loads(
                (private_reference_dir / "shapes.json").read_text(encoding="utf-8")
            )
        )


    def _prompt(
        self,
        *,
        episode: int,
        version: int,
        worktree: EpisodeWorktree,
        conversion_pending: bool,
        resumed: bool = False,
        agent_workspace: Path | None = None,
        verifier: GatewayABBAValidator | None = None,
        episode_mode: str = "episode",
    ) -> str:
        directives = main_adapter.episode_directives(
            self.base_campaign, version
        )
        from .recorded_verifier import RecordedABBAValidator
        acceptance_request = ""
        selected_verifier = verifier or self.verifier
        if isinstance(selected_verifier, RecordedABBAValidator):
            context = self.base_campaign.acceptance_measurement_context(
                git_blob(worktree.path, worktree.base_commit, "kernel.py"),
            )
            acceptance_request = selected_verifier.agent_instructions(**context)
        fragments = {
            "RUNTIME_JOURNAL_CONTRACT": "runtime_journal.md",
            "WIKI_DIRECTIVE": "episode_wiki.md",
            "CONVERSION_DIRECTIVE": "",
            "PPU_DIRECTIVE": "",
        }
        if conversion_pending:
            fragments["CONVERSION_DIRECTIVE"] = "episode_conversion.md"
            fragments["WIKI_DIRECTIVE"] = ""  # Conversion has its own initial query.
        if (
            hardware_vendor(self.base_campaign.platform, self.base_campaign.arch) == "ppu"
        ):
            fragments["PPU_DIRECTIVE"] = "episode_ppu.md"
        template = PROMPT_PATH.read_text(encoding="utf-8")
        # Insert fragments before substituting values so nested task placeholders resolve too.
        for marker, name in fragments.items():
            template = template.replace(
                "{{" + marker + "}}",
                (PROMPTS_DIR / name).read_text(encoding="utf-8").strip() if name else "",
            )
        return _render(
            template,
            {
                "EPISODE": episode,
                "GOAL_DIRECTIVE": (
                    "This is a goal-oriented Episode after sustained stalls. Reassess the operator "
                    "roadmap and explore materially different Directions sequentially within the "
                    "Runtime's Direction limits. Preserve measured Kernel IDs so you can restore "
                    "the best validated candidate before reporting. Continue until the useful "
                    "roadmap is completed or exhausted; do not stop at the first failed Direction. "
                    "Journal, correctness, production policy, and ABBA promotion rules are unchanged."
                    if episode_mode == "goal" else ""
                ),
                "VERSION": version,
                "WORKSPACE": agent_workspace or worktree.path,
                "ACCEPTANCE_REQUEST": acceptance_request,
                "OPERATOR": self.base_campaign.name,
                "PLATFORM": self.base_campaign.platform,
                "ARCH": self.base_campaign.arch or "<authoritative runtime architecture>",
                "FRAMEWORK": self.base_campaign.framework,
                "NOTES": self.base_campaign.notes,
                "MODE_POLICY": directives["mode_policy"],
                "EVALUATOR": directives["evaluator"],
                "HARDWARE": directives["hardware"],
                "SANDBOX": directives["sandbox"],
                "AGENT_RUNTIME": directives["agent_runtime"],
                "PLUGINS": directives["plugins"],
                "RESUME_DIRECTIVE": (
                    "This episode is resuming after a supervisor restart. Keep and reuse the "
                    "existing workspace, scratch files, and source "
                    "edits. Inspect them before acting; do not discard them. Use the "
                    "Runtime Journal list/load tools to inspect durable state, then continue the "
                    "in-progress work or repair and resubmit its terminal report."
                    if resumed
                    else "This is a new episode worktree. `scratch/` starts empty; files from "
                    "earlier Episodes are not inherited."
                ),
                "CONVERSION_TOLERANCE": f"{main_adapter.CONVERT_PERF_TOL:.0%}",
            },
        )


    def _require_canonical_memory(self, version: int) -> None:
        """Fail closed unless this episode's memory is valid and committed at HEAD."""
        memory = main_adapter.read_memory(self.workspace, version)
        if not isinstance(memory, dict) or memory.get("version") != f"v{version}":
            raise RuntimeError(f"episode did not write valid canonical memory/v{version}.json")
        committed_text = git_text(
            self.workspace,
            "show",
            f"HEAD:memory/v{version}.json",
            check=False,
        )
        try:
            committed_memory = json.loads(committed_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"episode did not commit canonical memory/v{version}.json"
            ) from exc
        if committed_memory != memory:
            raise RuntimeError(
                f"working canonical memory/v{version}.json differs from committed HEAD"
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
        private_journal = None

        def load_private_runtime_journal() -> dict[str, Any]:
            # Called only for a claimed Runtime projection. During recovery the
            # Runtime/binding may not exist yet; read persisted state without
            # register_episode, which would otherwise create a missing Journal.
            nonlocal private_journal
            private_journal = self.base_campaign.read_runtime_episode_journal(worktree.episode)
            return private_journal

        diagnosis = validate_terminal(
            journal_path,
            expected_episode=worktree.episode,
            base_commit=worktree.base_commit,
            branch=worktree.branch,
            state=handoff.status,
            candidate_commit=candidate,
            runtime_journal_loader=load_private_runtime_journal,
        )
        if diagnosis:
            return diagnosis
        if handoff.status != "candidate_ready":
            return ""
        if private_journal is not None:
            from supervisor.journal import sealed_candidate_source

            try:
                source = sealed_candidate_source(private_journal, worktree.path)
                if source is None:
                    return "private report has no sealed candidate"
                violation, _ = worktree.restore_sealed_candidate(candidate, source)
            except (OSError, ValueError, RuntimeError) as error:
                return f"cannot restore sealed candidate: {error}"
        else:
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
        episode_workspace: Path,
        episode_mode: str = "episode",
    ) -> dict[str, Any]:
        representative = _representative_candidate_result(verification)
        by_shape, shape_measurement_repeats = _candidate_shape_latencies(verification)
        expected_shapes = self._expected_shape_ids()
        if expected_shapes is not None:
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
                # ABBA projections may omit this summary. Derive it from the
                # persisted per-Shape aggregate, not the GeoMean or last repeat.
                "latency_us_arith_mean": (
                    sum(by_shape.values()) / len(by_shape)
                    if by_shape
                    else _positive_finite(representative.get("latency_us_arith_mean"))
                ),
                "latency_us_by_shape": by_shape if isinstance(by_shape, dict) else {},
                "measurement_scope": "real_evaluator_shapes",
                "shape_ids_are_opaque": self.base_campaign.private_reference_dir
                is not None,
                "measurement_status": "complete",
                "measured_shape_count": len(by_shape),
                "expected_shape_count": (
                    len(expected_shapes) if expected_shapes is not None else None
                ),
                "shape_measurement_repeats": shape_measurement_repeats,
                "measurement_subject": "candidate",
                "measurement_source": (
                    "authoritative_verification"
                ),
                "comparison_method": (
                    "same_allocation_abba"
                ),
                "carried_from_version": None,
                "performance_objective": representative.get("performance_objective"),
                "performance_score": verification.candidate_performance_score,
                "speedup_vs_ref_mean": (
                    verification.candidate_performance_score
                    if representative.get("performance_objective")
                    == "shape_speedup_arithmetic_mean"
                    else representative.get("speedup_vs_ref_mean")
                ),
                "speedup_vs_ref_geomean": (
                    None
                    if representative.get("performance_objective")
                    == "shape_speedup_arithmetic_mean"
                    else representative.get("speedup_vs_ref_geomean")
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
                "action_category": (
                    "long_horizon_episode"
                ),
                "action_description": str(
                    outcome.get("summary", "verified long-horizon candidate")
                ),
                "expected_impact": (
                    "independently verified incumbent/candidate latency reduction"
                ),
                "risks_and_rollback": "candidate retained on isolated episode branch",
            },
            "profile_evidence": _memory_profile_evidence(
                journal,
                version=version,
                is_ppu=(
                    hardware_vendor(
                        self.base_campaign.platform, self.base_campaign.arch
                    )
                    == "ppu"
                ),
                promoted=True,
                episode_workspace=episode_workspace,
            ),
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
            "long_horizon": {
                "status": "candidate_ready",
                "mode": episode_mode,
                "verification": "abba",
            },
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
        episode_mode: str = "episode",
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
        # This path records a non-promotion outcome.  Even a correctness-complete
        # candidate result cannot describe the canonical incumbent.
        measurement_complete = False
        measurement_subject = "unavailable"
        measurement_source = "none"
        carried_from_version: int | None = None
        carried: tuple[dict[str, Any], int] | None = None
        if not measurement_complete:
            episode_performance = (
                _latest_complete_episode_performance(
                    episode_workspace,
                    expected_shape_ids=expected_shape_ids,
                    required_performance_objective="shape_speedup_arithmetic_mean",
                )
                if _episode_head_matches_incumbent(
                    self.workspace, episode_workspace
                )
                else None
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
                    required_performance_objective="shape_speedup_arithmetic_mean",
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
                    "performance_objective": incumbent_performance.get(
                        "performance_objective"
                    ),
                    "performance_score": incumbent_performance.get(
                        "performance_score"
                    ),
                    "speedup_vs_ref_mean": incumbent_performance.get(
                        "speedup_vs_ref_mean"
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
                "measurement_status": (
                    "complete"
                    if measurement_complete
                    else "not_evaluated_or_incomplete"
                ),
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
                "performance_objective": representative.get("performance_objective"),
                "performance_score": representative.get("performance_score"),
                "speedup_vs_ref_mean": representative.get("speedup_vs_ref_mean"),
                "speedup_vs_ref_geomean": (
                    None
                    if representative.get("performance_objective")
                    == "shape_speedup_arithmetic_mean"
                    else main_adapter.speedup_vs_reference(
                        self.workspace,
                        representative.get("latency_us_geomean"),
                        representative.get("speedup_vs_ref_geomean"),
                    )
                ),
            },
            "optimization": {
                "action_category": (
                    "long_horizon_episode"
                ),
                "action_description": str(outcome.get("summary", status)),
                "expected_impact": "episode exploration did not produce a promotable improvement",
                "risks_and_rollback": "incumbent kernel was preserved",
            },
            "profile_evidence": _memory_profile_evidence(
                journal,
                version=version,
                is_ppu=(
                    hardware_vendor(
                        self.base_campaign.platform, self.base_campaign.arch
                    )
                    == "ppu"
                ),
                promoted=False,
                episode_workspace=episode_workspace,
            ),
            "experience": _memory_experience(journal),
            "correctness": {
                "status": (
                    "PASS"
                    if measurement_complete
                    else ("FAIL" if violation else "UNKNOWN")
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
                "mode": episode_mode,
            },
        }

    def _assess_terminal_handoff(
        self,
        store: CampaignStore,
        active: dict[str, Any],
        worktree: EpisodeWorktree,
        handoff: EpisodeHandoff,
        *,
        memory_version: int,
        conversion_pending: bool,
        verifier: GatewayABBAValidator,
    ) -> tuple[str, list[str], VerificationResult | None, bool]:
        """Apply the authoritative candidate gates to one terminal handoff."""
        if handoff.status != "candidate_ready":
            return "", [], None, False

        candidate_commit = handoff.candidate_commit
        diagnosis = self._completion_check(
            worktree, worktree.path / RUNTIME_DIR / "journal.json", handoff,
        )
        if diagnosis:
            return diagnosis, [], None, False
        violation, paths = worktree.validate_candidate(candidate_commit)
        if (
            not violation
            and conversion_pending
            and not main_adapter.candidate_is_gluon(worktree.path)
        ):
            violation = "mandatory conversion candidate is not a committed Gluon kernel"
        if not violation:
            policy_violations = main_adapter.candidate_policy_violations(
                self.base_campaign,
                worktree.path,
                require_gluon=(
                    conversion_pending
                    or main_adapter.candidate_is_gluon(self.workspace)
                ),
            )
            if policy_violations:
                violation = (
                    "production policy rejected candidate: "
                    + "; ".join(policy_violations)
                )
        verification: VerificationResult | None = None
        accepted = False
        if not violation:
            active["phase"] = (
                "verifying"
            )
            store.save_active(active)
            verification = verifier.verify(
                worktree.path,
                base_commit=worktree.base_commit,
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
                    candidate_performance_score=(
                        verification.candidate_performance_score
                    ),
                    incumbent_performance_score=(
                        verification.incumbent_performance_score
                    ),
                    gateway_record_id=verification.gateway_record_id,
                    reused=verification.reused,
                )
            accepted = verification.passed
        return violation, paths, verification, accepted

    def _record_terminal_episode(
        self,
        store: CampaignStore,
        state: SupervisorState,
        active: dict[str, Any],
        worktree: EpisodeWorktree,
        *,
        memory_version: int,
        status: str,
        candidate_commit: str,
        violation: str,
        paths: list[str],
        verification: VerificationResult | None,
        accepted: bool,
        session_id: str = "",
        resume_count: int = 0,
        tokens: int = 0,
        tokens_accounted: bool = False,
        invocations: tuple[Any, ...] = (),
        recovered_after_supervisor_interruption: bool = False,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Archive and commit one terminal episode exactly once."""
        episode = worktree.episode
        base_commit = worktree.base_commit
        episode_mode = self._episode_mode(state, active)
        journal_path = worktree.path / RUNTIME_DIR / "journal.json"
        state.episodes = max(state.episodes, episode)
        if not tokens_accounted:
            state.tokens += max(0, int(tokens))

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
            "mode": episode_mode,
            "status": status,
            "accepted": accepted,
            "violation": violation or None,
            "base_commit": base_commit,
            "episode_branch": worktree.branch,
            "episode_head": git_head(worktree.path),
            "candidate_commit": candidate_commit or None,
            "changed_paths": paths,
            "session_id": session_id or None,
            "resume_count": max(0, int(resume_count)),
            "tokens": max(0, int(tokens)),
            "summary": outcome.get("summary")
            if isinstance(outcome, dict)
            else None,
            "next_directions": outcome.get("next_directions")
            if isinstance(outcome, dict)
            else None,
            "verification": verification.as_dict() if verification else None,
        }
        if recovered_after_supervisor_interruption:
            attempt["recovered_after_supervisor_interruption"] = True
            attempt["recovered_terminal_handoff"] = True
        try:
            telemetry = summarize_episode(
                episode=episode,
                version=memory_version,
                status=status,
                accepted=accepted,
                control_tokens=max(0, int(tokens)),
                resume_count=max(0, int(resume_count)),
                invocations=invocations,
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
            attempt["blocked_retry_scheduled"] = True
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
                episode_workspace=worktree.path,
                verification=verification,
                episode_mode=episode_mode,
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
                episode_mode=episode_mode,
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
        self._require_canonical_memory(memory_version)
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
        recovery_label = (
            " recovered=true" if recovered_after_supervisor_interruption else ""
        )
        print(
            f"[long-horizon] episode={episode} mode={episode_mode} "
            f"status={status} accepted={accepted} "
            f"version=v{memory_version} tokens={max(0, int(tokens))} "
            f"commit={promotion_commit or outcome_commit or '-'}{recovery_label}",
            flush=True,
        )
        return memory, valid_blocked

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

    def _recover_completed_handoff(
        self,
        store: CampaignStore,
        state: SupervisorState,
        active: dict[str, Any],
        worktree: EpisodeWorktree,
        *,
        verifier: GatewayABBAValidator,
    ) -> bool:
        """Finish a terminal handoff that arrived after the supervisor exited."""
        runtime = worktree.path / RUNTIME_DIR
        handoff = read_handoff(runtime / "handoff.json")
        if handoff is None:
            return False
        diagnosis = self._completion_check(
            worktree,
            runtime / "journal.json",
            handoff,
        )
        if diagnosis:
            print(
                "[long-horizon] interrupted episode has a terminal handoff but "
                f"cannot recover it: {diagnosis}",
                flush=True,
            )
            return False

        memory_version = int(active.get("memory_version", 0) or 0)
        if memory_version <= 0:
            raise RuntimeError("interrupted episode has no canonical memory version")
        conversion_pending = main_adapter.conversion_required(
            self.base_campaign,
            state.consecutive_without_promotion,
            self.workspace,
        )
        violation, paths, verification, accepted = self._assess_terminal_handoff(
            store,
            active,
            worktree,
            handoff,
            memory_version=memory_version,
            conversion_pending=conversion_pending,
            verifier=verifier,
        )
        self._record_terminal_episode(
            store,
            state,
            active,
            worktree,
            memory_version=memory_version,
            status=handoff.status,
            candidate_commit=handoff.candidate_commit,
            violation=violation,
            paths=paths,
            verification=verification,
            accepted=accepted,
            recovered_after_supervisor_interruption=True,
        )
        return True

    def _recover_interrupted(
        self,
        store: CampaignStore,
        state: SupervisorState,
        *,
        verifier: GatewayABBAValidator,
    ) -> tuple[EpisodeWorktree, dict[str, Any]] | None:
        active = store.load_active()
        if active is None:
            return None
        if active.get("mode") == "fast":
            raise RuntimeError(
                "Cannot resume an unfinished Fast Episode with the unified workflow. "
                "Finish it with the previous release, then upgrade at an Episode boundary."
            )
        episode_mode = self._episode_mode(state, active)
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
        registered = {
            Path(line.split(" ", 1)[1]).resolve()
            for line in git_text(
                self.workspace, "worktree", "list", "--porcelain"
            ).splitlines()
            if line.startswith("worktree ")
        }
        resumable_worktree = bool(
            git_head(self.workspace) == base_commit
            and phase in {
                "preparing",
                "exploring",
                "verifying",
                "promoting",
                "recording",
            }
            and episode > 0
            and memory_version > 0
            and worktree_path is not None
            and worktree_path != self.workspace.resolve()
            and worktree_path.is_dir()
            and worktree_path in registered
            and branch
            and git_text(
                worktree_path,
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
                check=False,
            )
            == branch
            and subprocess.run(
                ["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
                cwd=str(worktree_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        if resumable_worktree:
            assert worktree_path is not None
            if phase == "promoting" and working_changes(self.workspace):
                # The candidate remains authoritative in the episode worktree. Roll back
                # only an incomplete supervisor squash before retrying promotion.
                subprocess.run(
                    ["git", "reset", "--hard", base_commit],
                    cwd=str(self.workspace),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            worktree = EpisodeWorktree(episode, base_commit, branch, worktree_path)
            CampaignStore.ensure_excluded(worktree.path)
            if self._recover_completed_handoff(
                store,
                state,
                active,
                worktree,
                verifier=verifier,
            ):
                return None
            active["resumed_from_phase"] = phase
            active["phase"] = "exploring"
            active["restart_count"] = int(active.get("restart_count", 0) or 0) + 1
            store.save_active(active)
            print(
                f"[long-horizon] resuming interrupted episode={episode} in existing "
                f"worktree {worktree.path} with {len(working_changes(worktree.path))} "
                "visible intermediate path(s)",
                flush=True,
            )
            return worktree, active
        if git_head(self.workspace) != base_commit:
            message = git_text(self.workspace, "log", "-1", "--format=%s", check=False)
            parent = git_text(self.workspace, "rev-parse", "HEAD^", check=False)
            from .promotion_audit import PromotionAuditUnverifiable
            from .audit_recovery import recover_promotion_audit, pause_for_audit_repair
            try:
                audit_recovery = recover_promotion_audit(
                    self.workspace, episode=episode, base_commit=base_commit,
                    branch=branch, version=memory_version,
                )
            except PromotionAuditUnverifiable as error:
                raise pause_for_audit_repair(self.workspace, store, active, error) from error
            promoted = (
                phase in {"promoting", "promoted"}
                and parent == base_commit
                and message
                == f"episode {episode}: promote verified long-horizon candidate"
                and audit_recovery is not None
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
                # The episode never reached a terminal handoff and its base is now
                # stale, so it is superseded rather than rejected on its merits.
                terminal_status = "interrupted"
            self._require_canonical_memory(memory_version)
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
                    "mode": episode_mode,
                    "recovered_after_supervisor_interruption": True,
                }
                if promoted:
                    state.accepted += 1
                    state.consecutive_without_promotion = 0
                    recovered_attempt["promotion_commit"] = git_head(self.workspace)
                    recovered_attempt["promotion_audit"] = audit_recovery
                else:
                    state.consecutive_without_promotion += 1
                    recovered_attempt["outcome_commit"] = git_head(self.workspace)
                    if terminal_status == "pivot":
                        state.pivoted += 1
                    elif terminal_status == "blocked":
                        state.blocked += 1
                        recovered_attempt["blocked_retry_scheduled"] = True
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
            if phase == "promoting" and working_changes(self.workspace):
                subprocess.run(
                    ["git", "reset", "--hard", base_commit],
                    cwd=str(self.workspace),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if (
                worktree_path is not None
                and worktree_path != self.workspace.resolve()
                and worktree_path.is_dir()
                and worktree_path in registered
                and branch
            ):
                worktree = EpisodeWorktree(episode, base_commit, branch, worktree_path)
                if self._recover_completed_handoff(
                    store,
                    state,
                    active,
                    worktree,
                    verifier=verifier,
                ):
                    return
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
                episode_mode=episode_mode,
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
            self._require_canonical_memory(memory_version)
            attempt = {
                "episode": episode,
                "version": memory_version,
                "status": "interrupted",
                "accepted": False,
                "violation": "supervisor process interrupted",
                "base_commit": base_commit,
                "episode_branch": branch,
                "mode": episode_mode,
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
            if worktree_path in registered:
                EpisodeWorktree(
                    episode, base_commit, branch or "atrex/recovery", worktree_path
                ).remove(self.workspace)
        main_adapter.save_stall(self.workspace, state.consecutive_without_promotion)
        store.save_state(state)
        store.clear_active()
        return None

    def run(self) -> str:
        main_adapter.prepare_campaign(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        if state.episodes == 0 and state.consecutive_without_promotion == 0:
            state.consecutive_without_promotion = main_adapter.restored_stall(
                self.workspace
            )
        from .recorded_verifier import RecordedABBAValidator
        verifier = self.verifier or RecordedABBAValidator(
            execute=self.base_campaign.measure_for_acceptance,
            hardware=self.base_campaign.sandbox_hardware,
            profile=self.base_campaign.sandbox_profile,
            url=self.base_campaign.sandbox_url,
            ssh=self.base_campaign.sandbox_ssh,
            ssh_init=self.base_campaign.sandbox_ssh_init,
            health_command=self.base_campaign.sandbox_health_command,
            timeout=self.base_campaign.sandbox_timeout,
            private_reference_dir=self.base_campaign.private_reference_dir,
        )
        recovered_episode = self._recover_interrupted(
            store, state, verifier=verifier
        )
        runner = self.session_runner or LongSessionRunner(
            agent_cli=getattr(self.base_campaign, "agent_cli", "claude")
        )
        starting_episodes = state.episodes
        reason = "budget: max-iters" if self.max_version is not None else "max-episodes"

        while True:
            conversion_pending = main_adapter.conversion_required(
                self.base_campaign, state.consecutive_without_promotion, self.workspace
            )
            if self.max_version is not None:
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
            ):
                reason = "max-episodes"
                break
            if (
                self.episode_limit
                and state.episodes - starting_episodes >= self.episode_limit
            ):
                reason = "episode-limit"
                break
            if (
                self.token_budget
                and state.tokens >= self.token_budget
            ):
                if conversion_pending:
                    raise RuntimeError(
                        "mandatory Triton->Gluon conversion did not succeed before token-budget"
                    )
                reason = "budget: token-budget"
                break
            resumed = recovered_episode is not None
            if recovered_episode is not None:
                worktree, active = recovered_episode
                recovered_episode = None
                episode = worktree.episode
                memory_version = int(active["memory_version"])
                base_commit = worktree.base_commit
                episode_mode = self._episode_mode(state, active)
            else:
                episode = state.episodes + 1
                episode_mode = self._episode_mode(state)


            if resumed:
                active.setdefault("mode", episode_mode)
                if active.get("resumed_from_phase") == "preparing":
                    worktree.reset_scratch()
                    main_adapter.link_episode_runtime(
                        self.base_campaign, worktree.path
                    )
            else:
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
                    "mode": episode_mode,
                    "phase": "preparing",
                }
                store.save_active(active)
                worktree.materialize(self.workspace)
                worktree.reset_scratch()
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
            if not resumed or not journal_path.is_file():
                initialize_journal(
                    journal_path,
                    episode=episode,
                    memory_version=memory_version,
                    base_commit=base_commit,
                    branch=worktree.branch,
                    live_path=store.live_memory_path,
                )
            # Keep control state and Git private; the Agent receives only its
            # persistent draft and submits handoffs through Runtime Journal.
            agent_workspace = self.base_campaign.register_runtime_episode(
                worktree.path, episode=episode, memory_version=memory_version,
                base_commit=base_commit, branch=worktree.branch,
                minimum_experiments=0,
            )
            prompt = self._prompt(
                episode=episode,
                version=memory_version,
                worktree=worktree,
                conversion_pending=conversion_pending,
                resumed=resumed,
                agent_workspace=agent_workspace,
                verifier=verifier,
                episode_mode=episode_mode,
            )
            store.write_brief(episode, prompt)
            telemetry_environment = {
                "ATREX_EPISODE_MODE": episode_mode,
                "ATREX_SESSION_CAPTURE_DIR": str(store.episode_dir(episode) / "sessions"),
                "ATREX_TELEMETRY_CAMPAIGN_ID": str(
                    getattr(self.base_campaign, "campaign_name", self.workspace.name)
                ),
                "ATREX_TELEMETRY_ITERATION_ID": f"episode-{episode:04d}",
                "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
            }
            telemetry_environment.update(
                self.base_campaign.agent_environment()
            )
            usage_receipt = uuid.uuid4().hex
            result = runner.run(
                agent_workspace,
                prompt,
                handoff_path=handoff_path,
                handoff_resumes=(
                    max(self.handoff_resumes, GOAL_HANDOFF_RESUMES)
                    if episode_mode == "goal" else self.handoff_resumes
                ),
                completion_check=lambda handoff: self._completion_check(
                    worktree,
                    journal_path,
                    handoff,
                ),
                reasoning_effort=EPISODE_REASONING_EFFORT,
                telemetry_environment=telemetry_environment,
                record_usage=lambda tokens: store.record_usage(
                    state, usage_receipt, tokens
                ),
            )
            # Also persist before verification: a GPU outage there must not
            # discard the coding session's usage. Receipt replay is harmless.
            store.record_usage(state, usage_receipt, result.tokens)
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
                violation, paths, verification, accepted = (
                    self._assess_terminal_handoff(
                        store,
                        active,
                        worktree,
                        handoff,
                        memory_version=memory_version,
                        conversion_pending=conversion_pending,
                        verifier=verifier,
                    )
                )

            memory, valid_blocked = self._record_terminal_episode(
                store,
                state,
                active,
                worktree,
                memory_version=memory_version,
                status=status,
                candidate_commit=candidate_commit,
                violation=violation,
                paths=paths,
                verification=verification,
                accepted=accepted,
                session_id=result.session_id,
                resume_count=result.resume_count,
                tokens=result.tokens,
                tokens_accounted=True,
                invocations=result.invocations,
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
                print(
                    f"[long-horizon] blocked at episode={episode}; starting a fresh "
                    "long-horizon episode retry",
                    flush=True,
                )
                continue
            if (
                self.max_stall
                and state.consecutive_without_promotion >= self.max_stall
                and self._episode_mode(state) != "goal"
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
