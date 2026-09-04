from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
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
from orchestrator.constants import DEFAULT_FAST_EPISODES, DEFAULT_FAST_TRIALS

from .models import (
    EpisodeHandoff,
    SupervisorState,
    VerificationResult,
    VerificationRun,
)
from .protocol import read_handoff
from .session import LongSessionRunner
from .store import CampaignStore, RUNTIME_DIR, VERIFY_DIR
from .telemetry import summarize_episode
from .verifier import GatewayABBAValidator


MODULE_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = MODULE_ROOT / "orchestrator" / "prompts" / "episode.md"
FAST_PROMPT_PATH = MODULE_ROOT / "orchestrator" / "prompts" / "fast_episode.md"
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
FAST_POLICY_REVIEW_REQUEST_PATH = Path(
    ".atrex_long_horizon/policy_review_request.json"
)
FAST_REASONING_EFFORT = "max"
FULL_REASONING_EFFORT = "max"
STAGED_STALLED_RETRY_THRESHOLD = 3


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


def _episode_evaluation_count(episode_workspace: Path) -> int:
    """Count durable evaluator results emitted by one episode."""
    path = episode_workspace / EPISODE_EVALUATIONS_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return 0
    count = 0
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            count += 1
    return count


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


@dataclass
class LongHorizonCampaign:
    base_campaign: main_adapter.Campaign
    max_episodes: int = 8
    max_version: int | None = None
    fast_episodes: int = DEFAULT_FAST_EPISODES
    fast_trials: int = DEFAULT_FAST_TRIALS
    episode_limit: int = 0
    token_budget: int = 0
    handoff_resumes: int = 2
    max_stall: int = 0
    verifier: GatewayABBAValidator | None = None
    session_runner: LongSessionRunner | None = None
    worktree_root: Path | None = None
    max_staged_episodes: int = 4
    staged_after_episodes: int = 40
    staged_after_stall: int = 8

    def __post_init__(self) -> None:
        if self.fast_episodes < 0:
            raise ValueError("fast_episodes must be non-negative")
        if self.fast_trials < 1:
            raise ValueError("fast_trials must be positive")
        if self.max_staged_episodes < 0:
            raise ValueError("max_staged_episodes must be non-negative")
        if self.staged_after_episodes < 0:
            raise ValueError("staged_after_episodes must be non-negative")
        if self.staged_after_stall < 0:
            raise ValueError("staged_after_stall must be non-negative")

    @property
    def workspace(self) -> Path:
        return self.base_campaign.workspace

    def _is_fast_episode(self, episode: int) -> bool:
        """Use the lightweight path for the first N optimization episodes."""
        return self.fast_episodes > 0 and 1 <= episode <= self.fast_episodes

    def _active_fast_trials(
        self, active: dict[str, Any], *, fast_mode: bool
    ) -> int:
        """Keep an in-flight fast episode's original trial contract across restarts."""
        value = active.get("fast_trials")
        if (
            fast_mode
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            return value
        return self.fast_trials

    @staticmethod
    def _episode_reasoning_effort(*, fast_mode: bool) -> str:
        return FAST_REASONING_EFFORT if fast_mode else FULL_REASONING_EFFORT

    def _staged_trigger(
        self, state: SupervisorState, *, fast_mode: bool
    ) -> str:
        if self.max_staged_episodes <= 0 or fast_mode:
            return ""
        if state.consecutive_staged >= self.max_staged_episodes:
            return ""
        if state.consecutive_staged > 0:
            return "continuation"
        if (
            self.staged_after_stall > 0
            and state.consecutive_without_promotion >= self.staged_after_stall
        ):
            return "promotion_drought"
        if state.episodes >= self.staged_after_episodes:
            return "episode_count"
        return ""

    def _staged_allowed(
        self, state: SupervisorState, *, fast_mode: bool
    ) -> bool:
        return bool(self._staged_trigger(state, fast_mode=fast_mode))

    @staticmethod
    def _staged_blocked_retries(
        state: SupervisorState,
        staged_checkpoint: dict[str, Any] | None,
    ) -> int:
        """Count blocked landing attempts since the current stage was created."""
        if not staged_checkpoint:
            return 0
        initiative_id = str(staged_checkpoint.get("initiative_id", "")).strip()
        source_episode = staged_checkpoint.get("source_episode", 0)
        if not initiative_id or not isinstance(source_episode, int):
            return 0
        return sum(
            1
            for attempt in state.attempts
            if attempt.get("status") == "blocked"
            and not attempt.get("violation")
            and attempt.get("initiative_id") == initiative_id
            and isinstance(attempt.get("episode"), int)
            and int(attempt["episode"]) > source_episode
        )

    def _staged_rewrite_directive(
        self,
        *,
        staged_allowed: bool,
        staged_checkpoint: dict[str, Any] | None,
        completed_episodes: int,
        promotion_drought: int,
        staged_trigger: str,
        staged_blocked_retries: int,
        fast_mode: bool,
    ) -> str:
        if self.max_staged_episodes <= 0:
            return (
                "Staged architectural rewrites are disabled for this campaign. Do not use "
                "`staged_ready`; finish with `candidate_ready`, `pivot`, or `blocked`."
            )
        if (
            staged_checkpoint is None
            and completed_episodes < self.staged_after_episodes
            and not (
                self.staged_after_stall > 0
                and promotion_drought >= self.staged_after_stall
            )
        ):
            stall_clause = (
                f" or {self.staged_after_stall} consecutive episodes without promotion"
                if self.staged_after_stall > 0
                else ""
            )
            return (
                "Staged architectural rewrites activate after "
                f"{self.staged_after_episodes} completed optimization episodes"
                f"{stall_clause}. This campaign has completed {completed_episodes} "
                f"episodes with a promotion drought of {promotion_drought}; do not use "
                "`staged_ready` yet."
            )
        if fast_mode:
            return (
                "Staged architectural rewrites are unavailable in fast episodes. Do not use "
                "`staged_ready` in this episode."
            )
        if not staged_allowed:
            if staged_checkpoint is None:
                return (
                    "This episode's terminal contract was frozen before staged eligibility was "
                    "latched. Do not use `staged_ready` in this in-flight episode; the next new "
                    "full episode will reevaluate the architectural escape trigger."
                )
            return (
                f"The {self.max_staged_episodes}-checkpoint initiative budget was reached. "
                "Do not use `staged_ready`; "
                "produce a final candidate, pivot, or report a real blocker."
            )
        if staged_checkpoint:
            directive = (
                "This worktree was bootstrapped from a non-production staged checkpoint for "
                f"initiative `{staged_checkpoint.get('initiative_id')}`, stage "
                f"{staged_checkpoint.get('stage')}. Continue its declared next stage: "
                f"{staged_checkpoint.get('next_stage')}. Its escape hypothesis is "
                f"{staged_checkpoint.get('escape_hypothesis')}; its architectural delta is "
                f"{staged_checkpoint.get('architectural_delta')}. Keep the declared final success "
                f"criterion ({staged_checkpoint.get('final_success_criterion')}) and abort "
                f"criterion ({staged_checkpoint.get('abort_criterion')}) stable. If the abort "
                "criterion is met, pivot instead of preserving the checkpoint. You may publish "
                "another `staged_ready` "
                "checkpoint if a coherent enabling stage is complete but the full initiative is "
                "not yet promotion-ready."
            )
            if staged_blocked_retries >= STAGED_STALLED_RETRY_THRESHOLD:
                directive += (
                    f" This checkpoint has already consumed {staged_blocked_retries} valid "
                    "blocked continuation episodes since its last stage advancement, so treat "
                    "the evidence plan as stalled. Do not replay the same measurement budget "
                    "unchanged. Before another expensive campaign, pre-register a materially "
                    "different, variance-aware sequential stopping rule using same-allocation "
                    "paired evidence and explicit success and abort bounds; keep the final "
                    "production correctness, policy, and performance gates unchanged. If no new "
                    "falsifiable evidence plan or architectural advancement can resolve the "
                    "final criterion, pivot so another initiative can explore the search space. "
                    "A genuine external blocker may still finish as `blocked` and preserve the "
                    "checkpoint, but unchanged bytes or another identical retry must not be "
                    "reported as stage advancement."
                )
            return directive
        if staged_trigger == "promotion_drought":
            return (
                f"This campaign has gone {promotion_drought} consecutive episodes without a "
                "production promotion, so this is an architectural escape episode. Start from a "
                "bottleneck that local tuning of the incumbent cannot remove, cite the exhausted "
                "local directions, and implement a materially different dataflow, layout, "
                "pipeline, synchronization, or communication design. Use `candidate_ready` if "
                "the structural candidate is already mature; otherwise use `staged_ready` only "
                "for a coherent first enabling stage. Do not spend this escape opportunity on "
                "another parameter-only tweak."
            )
        return (
            "This full episode may start a multi-episode architectural initiative. Use "
            "`staged_ready` only for a coherent, committed enabling stage that is intentionally "
            "not promotion-ready; identify the initiative, stage number, concrete next stage, "
            "escape hypothesis, architectural delta, final success criterion, and abort criterion."
        )

    def _restore_staged_checkpoint(
        self,
        store: CampaignStore,
        worktree: EpisodeWorktree,
        active: dict[str, Any],
    ) -> None:
        if isinstance(active.get("staged_checkpoint"), dict):
            return
        snapshot = store.load_staged_checkpoint()
        if snapshot is None:
            return
        metadata, kernel = snapshot
        current_kernel = (worktree.path / "kernel.py").read_bytes()
        if git_head(worktree.path) != worktree.base_commit and current_kernel != kernel:
            raise RuntimeError(
                "cannot restore staged checkpoint over divergent episode work"
            )
        development_base_commit = worktree.bootstrap_staged_kernel(
            kernel,
            initiative_id=str(metadata["initiative_id"]),
            stage=int(metadata["stage"]),
        )
        active["staged_checkpoint"] = metadata
        active["development_base_commit"] = development_base_commit
        store.save_active(active)

    def _expected_shape_ids(self) -> set[str] | None:
        private_reference_dir = self.base_campaign.private_reference_dir
        if private_reference_dir is None:
            return None
        return set(
            json.loads(
                (private_reference_dir / "shapes.json").read_text(encoding="utf-8")
            )
        )

    def _fast_evaluator_command(self, version: int) -> str:
        command = ["python", "tools/sandbox.py", "--kind", "run"]
        if self.base_campaign.sandbox_hardware:
            command += ["--hardware", self.base_campaign.sandbox_hardware]
        if self.base_campaign.sandbox_url:
            command += ["--url", self.base_campaign.sandbox_url]
        elif self.base_campaign.sandbox_profile:
            command += ["--gateway-profile", self.base_campaign.sandbox_profile]
        command += [
            "--no-sync",
            "--",
            "python",
            "test_kernel.py",
            "--version",
            f"v{version}",
            "--no-memory",
        ]
        return shlex.join(command)

    def _review_fast_candidate_snapshot(
        self,
        worktree: EpisodeWorktree,
        candidate_commit: str,
        *,
        require_gluon: bool,
    ) -> None:
        """Prewarm the production-review cache from an immutable candidate commit."""
        resolved = git_text(
            worktree.path,
            "rev-parse",
            "--verify",
            f"{candidate_commit}^{{commit}}",
            check=False,
        )
        if not resolved:
            return
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", worktree.base_commit, resolved],
            cwd=str(worktree.path),
            capture_output=True,
            check=False,
        )
        if ancestor.returncode != 0:
            return
        with tempfile.TemporaryDirectory(prefix="atrex-fast-policy-snapshot-") as value:
            snapshot = Path(value)
            for relative in ("kernel.py", "solution.json"):
                blob = subprocess.run(
                    ["git", "show", f"{resolved}:{relative}"],
                    cwd=str(worktree.path),
                    capture_output=True,
                    check=False,
                )
                if blob.returncode != 0:
                    if relative == "kernel.py":
                        return
                    continue
                (snapshot / relative).write_bytes(blob.stdout)
            print(
                "[long-horizon] fast candidate "
                f"{resolved[:12]}: starting policy review alongside evaluator",
                flush=True,
            )
            # The campaign reviewer caches by the exact bounded candidate digest. The
            # final call on the live worktree therefore only persists/reuses this verdict.
            main_adapter.candidate_policy_violations(
                self.base_campaign,
                snapshot,
                require_gluon=require_gluon,
            )

    def _prewarm_fast_policy_reviews(
        self,
        worktree: EpisodeWorktree,
        *,
        require_gluon: bool,
        stop_event: Event,
    ) -> None:
        """Review each atomically submitted fast candidate while its evaluator runs."""
        request_path = worktree.path / FAST_POLICY_REVIEW_REQUEST_PATH
        reviewed: set[str] = set()

        def review_latest_request() -> None:
            try:
                payload = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return
            candidate_commit = (
                payload.get("candidate_commit") if isinstance(payload, dict) else None
            )
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or not isinstance(candidate_commit, str)
                or not candidate_commit.strip()
            ):
                return
            candidate_commit = candidate_commit.strip()
            if candidate_commit in reviewed:
                return
            reviewed.add(candidate_commit)
            self._review_fast_candidate_snapshot(
                worktree,
                candidate_commit,
                require_gluon=require_gluon,
            )

        while not stop_event.wait(0.1):
            review_latest_request()
        # Close the race where the final request is renamed immediately before the
        # episode agent exits and the supervisor signals this watcher to stop.
        review_latest_request()

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
        fast_mode: bool,
        fast_trials: int | None = None,
        resumed: bool = False,
        staged_allowed: bool = False,
        staged_checkpoint: dict[str, Any] | None = None,
        completed_episodes: int = 0,
        promotion_drought: int = 0,
        staged_trigger: str = "",
        staged_blocked_retries: int = 0,
    ) -> str:
        directives = main_adapter.episode_directives(self.base_campaign, version)
        fast_trial_count = fast_trials or self.fast_trials
        journal_command = (
            f"PYTHONPATH={MODULE_ROOT} python -m long_horizon.journal "
            f"--live-path {json.dumps(str(live_memory_path))}"
        )
        fast_trial_plan_paths = "\n".join(
            f"- Trial {trial}: `plans/v{version}_trial{trial}_draft.md` -> "
            f"`plans/v{version}_trial{trial}_plan.md`"
            for trial in range(1, fast_trial_count + 1)
        )
        return _render(
            (FAST_PROMPT_PATH if fast_mode else PROMPT_PATH).read_text(
                encoding="utf-8"
            ),
            {
                "EPISODE": episode,
                "VERSION": version,
                "WORKSPACE": worktree.path,
                "PLATFORM": self.base_campaign.platform,
                "FRAMEWORK": self.base_campaign.framework,
                "BASE_COMMIT": worktree.base_commit,
                "DEVELOPMENT_BASE_COMMIT": git_head(worktree.path),
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
                "FAST_TRIALS": fast_trial_count,
                "FAST_TRIAL_PLAN_PATHS": fast_trial_plan_paths,
                "FAST_EVALUATOR_COMMAND": self._fast_evaluator_command(version),
                "RESUME_DIRECTIVE": (
                    "This episode is resuming after a supervisor restart. Keep and reuse the "
                    "existing worktree, checkpoints, journal, plans, profiles, generated files, "
                    "and source edits. Inspect them before acting; do not reset, clean, or stash "
                    "them. If the journal is already finalized and consistent, republish its "
                    "matching handoff. Otherwise continue the in-progress engineering work."
                    if resumed
                    else "This is a new episode worktree with no interrupted work to recover."
                ),
                "CONVERSION_DIRECTIVE": (
                    "This episode is a mandatory Triton-to-Gluon conversion attempt. Do not "
                    "submit another Triton kernel. A candidate must be a committed Gluon kernel, "
                    f"pass correctness, and stay within {main_adapter.CONVERT_PERF_TOL:.0%} of "
                    "the incumbent latency."
                    if conversion_pending
                    else "No mandatory framework conversion is currently latched."
                ),
                "STAGED_REWRITE_DIRECTIVE": self._staged_rewrite_directive(
                    staged_allowed=staged_allowed,
                    staged_checkpoint=staged_checkpoint,
                    completed_episodes=completed_episodes,
                    promotion_drought=promotion_drought,
                    staged_trigger=staged_trigger,
                    staged_blocked_retries=staged_blocked_retries,
                    fast_mode=fast_mode,
                ),
            },
        )

    def _fast_verification_result(
        self, episode_workspace: Path, *, memory_version: int
    ) -> VerificationResult:
        """Score the final recorded evaluator result without launching ABBA.

        ``tools/sandbox.py`` fingerprints ``kernel.py`` in every episode result.  The
        reader below selects a complete passing trial result whose fingerprint matches
        the final selected candidate, then compares that measurement with canonical
        incumbent memory.  This deliberately trades statistical rigor for turnaround.
        """
        artifact = str(episode_workspace / EPISODE_EVALUATIONS_PATH)
        expected_shape_ids = self._expected_shape_ids()
        candidate_result = _latest_complete_episode_performance(
            episode_workspace,
            expected_shape_ids=expected_shape_ids,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if candidate_result is None:
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                error=(
                    "fast mode requires one complete passing evaluator result for the "
                    "final kernel.py"
                ),
                artifact=artifact,
            )
        candidate_latency = float(candidate_result["latency_us_geomean"])
        candidate_score = float(candidate_result["performance_score"])
        run = VerificationRun(
            revision="candidate",
            repeat=0,
            exit_code=0,
            result=candidate_result,
        )
        incumbent = _latest_complete_canonical_performance(
            self.workspace,
            before_version=memory_version,
            expected_shape_ids=expected_shape_ids,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if incumbent is None:
            return VerificationResult(
                "FAIL",
                candidate_latency,
                None,
                None,
                runs=[run],
                error="fast mode could not find complete canonical incumbent performance",
                artifact=artifact,
            )
        incumbent_performance, _incumbent_version = incumbent
        incumbent_latency = float(incumbent_performance["latency_us_geomean"])
        incumbent_score = float(incumbent_performance["performance_score"])
        improvement = (candidate_score / incumbent_score - 1.0) * 100.0
        threshold = float(self.base_campaign.min_improvement_pct)
        if improvement <= threshold:
            return VerificationResult(
                "FAIL",
                candidate_latency,
                incumbent_latency,
                improvement,
                runs=[run],
                error=(
                    "fast evaluator performance-score improvement "
                    f"{improvement:.6f}% did not exceed {threshold:.3f}%"
                ),
                artifact=artifact,
                candidate_performance_score=candidate_score,
                incumbent_performance_score=incumbent_score,
            )
        return VerificationResult(
            "PASS",
            candidate_latency,
            incumbent_latency,
            improvement,
            runs=[run],
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
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
        *,
        fast_mode: bool = False,
        fast_trials: int | None = None,
        staged_allowed: bool = False,
    ) -> str:
        candidate = (
            handoff.candidate_commit if handoff.status == "candidate_ready" else ""
        )
        checkpoint = (
            handoff.checkpoint_commit if handoff.status == "staged_ready" else ""
        )
        diagnosis = validate_terminal(
            journal_path,
            expected_episode=worktree.episode,
            base_commit=worktree.base_commit,
            branch=worktree.branch,
            state=handoff.status,
            candidate_commit=candidate,
            checkpoint_commit=checkpoint,
        )
        if diagnosis:
            return diagnosis
        if handoff.status == "staged_ready" and not staged_allowed:
            return "staged_ready is not enabled for this episode"
        if fast_mode and handoff.status != "blocked":
            required_fast_trials = fast_trials or self.fast_trials
            try:
                journal = load_journal(journal_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                return f"cannot validate fast trial evidence: {exc}"
            experiments = journal.get("experiments")
            experiment_count = len(experiments) if isinstance(experiments, list) else 0
            if experiment_count < required_fast_trials:
                return (
                    f"fast episode requires {required_fast_trials} recorded trial experiments; "
                    f"found {experiment_count}"
                )
            evaluation_count = _episode_evaluation_count(worktree.path)
            if evaluation_count < required_fast_trials:
                return (
                    f"fast episode requires {required_fast_trials} evaluator results; "
                    f"found {evaluation_count}"
                )
        if handoff.status not in {"candidate_ready", "staged_ready"}:
            return ""
        terminal_commit = candidate or checkpoint
        violation, _ = worktree.validate_candidate(terminal_commit)
        if violation:
            return violation
        try:
            journal = load_journal(journal_path)
            finalized_at = _iso_timestamp(str(journal["finalized_at"]))
            committed_at = float(
                git_text(
                    worktree.path, "show", "-s", "--format=%ct", terminal_commit
                )
            )
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return f"cannot validate terminal journal ordering: {exc}"
        if finalized_at <= committed_at:
            return (
                "terminal journal must be finalized after the exact handoff commit"
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
        fast_mode: bool = False,
        fast_trials: int | None = None,
    ) -> dict[str, Any]:
        fast_trial_count = fast_trials or self.fast_trials
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
                    len(expected_shapes) if expected_shapes is not None else None
                ),
                "shape_measurement_repeats": shape_measurement_repeats,
                "measurement_subject": "candidate",
                "measurement_source": (
                    "episode_evaluator_result"
                    if fast_mode
                    else "authoritative_verification"
                ),
                "comparison_method": (
                    "single_candidate_vs_canonical_incumbent"
                    if fast_mode
                    else "same_allocation_abba"
                ),
                "carried_from_version": None,
                "performance_objective": representative.get(
                    "performance_objective"
                ),
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
                    "fast_long_horizon_episode"
                    if fast_mode
                    else "long_horizon_episode"
                ),
                "action_description": str(
                    outcome.get("summary", "verified long-horizon candidate")
                ),
                "expected_impact": (
                    f"best of {fast_trial_count} evaluator-backed trials against "
                    "canonical incumbent"
                    if fast_mode
                    else "independently verified incumbent/candidate latency reduction"
                ),
                "risks_and_rollback": "candidate retained on isolated episode branch",
            },
            "profile_evidence": {
                "tool_used": (
                    "none (fast mode)"
                    if fast_mode
                    else "episode-owned profiler evidence plus supervisor ABBA"
                ),
                "evidence_summary": f"{len(journal.get('experiments', []))} structured experiments",
                "bottleneck_type": (
                    "not_profiled_fast_mode" if fast_mode else "episode-derived"
                ),
                "evidence_chain": (
                    f"{fast_trial_count} reviewed plan -> implementation -> evaluator "
                    "trials -> best-candidate promotion"
                    if fast_mode
                    else "episode evidence -> candidate -> independent ABBA -> promotion"
                ),
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
            "long_horizon": {
                "status": "candidate_ready",
                "mode": "fast" if fast_mode else "full",
                "verification": "single_evaluator" if fast_mode else "abba",
                "fast_trials": fast_trial_count if fast_mode else None,
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
        checkpoint_commit: str = "",
        verification: VerificationResult | None = None,
        episode_workspace: Path | None = None,
        fast_mode: bool = False,
        fast_trials: int | None = None,
    ) -> dict[str, Any]:
        fast_trial_count = fast_trials or self.fast_trials
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
                "performance_objective": representative.get(
                    "performance_objective"
                ),
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
                    "fast_long_horizon_episode"
                    if fast_mode
                    else "long_horizon_episode"
                ),
                "action_description": str(outcome.get("summary", status)),
                "expected_impact": "episode exploration did not produce a promotable improvement",
                "risks_and_rollback": "incumbent kernel was preserved",
            },
            "profile_evidence": {
                "tool_used": "none (fast mode)" if fast_mode else "episode journal",
                "evidence_summary": f"{len(journal.get('experiments', []))} structured experiments",
                "bottleneck_type": (
                    "not_profiled_fast_mode" if fast_mode else "episode-derived"
                ),
                "evidence_chain": (
                    f"{fast_trial_count} reviewed plan -> implementation -> evaluator "
                    "trials -> no promotion"
                    if fast_mode
                    else "episode evidence -> terminal handoff -> no promotion"
                ),
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
                "checkpoint_commit": checkpoint_commit or None,
                "initiative_id": outcome.get("initiative_id"),
                "stage": outcome.get("stage"),
                "next_stage": outcome.get("next_stage"),
                "mode": "fast" if fast_mode else "full",
                "fast_trials": fast_trial_count if fast_mode else None,
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
        fast_mode: bool,
        conversion_pending: bool,
        verifier: GatewayABBAValidator,
    ) -> tuple[str, list[str], VerificationResult | None, bool]:
        """Apply the authoritative candidate gates to one terminal handoff."""
        if handoff.status not in {"candidate_ready", "staged_ready"}:
            return "", [], None, False

        terminal_commit = (
            handoff.candidate_commit
            if handoff.status == "candidate_ready"
            else handoff.checkpoint_commit
        )
        violation, paths = worktree.validate_candidate(terminal_commit)
        if handoff.status == "staged_ready":
            try:
                journal = load_journal(worktree.path / RUNTIME_DIR / "journal.json")
                outcome = journal["outcome"]
                initiative_id = str(outcome["initiative_id"])
                stage = int(outcome["stage"])
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                return "staged_ready journal metadata is invalid", paths, None, False
            prior = active.get("staged_checkpoint")
            if isinstance(prior, dict):
                if initiative_id != str(prior.get("initiative_id", "")):
                    violation = "staged_ready must continue the active initiative_id"
                elif stage != int(prior.get("stage", 0)) + 1:
                    violation = "staged_ready stage must increment the active stage by one"
                else:
                    for field in (
                        "escape_hypothesis",
                        "architectural_delta",
                        "final_success_criterion",
                        "abort_criterion",
                    ):
                        if str(outcome.get(field, "")).strip() != str(
                            prior.get(field, "")
                        ).strip():
                            violation = (
                                "staged_ready continuation must preserve the active "
                                f"{field}"
                            )
                            break
            elif stage != 1:
                violation = "a new staged initiative must begin at stage 1"
            return violation, paths, None, False

        candidate_commit = handoff.candidate_commit
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
                "checking_fast_evaluator" if fast_mode else "verifying"
            )
            store.save_active(active)
            if fast_mode:
                verification = self._fast_verification_result(
                    worktree.path,
                    memory_version=memory_version,
                )
            else:
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
        fast_mode: bool,
        status: str,
        candidate_commit: str,
        checkpoint_commit: str = "",
        violation: str,
        paths: list[str],
        verification: VerificationResult | None,
        accepted: bool,
        session_id: str = "",
        resume_count: int = 0,
        tokens: int = 0,
        invocations: tuple[Any, ...] = (),
        fast_trials: int | None = None,
        recovered_after_supervisor_interruption: bool = False,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Archive and commit one terminal episode exactly once."""
        episode = worktree.episode
        base_commit = worktree.base_commit
        journal_path = worktree.path / RUNTIME_DIR / "journal.json"
        state.episodes = max(state.episodes, episode)
        state.tokens += max(0, int(tokens))
        fast_trial_count = fast_trials or self._active_fast_trials(
            active, fast_mode=fast_mode
        )

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
        prior_staged = (
            active.get("staged_checkpoint")
            if isinstance(active.get("staged_checkpoint"), dict)
            else {}
        )
        attempt = {
            "episode": episode,
            "version": memory_version,
            "mode": "fast" if fast_mode else "full",
            "fast_trials": fast_trial_count if fast_mode else None,
            "status": status,
            "accepted": accepted,
            "violation": violation or None,
            "base_commit": base_commit,
            "episode_branch": worktree.branch,
            "episode_head": git_head(worktree.path),
            "candidate_commit": candidate_commit or None,
            "checkpoint_commit": checkpoint_commit or None,
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
            "initiative_id": outcome.get("initiative_id")
            or prior_staged.get("initiative_id"),
            "initiative_stage": outcome.get("stage")
            or prior_staged.get("stage"),
            "staged_trigger": active.get("staged_trigger") or None,
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
                verification=verification,
                fast_mode=fast_mode,
                fast_trials=fast_trial_count,
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
            if prior_staged:
                state.staged_initiatives_promoted += 1
            state.consecutive_without_promotion = 0
            state.consecutive_staged = 0
            store.clear_staged_checkpoint()
            main_adapter.save_stall(self.workspace, 0)
            active["phase"] = "promoted"
            active["promotion_commit"] = promotion_commit
            store.save_active(active)
        else:
            valid_staged = status == "staged_ready" and not violation
            if valid_staged:
                active["phase"] = "staging"
                store.save_active(active)
                staged_metadata = store.save_staged_checkpoint(
                    {
                        "initiative_id": str(outcome["initiative_id"]),
                        "stage": int(outcome["stage"]),
                        "next_stage": str(outcome["next_stage"]),
                        "escape_hypothesis": str(outcome["escape_hypothesis"]),
                        "architectural_delta": str(outcome["architectural_delta"]),
                        "final_success_criterion": str(
                            outcome["final_success_criterion"]
                        ),
                        "abort_criterion": str(outcome["abort_criterion"]),
                        "checkpoint_commit": checkpoint_commit,
                        "source_episode": episode,
                        "source_base_commit": base_commit,
                    },
                    (worktree.path / "kernel.py").read_bytes(),
                )
                attempt["staged_checkpoint"] = staged_metadata
                active["staged_checkpoint"] = staged_metadata
            active["phase"] = "recording"
            active["terminal_status"] = status
            store.save_active(active)
            memory = self._outcome_memory_record(
                version=memory_version,
                status=status,
                violation=violation,
                journal=journal,
                candidate_commit=candidate_commit,
                checkpoint_commit=checkpoint_commit,
                verification=verification,
                episode_workspace=worktree.path,
                fast_mode=fast_mode,
                fast_trials=fast_trial_count,
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
            if valid_staged:
                state.staged += 1
                state.consecutive_staged += 1
                if not prior_staged:
                    state.staged_initiatives_started += 1
                state.consecutive_without_promotion += 1
                main_adapter.save_stall(
                    self.workspace, state.consecutive_without_promotion
                )
            else:
                state.consecutive_without_promotion += 1
                main_adapter.save_stall(
                    self.workspace, state.consecutive_without_promotion
                )
            if status == "pivot" and not violation:
                state.pivoted += 1
                if prior_staged:
                    state.staged_initiatives_abandoned += 1
                state.consecutive_staged = 0
                store.clear_staged_checkpoint()
            elif status == "blocked" and not violation:
                state.blocked += 1
            elif status == "invalid_handoff":
                state.protocol_failures += 1
            elif not valid_staged:
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
            f"[long-horizon] episode={episode} mode={'fast' if fast_mode else 'full'} "
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
        fast_mode = active.get("mode") == "fast"
        fast_trials = self._active_fast_trials(active, fast_mode=fast_mode)
        diagnosis = self._completion_check(
            worktree,
            runtime / "journal.json",
            handoff,
            fast_mode=fast_mode,
            fast_trials=fast_trials,
            staged_allowed=bool(
                active.get(
                    "staged_allowed",
                    self._staged_allowed(state, fast_mode=fast_mode),
                )
            ),
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
            fast_mode=fast_mode,
            conversion_pending=conversion_pending,
            verifier=verifier,
        )
        self._record_terminal_episode(
            store,
            state,
            active,
            worktree,
            memory_version=memory_version,
            fast_mode=fast_mode,
            status=handoff.status,
            candidate_commit=handoff.candidate_commit,
            checkpoint_commit=handoff.checkpoint_commit,
            violation=violation,
            paths=paths,
            verification=verification,
            accepted=accepted,
            fast_trials=fast_trials,
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
        fast_mode = active.get("mode") == "fast"
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
                "staging",
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
                # The episode never reached a terminal handoff and its base is now
                # stale, so it is superseded rather than rejected on its merits.
                terminal_status = "interrupted"
            self._require_canonical_memory(memory_version)
            if not already_recorded:
                recovered_prior_staged = (
                    active.get("staged_checkpoint")
                    if isinstance(active.get("staged_checkpoint"), dict)
                    else {}
                )
                state.episodes = max(state.episodes, episode)
                recovered_attempt: dict[str, Any] = {
                    "episode": episode,
                    "version": memory_version,
                    "status": "candidate_ready" if promoted else terminal_status,
                    "accepted": promoted,
                    "violation": None,
                    "base_commit": base_commit,
                    "episode_branch": branch,
                    "mode": "fast" if fast_mode else "full",
                    "recovered_after_supervisor_interruption": True,
                }
                if promoted:
                    state.accepted += 1
                    if recovered_prior_staged:
                        state.staged_initiatives_promoted += 1
                    state.consecutive_without_promotion = 0
                    state.consecutive_staged = 0
                    store.clear_staged_checkpoint()
                    recovered_attempt["promotion_commit"] = git_head(self.workspace)
                else:
                    recovered_attempt["outcome_commit"] = git_head(self.workspace)
                    if terminal_status == "staged_ready":
                        snapshot = store.load_staged_checkpoint()
                        if snapshot is None:
                            raise RuntimeError(
                                "recorded staged episode has no persisted checkpoint"
                            )
                        staged_metadata, _staged_kernel = snapshot
                        recovered_attempt["checkpoint_commit"] = (
                            staged_metadata.get("checkpoint_commit")
                        )
                        recovered_attempt["staged_checkpoint"] = staged_metadata
                        recovered_attempt["initiative_id"] = staged_metadata.get(
                            "initiative_id"
                        )
                        recovered_attempt["initiative_stage"] = staged_metadata.get(
                            "stage"
                        )
                        state.staged += 1
                        state.consecutive_staged += 1
                        if (
                            int(staged_metadata.get("stage", 0)) == 1
                            and int(staged_metadata.get("source_episode", 0)) == episode
                        ):
                            state.staged_initiatives_started += 1
                        state.consecutive_without_promotion += 1
                    else:
                        state.consecutive_without_promotion += 1
                    if terminal_status == "pivot":
                        state.pivoted += 1
                        if recovered_prior_staged:
                            state.staged_initiatives_abandoned += 1
                        state.consecutive_staged = 0
                        store.clear_staged_checkpoint()
                    elif terminal_status == "blocked":
                        state.blocked += 1
                        recovered_attempt["blocked_retry_scheduled"] = True
                    elif terminal_status == "interrupted":
                        state.interrupted += 1
                        recovered_attempt["violation"] = "supervisor process interrupted"
                    elif terminal_status == "invalid_handoff":
                        state.protocol_failures += 1
                    elif terminal_status == "staged_ready":
                        pass
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
                fast_mode=fast_mode,
                fast_trials=self._active_fast_trials(active, fast_mode=fast_mode),
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
                "mode": "fast" if fast_mode else "full",
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
        staged_snapshot = store.load_staged_checkpoint()
        if staged_snapshot is not None and self.max_staged_episodes <= 0:
            raise RuntimeError(
                "a staged architectural checkpoint exists; resume with "
                "--max-staged-episodes greater than zero"
            )
        if staged_snapshot is not None:
            staged_metadata, _staged_kernel = staged_snapshot
            state.consecutive_staged = max(
                state.consecutive_staged, int(staged_metadata["stage"])
            )
        if state.episodes == 0 and state.consecutive_without_promotion == 0:
            state.consecutive_without_promotion = main_adapter.restored_stall(
                self.workspace
            )
        verifier = self.verifier or GatewayABBAValidator(
            hardware=self.base_campaign.sandbox_hardware,
            profile=self.base_campaign.sandbox_profile,
            url=self.base_campaign.sandbox_url,
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
                mode = active.get("mode")
                fast_mode = (
                    mode == "fast"
                    if mode in {"fast", "full"}
                    else self._is_fast_episode(episode)
                )
            else:
                episode = state.episodes + 1
                fast_mode = self._is_fast_episode(episode)

            episode_mode = "fast" if fast_mode else "full"
            self.base_campaign.ensure_plan_reviewer_availability(
                episode_mode=episode_mode
            )

            if resumed:
                active.setdefault("mode", "fast" if fast_mode else "full")
                active.setdefault(
                    "fast_trials", self.fast_trials if fast_mode else None
                )
                computed_staged_trigger = self._staged_trigger(
                    state, fast_mode=fast_mode
                )
                active.setdefault(
                    "staged_allowed", bool(computed_staged_trigger)
                )
                active.setdefault(
                    "staged_trigger",
                    computed_staged_trigger
                    if bool(active.get("staged_allowed"))
                    else "",
                )
                if active.get("resumed_from_phase") == "preparing":
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
                    "mode": "fast" if fast_mode else "full",
                    "fast_trials": self.fast_trials if fast_mode else None,
                    "phase": "preparing",
                    "staged_allowed": self._staged_allowed(
                        state, fast_mode=fast_mode
                    ),
                    "staged_trigger": self._staged_trigger(
                        state, fast_mode=fast_mode
                    ),
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
            self._restore_staged_checkpoint(store, worktree, active)
            staged_checkpoint = (
                active.get("staged_checkpoint")
                if isinstance(active.get("staged_checkpoint"), dict)
                else None
            )
            staged_blocked_retries = self._staged_blocked_retries(
                state, staged_checkpoint
            )
            active["staged_blocked_retries"] = staged_blocked_retries
            active["staged_stall_review"] = (
                staged_blocked_retries >= STAGED_STALLED_RETRY_THRESHOLD
            )
            store.save_active(active)
            fast_trial_count = self._active_fast_trials(
                active, fast_mode=fast_mode
            )
            staged_allowed = bool(
                active.get(
                    "staged_allowed",
                    self._staged_allowed(state, fast_mode=fast_mode),
                )
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
                    development_base_commit=git_head(worktree.path),
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
                fast_mode=fast_mode,
                fast_trials=fast_trial_count,
                resumed=resumed,
                staged_allowed=staged_allowed,
                staged_checkpoint=staged_checkpoint,
                completed_episodes=state.episodes,
                promotion_drought=state.consecutive_without_promotion,
                staged_trigger=str(active.get("staged_trigger", "")),
                staged_blocked_retries=staged_blocked_retries,
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
            telemetry_environment.update(
                self.base_campaign.agent_environment(episode_mode=episode_mode)
            )
            policy_stop: Event | None = None
            policy_executor: ThreadPoolExecutor | None = None
            policy_future: Future[None] | None = None
            if (
                fast_mode
                and getattr(self.base_campaign, "optimization_mode", "")
                == "production"
            ):
                policy_stop = Event()
                policy_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"fast-policy-e{episode:04d}",
                )
                policy_future = policy_executor.submit(
                    self._prewarm_fast_policy_reviews,
                    worktree,
                    require_gluon=(
                        conversion_pending
                        or main_adapter.candidate_is_gluon(self.workspace)
                    ),
                    stop_event=policy_stop,
                )
            try:
                result = runner.run(
                    worktree.path,
                    prompt,
                    handoff_path=handoff_path,
                    handoff_resumes=self.handoff_resumes,
                    completion_check=lambda handoff: self._completion_check(
                        worktree,
                        journal_path,
                        handoff,
                        fast_mode=fast_mode,
                        fast_trials=fast_trial_count,
                        staged_allowed=staged_allowed,
                    ),
                    reasoning_effort=self._episode_reasoning_effort(
                        fast_mode=fast_mode
                    ),
                    telemetry_environment=telemetry_environment,
                )
            finally:
                if policy_stop is not None:
                    policy_stop.set()
                if policy_future is not None:
                    try:
                        policy_future.result()
                    except Exception as exc:
                        # The final synchronous policy gate below remains authoritative
                        # and fail-closed; prewarming is only a latency optimization.
                        print(
                            "[long-horizon] WARNING: fast policy prewarm failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                if policy_executor is not None:
                    policy_executor.shutdown()
            handoff = result.handoff
            status = handoff.status if handoff else "invalid_handoff"
            violation = ""
            candidate_commit = handoff.candidate_commit if handoff else ""
            checkpoint_commit = handoff.checkpoint_commit if handoff else ""
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
            elif status in {"candidate_ready", "staged_ready"}:
                violation, paths, verification, accepted = (
                    self._assess_terminal_handoff(
                        store,
                        active,
                        worktree,
                        handoff,
                        memory_version=memory_version,
                        fast_mode=fast_mode,
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
                fast_mode=fast_mode,
                status=status,
                candidate_commit=candidate_commit,
                checkpoint_commit=checkpoint_commit,
                violation=violation,
                paths=paths,
                verification=verification,
                accepted=accepted,
                session_id=result.session_id,
                resume_count=result.resume_count,
                tokens=result.tokens,
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
                and store.load_staged_checkpoint() is None
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
            f"staged={state.staged} rejected={state.rejected} pivoted={state.pivoted} "
            f"blocked={state.blocked} staged_initiatives_started="
            f"{state.staged_initiatives_started} staged_initiatives_promoted="
            f"{state.staged_initiatives_promoted} staged_initiatives_abandoned="
            f"{state.staged_initiatives_abandoned} "
            f"protocol_failures={state.protocol_failures} tokens={state.tokens}",
            flush=True,
        )
        return reason
