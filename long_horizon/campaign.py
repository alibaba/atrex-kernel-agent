from __future__ import annotations

import json
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
from .journal import validate_terminal
from .models import EpisodeHandoff, SupervisorState, VerificationResult
from .session import LongSessionRunner
from .store import CampaignStore, RUNTIME_DIR, VERIFY_DIR
from .verifier import GatewayABBAValidator


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "episode.md"
MODULE_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PREFIXES = ("plans/", "profiles/")


def _render(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def _iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _conversion_parity_passes(verification: VerificationResult) -> bool:
    candidate = verification.candidate_latency_us
    incumbent = verification.incumbent_latency_us
    if not isinstance(candidate, (int, float)) or not isinstance(incumbent, (int, float)):
        return False
    if candidate > incumbent * (1.0 + main_adapter.CONVERT_PERF_TOL):
        return False
    return bool(verification.runs) and all(
        run.exit_code == 0
        and isinstance(run.result, dict)
        and bool(run.result.get("all_pass"))
        for run in verification.runs
    )


@dataclass
class LongHorizonCampaign:
    base_campaign: main_adapter.Campaign
    max_episodes: int = 8
    max_version: int | None = None
    episode_limit: int = 0
    token_budget: int = 0
    session_timeout: int = 18_000
    handoff_resumes: int = 2
    max_stall: int = 0
    verifier: GatewayABBAValidator | None = None
    session_runner: LongSessionRunner | None = None
    worktree_root: Path | None = None

    @property
    def workspace(self) -> Path:
        return self.base_campaign.workspace

    def _history(self, state: SupervisorState) -> str:
        compact = []
        for attempt in state.attempts[-8:]:
            compact.append(
                {
                    key: attempt.get(key)
                    for key in (
                        "episode", "status", "accepted", "summary", "next_directions",
                        "candidate_commit", "promotion_commit", "verification", "violation",
                    )
                    if attempt.get(key) is not None
                }
            )
        return json.dumps(compact, indent=2, ensure_ascii=False)

    def _prompt(
        self,
        *,
        episode: int,
        version: int,
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff_path: Path,
        state: SupervisorState,
        conversion_pending: bool,
    ) -> str:
        directives = main_adapter.episode_directives(self.base_campaign)
        journal_command = f"PYTHONPATH={MODULE_ROOT} python -m long_horizon.journal"
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
                "HISTORY": self._history(state),
                "JOURNAL_COMMAND": journal_command,
                "MAIN_ITERATION_PLAYBOOK": main_adapter.iteration_playbook(
                    self.base_campaign, worktree.path, version
                ),
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
        candidate = handoff.candidate_commit if handoff.status == "candidate_ready" else ""
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
            committed_at = float(git_text(worktree.path, "show", "-s", "--format=%ct", candidate))
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return f"cannot validate terminal journal ordering: {exc}"
        if finalized_at <= committed_at:
            return "candidate journal must be finalized after the exact candidate commit"
        return ""

    def _copy_runtime_artifacts(self, worktree: EpisodeWorktree, episode_dir: Path) -> None:
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
        candidate_runs = [
            run for run in verification.runs
            if run.revision == "candidate" and isinstance(run.result, dict)
        ]
        representative = candidate_runs[-1].result if candidate_runs else {}
        by_shape = representative.get("latency_us_by_shape", {}) if representative else {}
        outcome = journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
        directions = outcome.get("next_directions", []) if isinstance(outcome, dict) else []
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
                "speedup_vs_ref_geomean": representative.get(
                    "speedup_vs_ref_geomean", 0.0
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
                "action_description": str(outcome.get("summary", "verified long-horizon candidate")),
                "expected_impact": "independently verified incumbent/candidate latency reduction",
                "risks_and_rollback": "candidate retained on isolated episode branch",
            },
            "profile_evidence": {
                "tool_used": "episode-owned profiler evidence plus supervisor ABBA",
                "evidence_summary": f"{len(journal.get('experiments', []))} structured experiments",
                "bottleneck_type": "episode-derived",
                "evidence_chain": "episode evidence -> candidate -> independent ABBA -> promotion",
            },
            "correctness": {
                "status": "PASS",
                "max_abs_err": representative.get("max_abs_err", 0.0),
                "max_rel_err": representative.get("max_rel_err", 0.0),
            },
            "quality_gate": {"result": "PASS", "failure_reason": None},
            "open_directions": [
                {"direction": value, "rationale": "carried from terminal episode journal"}
                for value in directions if isinstance(value, str)
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
    ) -> dict[str, Any]:
        outcome = journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
        directions = outcome.get("next_directions", []) if isinstance(outcome, dict) else []
        failure = violation or str(outcome.get("summary", status))
        return {
            "version": f"v{version}",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "latency_us": None,
                "latency_us_geomean": None,
                "latency_us_by_shape": {},
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
            "correctness": {"status": "FAIL" if violation else "UNKNOWN"},
            "quality_gate": {"result": "FAIL", "failure_reason": failure},
            "open_directions": [
                {"direction": value, "rationale": "carried from terminal episode journal"}
                for value in directions
                if isinstance(value, str)
            ],
            "git_commit_hash": None,
            "long_horizon": {
                "status": status,
                "candidate_commit": candidate_commit or None,
            },
        }

    def _terminal_padding_memory_record(
        self,
        *,
        version: int,
        source_attempt: dict[str, Any],
        incumbent_commit: str,
    ) -> dict[str, Any]:
        source_version = int(source_attempt.get("version", 0) or 0)
        source_episode = int(source_attempt.get("episode", 0) or 0)
        reason = (
            "terminal padding after a repeated blocked episode; the incumbent kernel is "
            "unchanged and this record exists only to satisfy main's initial aggregation barrier"
        )
        return {
            "version": f"v{version}",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "latency_us": None,
                "latency_us_geomean": None,
                "latency_us_by_shape": {},
            },
            "optimization": {
                "action_category": "long_horizon_terminal_padding",
                "action_description": reason,
                "expected_impact": "no kernel or performance change",
                "risks_and_rollback": "metadata-only commit; incumbent kernel is preserved",
            },
            "profile_evidence": {
                "tool_used": "none",
                "evidence_summary": "no agent, sandbox, or GPU execution",
                "bottleneck_type": "aggregation compatibility",
                "evidence_chain": "repeated blocked episode -> terminal metadata padding",
            },
            "correctness": {"status": "UNKNOWN"},
            "quality_gate": {"result": "FAIL", "failure_reason": reason},
            "open_directions": [],
            "git_commit_hash": None,
            "long_horizon": {
                "status": "terminal_padding",
                "terminal_padding": True,
                "source_blocked_version": source_version,
                "source_blocked_episode": source_episode,
                "incumbent_commit": incumbent_commit,
                "incumbent_preserved": True,
                "reason": "initial aggregation compatibility",
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

    def _pad_terminal_block_for_aggregation(
        self, source_attempt: dict[str, Any]
    ) -> list[int]:
        target = main_adapter.initial_aggregation_padding_target(self.base_campaign)
        current = main_adapter.latest_version(self.workspace)
        if target is None or current >= target:
            return []
        if not main_adapter.has_accepted_bucket_kernel(self.base_campaign):
            print(
                "[long-horizon] terminal block is below the initial aggregation barrier, "
                "but this bucket has no accepted kernel; refusing to pad",
                flush=True,
            )
            return []

        incumbent_commit = git_head(self.workspace)
        source_episode = int(source_attempt.get("episode", 0) or 0)
        padded: list[int] = []
        for version in range(current + 1, target + 1):
            memory = self._terminal_padding_memory_record(
                version=version,
                source_attempt=source_attempt,
                incumbent_commit=incumbent_commit,
            )
            record_episode_outcome(
                self.workspace,
                base_commit=git_head(self.workspace),
                version=version,
                episode=source_episode,
                status="terminal-padding",
                memory_record=memory,
            )
            main_adapter.notify_iteration(
                self.base_campaign,
                version,
                memory,
                False,
            )
            padded.append(version)
            print(
                f"[long-horizon] terminal padding v{version}/v{target}; "
                "incumbent kernel unchanged",
                flush=True,
            )
        return padded

    def _recover_interrupted(self, store: CampaignStore, state: SupervisorState) -> None:
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
            attempt.get("episode") == episode and attempt.get("episode_branch") == branch
            for attempt in state.attempts
        )
        if git_head(self.workspace) != base_commit:
            message = git_text(self.workspace, "log", "-1", "--format=%s", check=False)
            parent = git_text(self.workspace, "rev-parse", "HEAD^", check=False)
            evidence = git_text(
                self.workspace, "show", f"HEAD:memory/long_horizon_e{episode:04d}.json",
                check=False,
            )
            promoted = (
                phase in {"promoting", "promoted"}
                and parent == base_commit
                and message == f"episode {episode}: promote verified long-horizon candidate"
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
                raise RuntimeError("incumbent advanced during an interrupted episode without proof")
            if not already_recorded:
                state.episodes = max(state.episodes, episode)
                if promoted:
                    state.accepted += 1
                    state.consecutive_without_promotion = 0
                else:
                    state.consecutive_without_promotion += 1
                    if terminal_status == "pivot":
                        state.pivoted += 1
                    elif terminal_status == "blocked":
                        state.blocked += 1
                        retry_of = (
                            state.attempts[-1]
                            if self._blocked_retry_pending(state)
                            else None
                        )
                        recovered_attempt: dict[str, Any] = {
                            "episode": episode,
                            "version": memory_version,
                            "status": "blocked",
                            "accepted": False,
                            "violation": None,
                            "base_commit": base_commit,
                            "episode_branch": branch,
                            "outcome_commit": git_head(self.workspace),
                            "recovered_after_supervisor_interruption": True,
                            "blocked_retry_scheduled": retry_of is None,
                            "blocked_terminal": retry_of is not None,
                        }
                        if retry_of is not None:
                            recovered_attempt["blocked_retry_of_episode"] = retry_of.get(
                                "episode"
                            )
                        state.attempts.append(recovered_attempt)
                        store.archive_attempt(episode, recovered_attempt)
                    elif terminal_status == "invalid_handoff":
                        state.protocol_failures += 1
                    else:
                        state.rejected += 1
        else:
            # A crash during squash promotion can leave the incumbent index/worktree dirty
            # while HEAD still points at the immutable base. The active marker proves these
            # are supervisor-owned partial changes, so roll them back before continuing.
            if working_changes(self.workspace):
                subprocess.run(
                    ["git", "reset", "--hard", base_commit],
                    cwd=str(self.workspace), check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            registered = {
                Path(line.split(" ", 1)[1]).resolve()
                for line in git_text(self.workspace, "worktree", "list", "--porcelain").splitlines()
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
                worktree.remove(self.workspace)
            if not already_recorded:
                state.episodes = max(state.episodes, episode)
                state.interrupted += 1
                state.consecutive_without_promotion += 1
                state.attempts.append(
                    {
                        "episode": episode,
                        "status": "interrupted",
                        "accepted": False,
                        "violation": "supervisor process interrupted",
                        "base_commit": base_commit,
                        "episode_branch": branch,
                    }
                )
        main_adapter.save_stall(self.workspace, state.consecutive_without_promotion)
        store.save_state(state)
        store.clear_active()

    def run(self) -> str:
        main_adapter.prepare_campaign(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        if state.episodes == 0 and state.consecutive_without_promotion == 0:
            state.consecutive_without_promotion = main_adapter.restored_stall(self.workspace)
        self._recover_interrupted(store, state)
        if working_changes(self.workspace):
            raise RuntimeError(
                "long-horizon campaign requires a clean incumbent workspace: "
                + ", ".join(working_changes(self.workspace)[:12])
            )
        terminal_block = self._terminal_blocked_attempt(state)
        if terminal_block is not None:
            self._pad_terminal_block_for_aggregation(terminal_block)
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
                    "runtime linking dirtied the episode boundary: " + ", ".join(unexpected)
                )
            runtime = worktree.path / RUNTIME_DIR
            journal_path = runtime / "journal.json"
            handoff_path = runtime / "handoff.json"
            initialize_journal(
                journal_path, episode=episode, base_commit=base_commit, branch=worktree.branch
            )
            prompt = self._prompt(
                episode=episode,
                version=memory_version,
                worktree=worktree,
                journal_path=journal_path,
                handoff_path=handoff_path,
                state=state,
                conversion_pending=conversion_pending,
            )
            store.write_brief(episode, prompt)
            result = runner.run(
                worktree.path,
                prompt,
                timeout=self.session_timeout,
                handoff_path=handoff_path,
                handoff_resumes=self.handoff_resumes,
                completion_check=lambda handoff: self._completion_check(
                    worktree, journal_path, handoff
                ),
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
                violation = result.completion_diagnosis or "session produced no valid terminal handoff"
            elif status == "candidate_ready":
                violation, paths = worktree.validate_candidate(candidate_commit)
                if (
                    not violation
                    and conversion_pending
                    and not main_adapter.candidate_is_gluon(worktree.path)
                ):
                    violation = "mandatory conversion candidate is not a committed Gluon kernel"
                if not violation:
                    policy_violations = main_adapter.candidate_policy_violations(
                        self.base_campaign, worktree.path
                    )
                    if policy_violations:
                        violation = "production policy rejected candidate: " + "; ".join(
                            policy_violations
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
            outcome = journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
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
                "summary": outcome.get("summary") if isinstance(outcome, dict) else None,
                "next_directions": outcome.get("next_directions") if isinstance(outcome, dict) else None,
                "verification": verification.as_dict() if verification else None,
            }
            valid_blocked = status == "blocked" and not violation
            if valid_blocked:
                retry_of = state.attempts[-1] if self._blocked_retry_pending(state) else None
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
            main_adapter.notify_iteration(
                self.base_campaign,
                memory_version,
                memory,
                accepted,
                verification.incumbent_latency_us if verification else None,
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
                self._pad_terminal_block_for_aggregation(attempt)
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
