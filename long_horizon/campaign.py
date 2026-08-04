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


def _render(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def _iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@dataclass
class LongHorizonCampaign:
    base_campaign: main_adapter.Campaign
    max_episodes: int = 8
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
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff_path: Path,
        state: SupervisorState,
    ) -> str:
        directives = main_adapter.episode_directives(self.base_campaign)
        journal_command = f"PYTHONPATH={MODULE_ROOT} python -m long_horizon.journal"
        return _render(
            PROMPT_PATH.read_text(encoding="utf-8"),
            {
                "EPISODE": episode,
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
                "latency_us_by_shape": by_shape if isinstance(by_shape, dict) else {},
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
            "correctness": {"status": "PASS"},
            "quality_gate": {"result": "PASS", "failure_reason": None},
            "open_directions": [
                {"direction": value, "rationale": "carried from terminal episode journal"}
                for value in directions if isinstance(value, str)
            ],
            "git_commit_hash": candidate_commit,
        }

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
            if not (
                phase in {"promoting", "promoted"}
                and parent == base_commit
                and message == f"episode {episode}: promote verified long-horizon candidate"
                and bool(evidence)
            ):
                raise RuntimeError("incumbent advanced during an interrupted episode without proof")
            if not already_recorded:
                state.episodes = max(state.episodes, episode)
                state.accepted += 1
                state.consecutive_without_promotion = 0
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
        store.save_state(state)
        store.clear_active()

    def run(self) -> str:
        main_adapter.prepare_campaign(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        self._recover_interrupted(store, state)
        if working_changes(self.workspace):
            raise RuntimeError(
                "long-horizon campaign requires a clean incumbent workspace: "
                + ", ".join(working_changes(self.workspace)[:12])
            )
        verifier = self.verifier or GatewayABBAValidator(
            hardware=self.base_campaign.sandbox_hardware,
            profile=self.base_campaign.sandbox_profile,
            url=self.base_campaign.sandbox_url,
            timeout=self.base_campaign.sandbox_timeout,
        )
        runner = self.session_runner or LongSessionRunner()
        reason = "max-episodes"

        while state.episodes < self.max_episodes:
            if self.token_budget and state.tokens >= self.token_budget:
                reason = "token-budget"
                break
            episode = state.episodes + 1
            base_commit = git_head(self.workspace)
            worktree = EpisodeWorktree.plan(
                self.workspace, episode, base_commit, root=self.worktree_root
            )
            active = {
                "episode": episode,
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
                worktree=worktree,
                journal_path=journal_path,
                handoff_path=handoff_path,
                state=state,
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
                if not violation:
                    active["phase"] = "verifying"
                    store.save_active(active)
                    verification = verifier.verify(
                        worktree.path,
                        base_commit=base_commit,
                        candidate_commit=candidate_commit,
                        changed_paths=paths,
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
            promotion_commit = ""
            if accepted and verification is not None:
                active["phase"] = "promoting"
                store.save_active(active)
                memory_version = main_adapter.latest_version(self.workspace) + 1
                evidence = {**attempt, "journal": journal}
                promotion_commit = promote_candidate(
                    self.workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    episode=episode,
                    evidence=evidence,
                    memory_version=memory_version,
                    memory_record=self._memory_record(
                        version=memory_version,
                        candidate_commit=candidate_commit,
                        journal=journal,
                        verification=verification,
                    ),
                )
                attempt["promotion_commit"] = promotion_commit
                state.accepted += 1
                state.consecutive_without_promotion = 0
                active["phase"] = "promoted"
                active["promotion_commit"] = promotion_commit
                store.save_active(active)
            else:
                state.consecutive_without_promotion += 1
                if status == "pivot" and not violation:
                    state.pivoted += 1
                elif status == "blocked" and not violation:
                    state.blocked += 1
                elif status == "invalid_handoff":
                    state.protocol_failures += 1
                else:
                    state.rejected += 1
            store.archive_attempt(episode, attempt)
            state.attempts.append(attempt)
            store.save_state(state)
            worktree.remove(self.workspace)
            store.clear_active()
            print(
                f"[long-horizon] episode={episode} status={status} accepted={accepted} "
                f"tokens={result.tokens} promotion={promotion_commit or '-'}",
                flush=True,
            )
            if status == "blocked" and not violation:
                reason = "blocked"
                break
            if self.max_stall and state.consecutive_without_promotion >= self.max_stall:
                reason = f"max-stall:{state.consecutive_without_promotion}"
                break

        print(
            f"[long-horizon] STOP {reason}; episodes={state.episodes} accepted={state.accepted} "
            f"rejected={state.rejected} pivoted={state.pivoted} blocked={state.blocked} "
            f"protocol_failures={state.protocol_failures} tokens={state.tokens}",
            flush=True,
        )
        return reason
