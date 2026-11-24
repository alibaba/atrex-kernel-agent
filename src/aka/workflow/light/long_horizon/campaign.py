from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aka.workflow.light.orchestration.hardware import hardware_vendor
from aka.task.source.formats import from_git as source_from_git

import aka.workflow.light.long_horizon.main_adapter as main_adapter
from aka.task.git import git_head, git_text
from aka.workflow.light.long_horizon.models import SupervisorState
from aka.workflow.light.long_horizon.session import LongSessionRunner
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.light.long_horizon.telemetry import summarize_episode
from aka.agent.execution.home import SESSION_HOME_ROOT_ENV, SESSION_HOME_ID_ENV

from aka.resources.bundles import resource_path
PROMPTS_DIR = resource_path("prompts/research-episodes")
PROMPT_PATH = PROMPTS_DIR / "episode.md"
GOAL_AFTER_EPISODES = 50


def _session_terminal_status(result: Any) -> str:
    failure_kind = getattr(result, "terminal_failure_kind", None)
    if failure_kind == "dependency_policy_violation":
        return "blocked"
    if failure_kind == "episode_budget_exhausted":
        return "episode_budget_exhausted"
    if getattr(result, "exit_status", 0) == 0 and not getattr(result, "timed_out", False):
        return "completed"
    return "session_failed"


def _render(template: str, values: dict[str, object]) -> str:
    required = set(re.findall(r"\{\{([A-Z][A-Z0-9_]*)\}\}", template))
    missing = required - set(values)
    if missing:
        raise ValueError(
            "Episode prompt has unresolved fields: " + ", ".join(sorted(missing))
        )
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    while "\n\n\n" in template:
        template = template.replace("\n\n\n", "\n\n")
    return template.strip() + "\n"










@dataclass(frozen=True)
class SegmentBinding:
    """One audit segment over the persistent Agent repository."""

    episode: int
    base_commit: str
    branch: str
    path: Path


@dataclass
class LongHorizonCampaign:
    base_campaign: main_adapter.Campaign
    max_episodes: int = 8
    episode_limit: int = 0
    token_budget: int = 0
    session_runner: LongSessionRunner | None = None


    @property
    def workspace(self) -> Path:
        return self.base_campaign.workspace

    def _episode_mode(self, state: SupervisorState, active: dict[str, Any] | None = None) -> str:
        """Goal widens search, never changes the Journal or acceptance contract."""
        if active is not None:
            # Do not widen a running or legacy Episode after a restart.
            return "goal" if active.get("mode") == "goal" else "episode"
        return "goal" if (
            self.base_campaign.module_options.goal_episodes
            and state.episodes >= GOAL_AFTER_EPISODES
        ) else "episode"

    @staticmethod
    def _continuation_delta(state: SupervisorState) -> str:
        """Render a bounded authoritative settlement for the next Episode."""
        if not state.attempts:
            return ""
        previous = state.attempts[-1]
        delta = {
            "episode": previous.get("episode"),
            "status": previous.get("status"),
            "violation": previous.get("violation"),
            "agent_head": previous.get("agent_head"),
        }
        encoded = json.dumps(delta, sort_keys=True, ensure_ascii=False, default=str)
        if len(encoded) > 4_000:
            encoded = encoded[:3_900] + "...<bounded>"
        return (
            "Continue the SAME trajectory and native provider Session into a new Episode. "
            "Do not restart optimization from first principles or replay all history. "
            "The summary below describes only the previous audit segment; accepted "
            "submit-kernel receipts, local Git, and saved Gateway Records remain authoritative. "
            "Inspect the current trajectory state and continue the highest-value unfinished work.\n"
            f"Previous segment summary: {encoded}\n\n"
        )

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
        segment: SegmentBinding,
        agent_workspace: Path | None = None,
        episode_mode: str = "episode",
    ) -> str:
        directives = main_adapter.episode_directives(self.base_campaign)
        comparison_base_commit = segment.base_commit
        measurement_context = None
        if agent_workspace is not None:
            # Every commit exposed to the Agent belongs to this persistent
            # trajectory repository, so it can be inspected, diffed, checked
            # out, and supplied to Runtime directly.
            measurement_context = self.base_campaign.acceptance_measurement_context(
                agent_workspace,
                source_from_git(segment.path, segment.base_commit),
            )
            comparison_base_commit = measurement_context["baseline_kernel_id"]
        research_modules = set(self.base_campaign.research_modules)
        if research_modules == {"directions", "experiments"}:
            module_fragment, handoff_fragment = "episode_research.md", "runtime_submit.md"
        elif research_modules == {"directions"}:
            module_fragment, handoff_fragment = "episode_directions.md", "runtime_submit.md"
        elif research_modules == {"experiments"}:
            module_fragment, handoff_fragment = "episode_experiments.md", "runtime_submit.md"
        else:
            module_fragment, handoff_fragment = "episode_unrecorded.md", "runtime_submit.md"
        fragments = {
            "RESEARCH_MODULE_RULES": module_fragment,
            "SUBMISSION_CONTRACT": handoff_fragment,
            "PPU_DIRECTIVE": "",
        }
        if (
            hardware_vendor(self.base_campaign.platform, self.base_campaign.arch) == "ppu"
        ):
            fragments["PPU_DIRECTIVE"] = "episode_ppu.md"
        template = PROMPT_PATH.read_text(encoding="utf-8")
        coordinate = getattr(self.base_campaign, "search_coordinate", None)
        if coordinate is not None:
            role = ""
            if coordinate.get("role_specialization", False):
                trajectory_two_role = (
                    "T2 role: continue the highest-priority Direction marked as a major redesign "
                    "until its stop condition, blocker, or completion; then select the next "
                    "high-value major redesign."
                    if "directions" in research_modules
                    else "T2 role: continue the highest-value major redesign until its stop "
                    "condition, blocker, or completion; then select the next high-value redesign."
                )
                role = " " + {
                    1: "T1 role: bounded exploitation around the global or lineage best.",
                    2: trajectory_two_role,
                    3: "T3 role: capability probes, profiling, and independent exploration.",
                }.get(
                    coordinate["trajectory"],
                    "Additional-island role: preserve an independent line of evidence-backed exploration.",
                )
            template = (
                f"Your trajectory: T{coordinate['trajectory']}. Search location: Epoch "
                f"{coordinate['epoch']}, Episode {coordinate['episode']}. This persistent island "
                "keeps its working tree, Git history, "
                + (
                    "lineage best, active Directions and major-redesign WIP across Epochs. "
                    if "directions" in research_modules
                    else "lineage best and unfinished optimization WIP across Epochs. "
                )
                + "The Runtime updates accepted-submission winner metadata and imports peer Git refs "
                + (
                    "after each completed audit segment "
                    if coordinate.get("synchronization") == "asynchronous"
                    else "at synchronous Epoch barriers "
                )
                + "without overwriting local work. Adopt peer work explicitly and validate it normally."
                f"{role}\n\n"
                + template
            )
        # Insert fragments before substituting values so nested task placeholders resolve too.
        for marker, name in fragments.items():
            text = (PROMPTS_DIR / name).read_text(encoding="utf-8").strip() if name else ""
            template = template.replace("{{" + marker + "}}", text)
        return _render(
            template,
            {
                "EPISODE": episode,
                "GOAL_DIRECTIVE": (
                    "This goal-oriented Episode follows sustained stalls. Reassess the roadmap and test "
                    "materially different high-value approaches while preserving the measured best in Git. "
                    "All ordinary research, measurement, and submit-kernel rules remain unchanged."
                    if episode_mode == "goal" else ""
                ),
                "BASE_COMMIT": comparison_base_commit,
                "WORKSPACE": agent_workspace or segment.path,
                "ARCH": self.base_campaign.arch or "<authoritative runtime architecture>",
                "DSL_CONTEXT": self.base_campaign.dsl_context,
                "NOTES": self.base_campaign.notes,
                "PLUGINS": directives["plugins"],
                "REFERENCE_KNOWLEDGE": getattr(
                    self.base_campaign, "reference_knowledge_prompt", lambda: ""
                )(),
                "RUNTIME_WAIT_POLICY": (
                    "- Runtime requests: Agent and subagent execution is suspended while an "
                    "authenticated request is pending; this wait is excluded from the configured "
                    "active-time budget."
                    if self.base_campaign.tool_runtime.pause_agent_during_requests
                    else "- Runtime requests do not suspend Agent execution. Independent work and Runtime calls may "
                    "continue while a request is pending; collect its completed receipt before "
                    "making decisions that depend on its result. Request time counts toward any "
                    "configured Agent time limit."
                ),
            },
        )




    def _record_session_segment(
        self,
        store: CampaignStore,
        state: SupervisorState,
        active: dict[str, Any],
        segment: SegmentBinding,
        *,
        status: str,
        agent_workspace: Path,
        session_id: str,
        resume_count: int,
        tokens: int,
        invocations: tuple[Any, ...],
        output_limit_recoveries: int,
        dependency_policy_recoveries: int,
        violation: str = "",
    ) -> None:
        """Close an audit segment without sealing, promoting, or rewriting Git.

        Agent Git and accepted ``submit-kernel`` receipts are authoritative.  An
        Episode boundary records provider/session telemetry only; it does not
        nominate a candidate or create a second Supervisor Kernel identity.
        """
        episode = segment.episode
        state.episodes = max(state.episodes, episode)
        candidate_commit = git_text(
            agent_workspace, "rev-parse", "HEAD", check=False
        )
        attempt = {
            "episode": episode,
            "mode": self._episode_mode(state, active),
            "status": status,
            "violation": violation or None,
            "agent_head": candidate_commit or None,
            "session_id": session_id or None,
            "resume_count": max(0, int(resume_count)),
            "tokens": max(0, int(tokens)),
            "output_limit_recoveries": max(0, int(output_limit_recoveries)),
            "dependency_policy_recoveries": max(
                0, int(dependency_policy_recoveries)
            ),
        }
        telemetry = summarize_episode(
            episode=episode,
            version=int(active.get("memory_version", episode) or episode),
            status=status,
            accepted=False,
            control_tokens=max(0, int(tokens)),
            resume_count=max(0, int(resume_count)),
            invocations=invocations,
        )
        telemetry_path = store.archive_telemetry(episode, telemetry)
        attempt["telemetry"] = {
            "summary": str(telemetry_path),
            "measurement": telemetry["measurement"],
            "reason_codes": telemetry["reason_codes"],
        }
        store.archive_attempt(episode, attempt)
        state.attempts.append(attempt)
        if session_id:
            state.session_id = session_id
            state.session_backend = getattr(
                self.base_campaign, "agent_cli", "claude"
            )
        store.save_state(state)
        store.clear_active()
        print(
            f"[long-horizon] episode={episode} status={status} "
            f"tokens={max(0, int(tokens))} agent_head={candidate_commit or '-'}",
            flush=True,
        )

    def _recover_session_segment(
        self,
        store: CampaignStore,
        state: SupervisorState,
    ) -> tuple[SegmentBinding, dict[str, Any]] | None:
        """Resume an interrupted audit segment without terminal-handoff state."""
        active = store.load_active()
        if active is None:
            return None
        episode = int(active.get("episode", 0) or 0)
        base_commit = str(active.get("starting_commit_id", ""))
        if episode <= 0 or not base_commit or not (self.workspace / ".git").is_dir():
            raise RuntimeError(
                "Active Episode checkpoint cannot be resumed because its persistent "
                "Agent repository identity is incomplete"
            )
        try:
            source_from_git(self.workspace, base_commit)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "Active Episode starting commit is no longer reachable from Agent Git"
            ) from error
        branch = git_text(
            self.workspace,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        ) or "work"
        handle = SegmentBinding(episode, base_commit, branch, self.workspace)
        active["phase"] = "exploring"
        active["restart_count"] = int(active.get("restart_count", 0) or 0) + 1
        store.save_active(active)
        print(
            f"[long-horizon] resuming interrupted episode={episode} in persistent Agent workspace",
            flush=True,
        )
        return handle, active



    def run(self) -> str:
        main_adapter.prepare_campaign(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        recovered_episode = self._recover_session_segment(store, state)
        runner = self.session_runner or LongSessionRunner(
            agent=self.base_campaign.agent,
            agent_cli=getattr(self.base_campaign, "agent_cli", "claude"),
            usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0,
        )
        starting_episodes = state.episodes
        reason = "max-episodes"

        while True:
            if state.episodes >= self.max_episodes:
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
                reason = "budget: token-budget"
                break
            resumed = recovered_episode is not None
            if recovered_episode is not None:
                segment, active = recovered_episode
                recovered_episode = None
                episode = segment.episode
                memory_version = int(active["memory_version"])
                base_commit = segment.base_commit
                episode_mode = self._episode_mode(state, active)
            else:
                episode = state.episodes + 1
                episode_mode = self._episode_mode(state)


            if resumed:
                active.setdefault("mode", episode_mode)
                if active.get("resumed_from_phase") == "preparing":
                    main_adapter.link_episode_runtime(
                        self.base_campaign, segment.path
                    )
            else:
                memory_version = episode
                base_commit = git_head(self.workspace)
                if not base_commit:
                    raise RuntimeError("Persistent Agent workspace has no Git HEAD")
                branch = git_text(
                    self.workspace,
                    "symbolic-ref",
                    "--quiet",
                    "--short",
                    "HEAD",
                    check=False,
                ) or "work"
                segment = SegmentBinding(
                    episode, base_commit, branch, self.workspace
                )
                active = {
                    "episode": episode,
                    "memory_version": memory_version,
                    "starting_commit_id": base_commit,
                    "mode": episode_mode,
                    "phase": "preparing",
                }
                store.save_active(active)
                active.update(
                    {
                        "phase": "exploring",
                    }
                )
                store.save_active(active)
                main_adapter.link_episode_runtime(self.base_campaign, segment.path)
            # The Agent receives one persistent Git repository. Runtime Journal
            # stores measurements, research records and submit-kernel receipts;
            # ending this segment requires no handoff or Supervisor Git seal.
            session_capture_root = store.episode_dir(episode) / "sessions"
            agent_workspace = self.base_campaign.register_runtime_episode(
                segment.path, episode=episode, memory_version=memory_version,
                base_commit=base_commit, branch=segment.branch,
                baseline_kernel_commit_id=None,
                require_gluon=False,
                session_capture_root=session_capture_root,
            )
            prompt = self._prompt(
                episode=episode,
                segment=segment,
                agent_workspace=agent_workspace,
                episode_mode=episode_mode,
            )
            agent_cli = getattr(self.base_campaign, "agent_cli", "claude")
            from aka.agent.native import supports_same_session_resume

            resume_supported = supports_same_session_resume(agent_cli)
            if state.episodes > 0 and not resume_supported:
                raise RuntimeError(
                    f"{agent_cli} cannot resume the persistent optimization Session; "
                    "use a provider with native Session resume or run a single segment"
                )
            continuing_session_id = (
                state.session_id
                if resume_supported and state.session_backend == agent_cli
                else ""
            )
            if state.episodes > 0 and not continuing_session_id:
                raise RuntimeError(
                    "The prior optimization segment has no resumable provider Session ID"
                )
            if continuing_session_id:
                prompt = self._continuation_delta(state) + prompt
            store.write_brief(episode, prompt)
            telemetry_environment = {
                "ATREX_EPISODE_MODE": episode_mode,
                "ATREX_SESSION_CAPTURE_DIR": str(session_capture_root),
                "ATREX_TELEMETRY_CAMPAIGN_ID": str(
                    getattr(self.base_campaign, "campaign_name", self.workspace.name)
                ),
                "ATREX_TELEMETRY_ITERATION_ID": f"episode-{episode:04d}",
                "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
            }
            telemetry_environment.update(
                self.base_campaign.agent_environment()
            )
            telemetry_environment.update(
                self.base_campaign.plugin_registry.environment(agent_workspace)
            )
            # Both the Agent repository and provider transcript are stable for
            # the whole trajectory. Measurement and research records remain
            # independently durable across audit segments.
            telemetry_environment[SESSION_HOME_ROOT_ENV] = str(
                store.root / "agent-session-homes"
            )
            if agent_cli == "codex":
                # Codex allocates its thread ID after launch. The trajectory's
                # native storage must not change when that ID becomes known.
                telemetry_environment[SESSION_HOME_ID_ENV] = "optimization-session"
            usage_receipt = uuid.uuid4().hex
            with self.base_campaign._supervisor_runtime.session(
                agent_workspace, telemetry_environment,
                access=self.base_campaign.agent_access(agent_workspace),
            ) as grant:
                result = runner.run(
                    agent_workspace,
                    prompt,
                    output_directory=store.episode_dir(episode),
                    telemetry_environment=grant.environment,
                    access=grant.filesystem_access,
                    record_session=lambda identity: store.record_session(
                        state, session_id=identity, backend=agent_cli
                    ),
                    record_usage=lambda tokens: store.record_usage(
                        state, usage_receipt, tokens
                    ),
                    remaining_token_budget=(
                        max(0, self.token_budget - state.tokens) if self.token_budget else None
                    ),
                    session_id=continuing_session_id,
                    resume_session=bool(continuing_session_id),
                    initial_purpose=(
                        "trajectory_continuation"
                        if continuing_session_id
                        else "optimization"
                    ),
                    **main_adapter.session_execution_policy(self.base_campaign),
                )
            # Also persist before verification: a GPU outage there must not
            # discard the coding session's usage. Receipt replay is harmless.
            store.record_usage(state, usage_receipt, result.tokens)
            status = _session_terminal_status(result)
            violation = ""
            if result.exit_status != 0 or result.timed_out:
                violation = f"session failed: exit={result.exit_status} timeout={result.timed_out}"
                if result.completion_diagnosis:
                    violation += f"; {result.completion_diagnosis}"
            self._record_session_segment(
                store,
                state,
                active,
                segment,
                status=status,
                violation=violation,
                agent_workspace=agent_workspace,
                session_id=result.session_id,
                resume_count=result.resume_count,
                tokens=result.tokens,
                invocations=result.invocations,
                output_limit_recoveries=result.output_limit_recoveries,
                dependency_policy_recoveries=(
                    result.dependency_policy_recoveries
                ),
            )

        print(
            f"[long-horizon] STOP {reason}; episodes={state.episodes} "
            f"episode_budget_exhausted={state.episode_budget_exhausted} "
            f"interrupted={state.interrupted} tokens={state.tokens}",
            flush=True,
        )
        return reason
