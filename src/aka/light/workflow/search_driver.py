"""Reuse the existing serial Episode engine in isolated trajectory repositories.

Each trajectory keeps its own Journal, measurement/retry scope and adaptive
state. Only accepted Kernel bytes and authoritative metrics cross the barrier.
These are private controller repositories, not Agent-visible Git workspaces.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Event

import aka.light.workflow.context as main_adapter
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workspace.git import git_blob, git_head
from aka.legacy.task.persistence import durable_write_json, durable_replace
from aka.light.workspace.versions import version_record_path
from .search import latency, read_state
from .search_git import commit_files, git
from aka.light.workspace.sources import from_git, unpack, snapshot as source_snapshot


class EpisodeSearchDriver:
    def __init__(self, campaign, root: Path):
        self.campaign, self.root = campaign, root
        self.children = {}
        self.cancel_event = Event()

    def identity(self):
        # Store only a digest of settings (endpoint addresses remain private).
        values = {"request": asdict(self.campaign.request),
                  "workflow": asdict(self.campaign.module_options),
                  "agent": asdict(self.campaign.agent.options),
                  "inherit_state": self.campaign.candidate_workspace.inherit_state,
                  "research_records": self.campaign.research_records,
                  "wiki": self.campaign.tool_runtime.wiki,
                  "pause_agent_during_requests": (
                      self.campaign.tool_runtime.pause_agent_during_requests
                  ),
                  "repetitions": self.campaign.tool_runtime.measurement_repetitions,
                  "evaluation": {name: getattr(self.campaign, name) for name in (
                      "sandbox_hardware", "sandbox_profile", "sandbox_url", "sandbox_ssh",
                      "sandbox_timeout", "verify_repeats", "verify_run_timeout", "min_improvement_pct")}}
        return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()

    def bootstrap(self):
        main_adapter.prepare_campaign(self.campaign)
        # Do not reinterpret a previously serial campaign as pooled search.
        state_file = self.campaign.workspace / ".atrex_long_horizon/state.json"
        if state_file.exists() and read_state(state_file).get("episodes", 0):
            raise ValueError("Start pooled search in a new Campaign; serial Episode checkpoints are not migrated")
        active = self.campaign.workspace / ".atrex_long_horizon/active_episode.json"
        if active.exists():
            raise ValueError("Cannot enable pooled search with an active serial Episode")
        version = main_adapter.latest_version(self.campaign.workspace)
        record = main_adapter.read_memory(self.campaign.workspace, version)
        if not isinstance(record, dict) or record.get("quality_gate", {}).get("result") != "PASS":
            raise RuntimeError("Pooled search requires a verified Bootstrap/Framework baseline")
        return self._kernel(self.campaign.workspace, git_head(self.campaign.workspace), record, 0, 0)

    def _kernel(self, workspace, commit, record, trajectory, episode):
        source = from_git(workspace, commit)
        value = {"workspace": str(workspace), "commit": commit,
                 "sha256": hashlib.sha256(source).hexdigest(), "record": record,
                 "trajectory": trajectory, "episode": episode,
                 "latency_us_geomean": record["performance"]["latency_us_geomean"]}
        latency(value)
        # Keep selected historical candidates reachable after Episode worktree cleanup.
        git(workspace, "update-ref", f"refs/atrex-search/kernels/{value['sha256']}/{commit}", commit)
        return value

    def source(self, kernel):
        workspace = Path(kernel["workspace"])
        allowed = {self.campaign.workspace.resolve(), *(child.workspace.resolve() for child in self.children.values())}
        if workspace.resolve() not in allowed:
            raise ValueError("Search Kernel belongs to an unknown controller workspace")
        source = from_git(workspace, kernel["commit"])
        if hashlib.sha256(source).hexdigest() != kernel["sha256"]:
            raise RuntimeError("Search Kernel bytes differ from the sealed checkpoint")
        return source

    def initialize(self, bootstrap, trajectories):
        for number in range(1, trajectories + 1):
            workspace = self.root / "trajectories" / f"t{number:04d}"
            marker = workspace / ".git/atrex-search-origin.json"
            expected = {"bootstrap": bootstrap["commit"], "trajectory": number,
                        "campaign": str(self.campaign.workspace)}
            if not workspace.exists():
                workspace.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with tempfile.TemporaryDirectory(prefix=".preparing-", dir=workspace.parent) as temporary:
                    staged = Path(temporary) / "repository"
                    git(workspace.parent, "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                        str(self.campaign.workspace), str(staged))
                    git(staged, "checkout", "--quiet", "--detach", bootstrap["commit"])
                    git(staged, "remote", "remove", "origin")
                    git(staged, "config", "user.name", "GPU Kernel Optimizer")
                    git(staged, "config", "user.email", "gpu-kernel-optimizer@local")
                    durable_write_json(staged / ".git/atrex-search-origin.json", expected)
                    durable_replace(staged, workspace)
            if read_state(marker) != expected:
                raise RuntimeError("Trajectory workspace does not match its frozen Bootstrap identity")
            child = copy.copy(self.campaign)
            child.request = replace(self.campaign.request, workspace=workspace)
            child._supervisor_runtime = None
            child._production_review_cache = {}
            child.workspace_suffix = f"trajectory-{number:04d}"
            child.target_util = 0.0  # Search budget, not a local utilization stop, owns admission.
            child.tokens_spent = 0
            child.search_coordinate = None
            child.search_cancel = self.cancel_event
            self.children[number] = child

    def _publish(self, workspace, epoch, kernel, label):
        source = self.source(kernel)
        if workspace.resolve() != Path(kernel["workspace"]).resolve():
            # Keep the provenance commit named in version facts locally readable,
            # without merging the other trajectory's tree or controller history.
            git(workspace, "fetch", "--quiet", "--no-tags", "--no-write-fetch-head",
                kernel["workspace"], kernel["commit"])
        git(workspace, "update-ref",
            f"refs/atrex-search/kernels/{kernel['sha256']}/{kernel['commit']}", kernel["commit"])
        receipt_path = self.root / "publications" / f"e{epoch:08d}-{label}.json"
        # Pin version and full publication content before the Git transaction so
        # a committed-but-unacknowledged publication does not allocate a new vN.
        try:
            plan = read_state(receipt_path)
        except FileNotFoundError:
            version = main_adapter.latest_version(workspace) + 1
            record = copy.deepcopy(kernel["record"])
            record.update(version=f"v{version}", git_commit_hash=kernel["commit"],
                          long_horizon={"status": "search_broadcast", "search_epoch": epoch,
                                        "trajectory": kernel["trajectory"], "episode": kernel["episode"]})
            _, next_files = unpack(source)
            _, previous_files = unpack(source_snapshot(workspace))
            plan = {"version": version, "sha256": kernel["sha256"], "record": record,
                    "removed_sources": sorted(previous_files.keys() - next_files.keys())}
            durable_write_json(receipt_path, plan)
        if plan["sha256"] != kernel["sha256"]:
            raise RuntimeError("Epoch broadcast winner changed during recovery")
        private_name = f".atrex_versions/v{plan['version']}.json"
        record = (json.dumps(plan["record"], sort_keys=True) + "\n").encode()
        _, files = unpack(source)
        files.update({name: None for name in plan.get("removed_sources", [])})
        files[private_name] = record
        if self.campaign.generate_memory:
            files[f"memory/v{plan['version']}.json"] = record
        return commit_files(workspace, receipt_path.with_suffix(".git.json"), files,
                            f"search epoch {epoch}: {label} Kernel {kernel['sha256'][:16]}")

    def prepare_epoch(self, epoch, kernel):
        for number, child in self.children.items():
            store = self._state(child)
            expected = (epoch - 1) * self.campaign.module_options.search.episodes_per_trajectory
            if store["episodes"] != expected or (child.workspace / ".atrex_long_horizon/active_episode.json").exists():
                raise RuntimeError("Cannot broadcast while a trajectory has an unfinished or unexpected Episode")
            self._publish(child.workspace, epoch, kernel, f"trajectory-{number:04d}")

    def _state(self, child):
        path = child.workspace / ".atrex_long_horizon/state.json"
        try:
            state = read_state(path)
        except FileNotFoundError:
            # The very first Episode writes its active marker/brief before its
            # first usage checkpoint. That interrupted preparation is resumable;
            # lost accounting for terminal/later Episodes is not a fresh start.
            control = child.workspace / ".atrex_long_horizon"
            archives = control / "episodes"
            if archives.exists() and any(archives.iterdir()):
                active = read_state(control / "active_episode.json")
                if (active.get("episode") != 1 or any(archives.glob("*/attempt.json"))
                        or any(path.name != "e0001" for path in archives.iterdir())):
                    raise RuntimeError("Trajectory state is missing; restore it before resuming")
            return {"episodes": 0, "tokens": 0}
        for name in ("episodes", "tokens"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError("Invalid trajectory Episode/usage checkpoint")
        return state

    def run_episode(self, epoch, trajectory, episode):
        child = self.children[trajectory]
        current = self._state(child)["episodes"]
        if current not in {episode - 1, episode}:
            raise RuntimeError("Trajectory Episode sequence differs from the Epoch schedule")
        child.search_coordinate = {"epoch": epoch, "trajectory": trajectory, "episode": episode}
        # An archived terminal Episode is not run again after coordinator restart.
        if current < episode or (child.workspace / ".atrex_long_horizon/active_episode.json").exists():
            child.start_runtime()
            from aka.light.workflow.session import LongSessionRunner
            runner = LongSessionRunner(agent=child.agent, agent_cli=child.agent_cli,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0)
            runner.cancel_event = self.cancel_event
            verifier = child.acceptance_verifier(
                execute=child.measure_for_acceptance, hardware=child.sandbox_hardware,
                profile=child.sandbox_profile, url=child.sandbox_url, ssh=child.sandbox_ssh,
                ssh_init=child.sandbox_ssh_init, health_command=child.sandbox_health_command,
                timeout=child.sandbox_timeout, repeats=child.verify_repeats,
                per_run_timeout=child.verify_run_timeout, min_improvement_pct=child.min_improvement_pct,
                private_reference_dir=child.private_reference_dir)
            LongHorizonCampaign(base_campaign=child, max_episodes=episode,
                                handoff_resumes=child.handoff_resumes, verifier=verifier,
                                session_runner=runner).run()
        if self._state(child)["episodes"] != episode:
            raise RuntimeError("Episode did not reach a durable terminal outcome")
        attempt = read_state(child.workspace / f".atrex_long_horizon/episodes/e{episode:04d}/attempt.json")
        if attempt.get("episode") != episode:
            raise RuntimeError("Episode archive identity mismatch")
        retained = None
        if attempt.get("accepted") is True:
            if attempt.get("verification", {}).get("gate") != "PASS":
                raise RuntimeError("Accepted Episode lacks a passing authoritative comparison")
            commit = attempt["promotion_commit"]
            record_path = version_record_path(child.workspace, attempt["version"])
            record = json.loads(git_blob(child.workspace, commit, record_path.relative_to(child.workspace).as_posix()))
            if record.get("quality_gate", {}).get("result") != "PASS":
                raise RuntimeError("Accepted Episode lacks a passing sealed version record")
            if (attempt.get("candidate_commit") and from_git(child.workspace, commit)
                    != from_git(child.workspace, attempt["candidate_commit"])):
                raise RuntimeError("Promoted search Kernel differs from the accepted Episode candidate")
            retained = self._kernel(child.workspace, commit, record, trajectory, episode)
        return {"epoch": epoch, "trajectory": trajectory, "episode": episode,
                "status": attempt["status"], "retained_kernel": retained,
                "archive": str(child.workspace / f".atrex_long_horizon/episodes/e{episode:04d}"),
                "tokens": attempt.get("tokens", 0)}

    def publish_epoch(self, epoch, kernel):
        self._publish(self.campaign.workspace, epoch, kernel, "campaign")

    def tokens(self):
        return sum(self._state(child)["tokens"] for child in self.children.values())

    def stop_reason(self, state):
        target = self.campaign.target_util
        if target > 0 and main_adapter.peak_util(state["seed"]["record"]) >= target:
            return "success: target utilization reached at Epoch boundary"
        limit = self.campaign.max_stall
        if (limit and not self.campaign.module_options.goal_episodes
                and state["consecutive_without_improvement"] >= limit):
            return "stall: Epochs without a better retained Kernel"
        return ""

    def interrupt(self):
        # Stop further HTTP work. Running Sessions retain their normal bounded
        # process lifecycle and unfinished Episodes retain their recovery points.
        self.cancel_event.set()
        for child in self.children.values():
            child.close_runtime()

    def close(self):
        self.interrupt()
