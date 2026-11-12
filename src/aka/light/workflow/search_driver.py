"""Reuse the serial Session engine in isolated persistent Agent trajectories.

Completed Epochs publish immutable research, measurements and peer Agent Git.
Lineage/global winners are derived from accepted ``submit-kernel`` commits with
passing full Evaluate evidence and broadcast as observations. They never select
the next checkout: every trajectory keeps its own HEAD and provider Session.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Lock

import aka.light.workflow.context as main_adapter
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workspace.git import git_head
from aka.legacy.task.persistence import durable_write_json, durable_replace
from aka.light.workspace.baseline import read_baseline_record, write_baseline_record
from .search import read_state
from .search_git import git
from aka.light.workspace.sources import from_git
from aka.light.experiment.store import MeasurementStore, digest as evidence_digest, private_write
from aka.light.experiment.frontier import kernel_pareto_frontier
from aka.light.runtime.paths import supervisor_campaign_root
from aka.light.workflow.journal_service import (
    accepted_submission_kernel_ids,
    copy_experiment_reference_archive,
    load_journal,
    research_projection,
    shared_research_path,
)
from aka.light.workspace.shared_history import export_epoch, import_exports
from aka.light.workspace.agent_git import AgentGit
from aka.light.research import ResearchStore


_OPEN_DIRECTION_STATUSES = {"proposed", "in_progress", "deferred", "blocked"}


def _shareable_claim(item):
    """Keep explicit low-quality reviews trajectory-local; preserve legacy claims."""
    quality = item.get("claim_review_quality", item.get("review_quality"))
    return quality not in {
        "partially_supported", "unreviewed", "not_reviewed"
    }


def _aggregate_direction_queue(research_states, directions, inherited_queue=()):
    """Merge trajectory-local rankings without letting the last writer win.

    Rank is normalized inside each trajectory before averaging so a longer
    local queue does not carry a larger numeric penalty.  A trajectory that
    has not observed a newly proposed Direction casts no vote for it instead
    of implicitly ranking it last.  Deterministic ties prefer broader
    agreement, then the lower mean raw rank, then the stable Direction ID.
    """
    open_ids = {
        direction_id
        for direction_id, direction in directions.items()
        if direction.get("status") in _OPEN_DIRECTION_STATUSES
    }
    votes = {direction_id: [] for direction_id in open_ids}
    for trajectory, state in sorted(research_states.items()):
        # Only the trajectory's independently maintained writable queue is a
        # vote. ``shared_queue`` is the result of an earlier aggregation; using
        # it as another vote would feed the global order back into itself and
        # progressively drown out independent local judgments.
        raw_queue = state.get("queue", [])
        if not isinstance(raw_queue, list):
            raise RuntimeError(
                f"Trajectory {trajectory} has an invalid Direction queue"
            )
        queue = []
        for direction_id in raw_queue:
            if direction_id in open_ids and direction_id not in queue:
                queue.append(direction_id)
        denominator = max(len(queue) - 1, 1)
        for offset, direction_id in enumerate(queue):
            votes[direction_id].append((offset / denominator, offset + 1))

    inherited_rank = {
        direction_id: rank
        for rank, direction_id in enumerate(inherited_queue)
        if direction_id in open_ids
    }

    def rank(direction_id):
        observed = votes[direction_id]
        if not observed:
            # A previously published queue is only a deterministic tiebreaker
            # for Directions that received no fresh local vote. It is never
            # counted as another trajectory vote, so inherited ordering cannot
            # amplify itself across Epochs.
            return (
                1.0,
                0,
                float("inf"),
                inherited_rank.get(direction_id, float("inf")),
                direction_id,
            )
        mean_normalized = sum(item[0] for item in observed) / len(observed)
        mean_raw = sum(item[1] for item in observed) / len(observed)
        return (mean_normalized, -len(observed), mean_raw, float("inf"), direction_id)

    return sorted(open_ids, key=rank)


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
                  "research_modules": list(self.campaign.research_modules),
                  "wiki": self.campaign.tool_runtime.wiki,
                  "pause_agent_during_requests": (
                      self.campaign.tool_runtime.pause_agent_during_requests
                  ),
                  "repetitions": self.campaign.tool_runtime.measurement_repetitions,
                  "evaluation": {name: getattr(self.campaign, name) for name in (
                      "sandbox_hardware", "sandbox_profile", "sandbox_url", "sandbox_ssh",
                      "sandbox_timeout", "verify_repeats", "evaluation_step_timeout",
                      "gpu_clock_mhz", "min_improvement_pct")}}
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
            raise RuntimeError("Pooled search requires measured V0 evidence")
        seed = self._kernel(
            self.campaign.workspace,
            git_head(self.campaign.workspace),
            record,
            0,
            0,
        )
        canonical = self._canonical_agent_root(seed)
        seed["kernel_id"] = canonical["agent_v0_commit_id"]
        return seed

    @property
    def _canonical_agent_bundle(self) -> Path:
        return self.root / "shared" / "git" / "canonical-agent-root.bundle"

    @property
    def _canonical_agent_receipt(self) -> Path:
        return self.root / "shared" / "git" / "canonical-agent-root.json"

    @staticmethod
    def _bundle_contains_commit(bundle: Path, commit_id: str) -> bool:
        """Check a private recovery bundle without trusting its advertised refs."""
        if not bundle.is_file():
            return False
        with tempfile.TemporaryDirectory(prefix="aka-agent-root-check-") as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            git(workspace, "init", "--bare", "--quiet")
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "protocol.file.allow=always",
                    "fetch",
                    "--quiet",
                    str(bundle),
                    "+refs/heads/*:refs/heads/*",
                    "+refs/tags/*:refs/tags/*",
                    "+refs/atrex/*:refs/atrex/*",
                ],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                return False
            present = subprocess.run(
                ["git", "cat-file", "-e", f"{commit_id}^{{commit}}"],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
            return present.returncode == 0

    def _canonical_agent_root(self, bootstrap: dict) -> dict:
        """Create the one Runtime-authored V0 commit shared by all trajectories."""
        source = self.source(bootstrap)
        source_sha256 = hashlib.sha256(source).hexdigest()
        bundle = self._canonical_agent_bundle
        receipt_path = self._canonical_agent_receipt
        if bundle.is_file() and receipt_path.is_file():
            receipt = read_state(receipt_path)
            if not isinstance(receipt, dict):
                raise RuntimeError("Canonical Agent root receipt is invalid")
            if receipt.get("source_sha256") != source_sha256:
                raise RuntimeError(
                    "Canonical Agent root source differs from measured V0"
                )
            v0_commit = receipt.get("agent_v0_commit_id")
            if not isinstance(v0_commit, str):
                raise RuntimeError("Canonical Agent root receipt is incomplete")
            if not self._bundle_contains_commit(bundle, v0_commit):
                raise RuntimeError("Canonical Agent root bundle lost its V0 commit")
            return receipt
        if bundle.exists() != receipt_path.exists():
            raise RuntimeError("Canonical Agent V0 bundle and receipt are incomplete")

        from aka.light.workspace.sources import install

        with tempfile.TemporaryDirectory(prefix="aka-canonical-agent-v0-") as temporary:
            workspace = Path(temporary) / "repo"
            install(workspace, source)
            repository = AgentGit(workspace, bundle)
            repository.prepare()
            v0_commit = repository.persist()
            if not isinstance(v0_commit, str):
                raise RuntimeError("Supervisor could not create the canonical Agent V0")
        if not self._bundle_contains_commit(bundle, v0_commit):
            raise RuntimeError("Canonical Agent bundle does not contain its V0 root")
        receipt = {
            "schema_version": 2,
            "source_sha256": source_sha256,
            "agent_v0_commit_id": v0_commit,
            "controller_v0_commit_id": bootstrap["commit"],
        }
        durable_write_json(receipt_path, receipt)
        return receipt

    def _seed_trajectory_agent_root(self, child, canonical: dict) -> None:
        """Install the immutable canonical bundle before a child Runtime starts."""
        target_root = supervisor_campaign_root(child.workspace)
        target = target_root / "adaptive-state" / "agent-git.bundle"
        baseline_commit = canonical["agent_v0_commit_id"]
        if target.is_file():
            if not self._bundle_contains_commit(target, baseline_commit):
                raise RuntimeError(
                    "Existing trajectory Agent Git predates the canonical shared V0; "
                    "resume it with its original code or start a new pooled Campaign"
                )
            return
        agent_workspace = target_root / "agent-workspace"
        if (agent_workspace / ".git").is_dir():
            present = subprocess.run(
                ["git", "cat-file", "-e", f"{baseline_commit}^{{commit}}"],
                cwd=agent_workspace,
                capture_output=True,
                check=False,
            )
            if present.returncode != 0:
                raise RuntimeError(
                    "Existing trajectory Agent repository has a different V0 root"
                )
            return
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copyfile(self._canonical_agent_bundle, temporary)
            temporary.chmod(0o600)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            durable_replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _kernel(
        self, workspace, commit, record, trajectory, episode, kernel_id=None
    ):
        source = from_git(workspace, commit)
        latency_value = record.get("performance", {}).get("latency_us_geomean")
        if (
            type(latency_value) not in {int, float}
            or not math.isfinite(float(latency_value))
            or float(latency_value) <= 0
        ):
            raise ValueError("Measured V0 requires a finite positive GeoMean latency")
        value = {"workspace": str(workspace), "commit": commit,
                 "kernel_id": kernel_id or commit,
                 "sha256": hashlib.sha256(source).hexdigest(), "record": record,
                 "trajectory": trajectory, "episode": episode,
                 "latency_us_geomean": float(latency_value)}
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

    def _research(self, child, episode):
        root = supervisor_campaign_root(child.workspace)
        journal = root / "journals" / "episodes" / f"e{episode:08d}" / "journal.json"
        return research_projection(journal, root / "journals")

    @staticmethod
    def _local_research_origins(child, terminal_episode):
        root = supervisor_campaign_root(child.workspace) / "journals" / "episodes"
        result = {"directions": {}, "experiments": {}}
        for path in sorted(root.glob("e*/journal.json")):
            try:
                episode = int(path.parent.name[1:])
            except ValueError:
                continue
            if episode > terminal_episode:
                continue
            journal = load_journal(path)
            for event in journal["direction_events"]:
                if isinstance(event, dict) and event.get("action") == "propose":
                    result["directions"].setdefault(event.get("direction_id"), episode)
            for experiment in journal["experiments"]:
                if isinstance(experiment, dict):
                    result["experiments"].setdefault(experiment.get("experiment_id"), episode)
        return result

    @staticmethod
    def _merge_named(values, prior_origins, local_episodes, jobs, label):
        merged, origins = {}, copy.deepcopy(prior_origins)
        all_ids = sorted({identifier for _, rows in values.items() for identifier in rows})
        episodes = {job["trajectory"]: job["episode"] for job in jobs}
        for identifier in all_ids:
            candidates = {number: rows[identifier] for number, rows in values.items() if identifier in rows}
            origin = origins.get(identifier, {})
            owner = origin.get("trajectory")
            if owner in candidates:
                selected = candidates[owner]
            else:
                serialized = {json.dumps(item, sort_keys=True, allow_nan=False) for item in candidates.values()}
                if len(serialized) != 1:
                    raise RuntimeError(f"Conflicting shared {label} identity: {identifier}")
                owner = min(candidates)
                selected = candidates[owner]
                origins[identifier] = {
                    "epoch": jobs[0]["epoch"],
                    "trajectory": owner,
                    "episode": local_episodes.get(owner, {}).get(identifier, episodes[owner]),
                }
            merged[identifier] = copy.deepcopy(selected)
        return merged, origins

    def epoch_sealed(self, epoch):
        return (self.root / "epochs" / f"e{epoch:08d}-shared.json").is_file()

    def seal_epoch(self, epoch, jobs):
        return self._seal_epoch(epoch, jobs, terminal_only=False)

    def seal_publication(self, publication, jobs):
        """Seal an asynchronous snapshot without reading newer live Episodes."""
        return self._seal_epoch(publication, jobs, terminal_only=True)

    @staticmethod
    def _terminal_record_ids(child, terminal_episode, projection):
        record_ids = set()
        for experiment in projection["experiments"]:
            values = experiment.get("gateway_record_ids")
            if isinstance(values, list):
                record_ids.update(item for item in values if isinstance(item, str))
        journal = load_journal(
            supervisor_campaign_root(child.workspace)
            / "journals"
            / "episodes"
            / f"e{terminal_episode:08d}"
            / "journal.json"
        )
        submitted_commits = {
            submission.get("kernel_commit_id")
            for submission in journal.get("kernel_submissions", [])
            if isinstance(submission, dict)
            and submission.get("accepted") is True
            and isinstance(submission.get("kernel_commit_id"), str)
        }
        if submitted_commits:
            measurements = MeasurementStore(
                supervisor_campaign_root(child.workspace) / "measurements"
            )
            for candidate_record_id in measurements.record_ids():
                try:
                    record = measurements.read(candidate_record_id)
                    kernel_id = record.get("kernels", {}).get("candidate", {}).get(
                        "kernel_id"
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if kernel_id in submitted_commits:
                    record_ids.add(candidate_record_id)
        outcome = journal.get("outcome")
        values = outcome.get("gateway_record_ids") if isinstance(outcome, dict) else None
        if isinstance(values, list):
            record_ids.update(item for item in values if isinstance(item, str))
        attempt_path = (
            child.workspace
            / ".atrex_long_horizon"
            / "episodes"
            / f"e{terminal_episode:04d}"
            / "attempt.json"
        )
        try:
            attempt = read_state(attempt_path)
        except FileNotFoundError:
            attempt = {}
        verification = attempt.get("verification")
        record_id = (
            verification.get("gateway_record_id")
            if isinstance(verification, dict)
            else None
        )
        if isinstance(record_id, str) and record_id:
            record_ids.add(record_id)
        return sorted(record_ids)

    @staticmethod
    def _submitted_winner(child, terminal_episode, trajectory):
        """Derive one lineage winner only from accepted submit-kernel commits."""
        runtime_root = supervisor_campaign_root(child.workspace)
        submitted = accepted_submission_kernel_ids(
            runtime_root / "journals",
            through_episode=terminal_episode,
        )
        if not submitted:
            return None
        frontier = kernel_pareto_frontier(
            MeasurementStore(runtime_root / "measurements"),
            kernel_ids=submitted,
        )
        winner = frontier.get("best_full_evaluation")
        if not isinstance(winner, dict):
            return None
        return dict(winner, trajectory=trajectory, lineage=f"T{trajectory}")

    def _seal_epoch(self, epoch, jobs, *, terminal_only):
        """Publish cumulative, read-only research and measurement evidence."""
        if not jobs or any(job.get("status") != "completed" for job in jobs):
            raise RuntimeError("Cannot seal shared evidence before every Epoch Episode is terminal")
        terminal = {}
        for job in jobs:
            terminal[job["trajectory"]] = max(
                terminal.get(job["trajectory"], 0), job["episode"]
            )
        job_rows = [
            {"epoch": epoch, "trajectory": trajectory, "episode": episode}
            for trajectory, episode in sorted(terminal.items())
        ]
        projections = {
            number: self._research(self.children[number], terminal[number])
            for number in sorted(terminal)
        }
        local_origins = {
            number: self._local_research_origins(self.children[number], terminal[number])
            for number in sorted(terminal)
        }
        accepted_submissions_by_lineage = {}
        for number in sorted(terminal):
            commit_ids = accepted_submission_kernel_ids(
                supervisor_campaign_root(self.children[number].workspace) / "journals",
                through_episode=terminal[number],
            )
            if commit_ids:
                accepted_submissions_by_lineage[f"T{number}"] = sorted(commit_ids)
        lineage_winners = {
            f"T{number}": winner
            for number in sorted(terminal)
            if isinstance(
                winner := self._submitted_winner(
                    self.children[number], terminal[number], number
                ),
                dict,
            )
        }
        global_best = (
            min(
                lineage_winners.values(),
                key=lambda item: float(item["latency_us_geomean"]),
            )
            if lineage_winners
            else None
        )
        manifest_path = self.root / "shared" / "manifest.json"
        try:
            previous = read_state(manifest_path)
        except FileNotFoundError:
            previous = {"directions": {}, "experiments": {}, "records": {}}
        direction_rows = {
            number: {
                key: item
                for key, item in projection["directions"].items()
                if _shareable_claim(item)
            }
            for number, projection in projections.items()
        }
        experiment_rows = {
            number: {
                item["experiment_id"]: item
                for item in projection["experiments"]
                if _shareable_claim(item)
            }
            for number, projection in projections.items()
        }
        directions, direction_origins = self._merge_named(
            direction_rows,
            previous.get("directions", {}),
            {number: value["directions"] for number, value in local_origins.items()},
            job_rows,
            "Direction",
        )
        for direction_id, direction in directions.items():
            trajectory = direction_origins.get(direction_id, {}).get("trajectory")
            if type(trajectory) is int and trajectory > 0:
                direction["proposed_by"] = f"T{trajectory}"
        experiments, experiment_origins = self._merge_named(
            experiment_rows,
            previous.get("experiments", {}),
            {number: value["experiments"] for number, value in local_origins.items()},
            job_rows,
            "Experiment",
        )
        for experiment_id, experiment in experiments.items():
            trajectory = experiment_origins.get(experiment_id, {}).get("trajectory")
            if type(trajectory) is int and trajectory > 0:
                experiment["recorded_by"] = f"T{trajectory}"
        for experiment_id, experiment in experiments.items():
            if not experiment.get("references"):
                continue
            experiment_origin = experiment_origins.get(experiment_id, {})
            owner = experiment_origin.get("trajectory")
            if owner not in self.children:
                raise RuntimeError(
                    f"Shared Experiment {experiment_id} has no archive owner"
                )
            source_root = (
                supervisor_campaign_root(self.children[owner].workspace) / "journals"
            )
            for child in self.children.values():
                destination_root = supervisor_campaign_root(child.workspace) / "journals"
                if destination_root.resolve() == source_root.resolve():
                    continue
                copy_experiment_reference_archive(
                    source_root,
                    destination_root,
                    experiment_id,
                    int(experiment.get("reference_revision", 1)),
                )

        research_stores = {
            number: ResearchStore(
                supervisor_campaign_root(self.children[number].workspace) / "research"
            )
            for number in sorted(terminal)
        }
        research_states = {
            number: (
                store.read_through_episode(terminal[number])
                if terminal_only
                else store.read()
            )
            for number, store in research_stores.items()
        }
        research_queue = _aggregate_direction_queue(
            research_states,
            directions,
            inherited_queue=previous.get("research_queue", []),
        )

        central = MeasurementStore(self.root / "shared" / "measurements")
        record_origins = copy.deepcopy(previous.get("records", {}))
        new_record_ids = []
        for number in sorted(terminal):
            child = self.children[number]
            source = MeasurementStore(supervisor_campaign_root(child.workspace) / "measurements")
            record_ids = (
                self._terminal_record_ids(child, terminal[number], projections[number])
                if terminal_only
                else source.record_ids()
            )
            winner = lineage_winners.get(f"T{number}")
            if isinstance(winner, dict):
                record_ids = sorted(
                    set(record_ids) | {winner["gateway_record_id"]}
                )
            for record_id in record_ids:
                if record_id in record_origins:
                    continue
                central.import_record(source, record_id)
                record_origins[record_id] = {"epoch": epoch, "trajectory": number}
                new_record_ids.append(record_id)

        for experiment_id, experiment in experiments.items():
            record_ids = experiment.get("gateway_record_ids")
            if not isinstance(record_ids, list):
                raise RuntimeError(
                    f"Shared Experiment {experiment_id} has invalid Gateway Record references"
                )
            for record_id in record_ids:
                try:
                    central.read(record_id)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise RuntimeError(
                        f"Shared Experiment {experiment_id} refers to unavailable evidence {record_id}"
                    ) from error

        research_value = {
            "directions": directions,
            "experiments": [experiments[key] for key in sorted(experiments)],
            "research_queue": research_queue,
            "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
            "lineage_winners": lineage_winners,
            "global_best": global_best,
        }
        revision = evidence_digest(research_value)
        for number, child in sorted(self.children.items()):
            destination_root = supervisor_campaign_root(child.workspace)
            destination = MeasurementStore(destination_root / "measurements")
            for record_id in new_record_ids:
                destination.import_record(central, record_id)
            visible_directions = {
                key: dict(
                    value,
                    **(
                        {"owner": f"T{direction_origins[key]['trajectory']}"}
                        if value.get("status") == "in_progress"
                        else {}
                    ),
                )
                for key, value in directions.items()
                if direction_origins[key]["trajectory"] != number
            }
            visible_experiments = [
                experiments[key] for key in sorted(experiments)
                if experiment_origins[key]["trajectory"] != number
            ]
            shared = {
                "schema_version": 1,
                "revision": revision,
                "through_epoch": epoch,
                "directions": visible_directions,
                "experiments": visible_experiments,
                "research_queue": research_queue,
                "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
                "lineage_winners": lineage_winners,
                "global_best": global_best,
            }
            private_write(shared_research_path(destination_root / "journals"), shared)

        manifest = {
            "schema_version": 1,
            "through_epoch": epoch,
            "revision": revision,
            "directions": direction_origins,
            "experiments": experiment_origins,
            "records": record_origins,
            "research_queue": research_queue,
            "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
            "lineage_winners": lineage_winners,
            "global_best": global_best,
        }
        private_write(manifest_path, manifest)
        receipt = {
            "schema_version": 1,
            "epoch": epoch,
            "revision": revision,
            "direction_ids": sorted(directions),
            "experiment_ids": sorted(experiments),
            "gateway_record_ids": central.record_ids(),
            "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
            "lineage_winners": lineage_winners,
            "global_best": global_best,
        }
        receipt_path = self.root / "epochs" / f"e{epoch:08d}-shared.json"
        try:
            existing = read_state(receipt_path)
        except FileNotFoundError:
            private_write(receipt_path, receipt)
        else:
            if existing != receipt:
                raise RuntimeError("Epoch shared-evidence receipt changed during recovery")

        for number, episode in sorted(terminal.items()):
            agent_workspace = (
                supervisor_campaign_root(self.children[number].workspace)
                / "agent-workspace"
            )
            if not (agent_workspace / ".git").is_dir():
                # Pre-v2/migrated Episodes have no Agent-private repository.
                # Their immutable Kernel/Measurement closure remains shared;
                # start filtered history with the next v2 Episode.
                private_write(
                    self.root / "shared" / "git" / f"T{number}" / f"e{epoch:08d}.json",
                    {
                        "schema_version": 1,
                        "trajectory": number,
                        "epoch": epoch,
                        "status": "unavailable",
                        "reason": "legacy_episode_without_agent_git",
                    },
                )
                continue
            export_epoch(
                agent_workspace,
                self.root / "shared" / "git" / f"T{number}" / f"e{epoch:08d}.bundle",
                trajectory=number,
                epoch=epoch,
                lineage_winner_commit_id=(
                    lineage_winners.get(f"T{number}", {}).get("kernel_commit_id")
                ),
                audit_path=self.root / "shared" / "git" / f"T{number}" / f"e{epoch:08d}.json",
            )

    def initialize(self, bootstrap, trajectories):
        baseline = read_baseline_record(self.campaign.workspace)
        if not isinstance(baseline, dict):
            raise RuntimeError("Pooled search requires private V0 baseline evidence")
        canonical = self._canonical_agent_root(bootstrap)
        if bootstrap.get("kernel_id") != canonical["agent_v0_commit_id"]:
            raise RuntimeError(
                "Pooled search checkpoint has a different canonical Agent V0; "
                "start a new Campaign rather than rewriting Agent Git history"
            )
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
            child._account_lock = Lock()
            child.workspace_suffix = f"trajectory-{number:04d}"
            child.target_util = 0.0  # Search budget, not a local utilization stop, owns admission.
            child.tokens_spent = 0
            child.search_coordinate = None
            child.search_cancel = self.cancel_event
            # Trajectory repositories contain source/version Git history only.
            # Copy V0 evaluator facts into each trajectory's private Supervisor
            # scope so scoring and validation never depend on a tracked V0 file.
            write_baseline_record(workspace, baseline)
            self._seed_trajectory_agent_root(child, canonical)
            self.children[number] = child

    def _import_peer_history(self, child, *, publication, episode, trajectory):
        """Import peer bundles into Runtime-owned refs without touching Agent HEAD."""
        if publication <= 0:
            return
        exports = []
        for peer in sorted(self.children):
            if peer == trajectory:
                continue
            root = self.root / "shared" / "git" / f"T{peer}"
            export = root / f"e{publication:08d}.bundle"
            if export.is_file():
                exports.append(export)
                continue
            receipt_path = root / f"e{publication:08d}.json"
            try:
                receipt = read_state(receipt_path)
            except FileNotFoundError:
                # An asynchronous cumulative publication may predate this
                # peer's first completed segment.
                continue
            if receipt.get("status") != "unavailable":
                raise RuntimeError(f"Peer Git export is missing: {export}")
        runtime_root = supervisor_campaign_root(child.workspace)
        audit_path = (
            runtime_root
            / "adaptive-state"
            / "git-imports"
            / f"episode-{episode:08d}-publication-{publication:08d}.json"
        )
        if exports:
            import_exports(
                runtime_root / "adaptive-state" / "agent-git.bundle",
                exports,
                audit_path=audit_path,
            )
        else:
            private_write(
                audit_path,
                {
                    "schema_version": 1,
                    "imports": [],
                    "status": "unavailable",
                    "reason": "no_peer_history_in_selected_publication",
                },
            )

    def prepare_epoch(self, epoch):
        for number, child in self.children.items():
            store = self._state(child)
            expected = (epoch - 1) * self.campaign.module_options.search.episodes_per_trajectory
            if store["episodes"] != expected or (child.workspace / ".atrex_long_horizon/active_episode.json").exists():
                raise RuntimeError(
                    "Cannot publish peers while a trajectory has an unfinished or "
                    "unexpected Episode"
                )
            self._import_peer_history(
                child,
                publication=epoch - 1,
                episode=expected + 1,
                trajectory=number,
            )
            child.start_runtime()

    def prepare_trajectory(
        self,
        epoch,
        trajectory,
        episode,
        shared_publication,
    ):
        """Prepare one asynchronous island without waiting for its peers."""
        child = self.children[trajectory]
        store = self._state(child)
        active = child.workspace / ".atrex_long_horizon/active_episode.json"
        if store["episodes"] == episode:
            # The Episode reached its durable outcome before the coordinator
            # recorded the result. run_episode() will read the archive only.
            return
        if store["episodes"] != episode - 1:
            raise RuntimeError(
                "Cannot prepare asynchronous trajectory with unexpected Episode progress"
            )
        if active.exists():
            # Preserve the exact WIP checkout and shared snapshot selected when
            # this Episode was first admitted. Normal Episode recovery owns it.
            child.start_runtime()
            return
        if type(shared_publication) is not int or shared_publication < 0:
            raise RuntimeError("Asynchronous shared publication is invalid")
        self._import_peer_history(
            child,
            publication=shared_publication,
            episode=episode,
            trajectory=trajectory,
        )
        child.start_runtime()

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
        child.search_coordinate = {
            "epoch": epoch,
            "trajectory": trajectory,
            "episode": episode,
            "cross_trajectory_sharing": True,
            "synchronization": self.campaign.module_options.search.synchronization,
            "role_specialization": (
                self.campaign.module_options.search.role_specialization
            ),
        }
        # An archived terminal Episode is not run again after coordinator restart.
        if current < episode or (child.workspace / ".atrex_long_horizon/active_episode.json").exists():
            child.start_runtime()
            from aka.light.workflow.session import LongSessionRunner
            runner = LongSessionRunner(agent=child.agent, agent_cli=child.agent_cli,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0)
            runner.cancel_event = self.cancel_event
            LongHorizonCampaign(base_campaign=child, max_episodes=episode,
                                verifier=None,
                                session_runner=runner).run()
        if self._state(child)["episodes"] != episode:
            raise RuntimeError("Episode did not reach a durable terminal outcome")
        attempt = read_state(child.workspace / f".atrex_long_horizon/episodes/e{episode:04d}/attempt.json")
        if attempt.get("episode") != episode:
            raise RuntimeError("Episode archive identity mismatch")
        agent_head = attempt.get("agent_head")
        if not isinstance(agent_head, str) or not agent_head:
            raise RuntimeError("Completed audit segment has no Agent-owned HEAD commit identity")
        runtime_root = supervisor_campaign_root(child.workspace)
        submitted = accepted_submission_kernel_ids(runtime_root / "journals")
        frontier = kernel_pareto_frontier(
            MeasurementStore(runtime_root / "measurements"),
            kernel_ids=submitted,
        )
        best_submitted = frontier.get("best_full_evaluation")
        return {"epoch": epoch, "trajectory": trajectory, "episode": episode,
                "status": attempt["status"],
                "agent_head_commit_id": agent_head,
                "best_submitted_evaluation": best_submitted,
                "archive": str(child.workspace / f".atrex_long_horizon/episodes/e{episode:04d}"),
                "tokens": attempt.get("tokens", 0),
                "session_id": attempt.get("session_id")}

    def tokens(self):
        return sum(self._state(child)["tokens"] for child in self.children.values())

    def interrupt(self):
        # Stop further HTTP work. Running Sessions retain their normal bounded
        # process lifecycle and unfinished Episodes retain their recovery points.
        self.cancel_event.set()
        for child in self.children.values():
            child.close_runtime()

    def close(self):
        self.interrupt()
