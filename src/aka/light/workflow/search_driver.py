"""Reuse the existing serial Episode engine in isolated trajectory repositories.

Each trajectory keeps its own writable Journal, measurement task/retry scope and
adaptive state.  Completed Epochs publish immutable research, measurements and
their Kernel source closure alongside the accepted Kernel.  These are private
controller repositories, not Agent-visible Git workspaces.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Lock

import aka.light.workflow.context as main_adapter
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workspace.git import git_blob, git_head
from aka.legacy.task.persistence import durable_write_json, durable_replace
from aka.light.workspace.versions import version_record_path
from aka.light.workspace.baseline import read_baseline_record, write_baseline_record
from .search import latency, read_state
from .search_git import commit_files, git
from aka.light.workspace.sources import from_git, unpack, snapshot as source_snapshot
from aka.light.experiment.store import MeasurementStore, digest as evidence_digest, private_write
from aka.light.runtime.paths import supervisor_campaign_root
from aka.light.workflow.journal_service import (
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
            raise RuntimeError("Pooled search requires a verified Bootstrap/Framework baseline")
        seed = self._kernel(
            self.campaign.workspace,
            git_head(self.campaign.workspace),
            record,
            0,
            0,
        )
        canonical = self._canonical_agent_root(seed)
        seed["kernel_id"] = canonical["framework_baseline_commit_id"]
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
        """Seal one Agent Git root shared by every pooled trajectory.

        The root Framework Baseline is executed once before search fan-out.  Its
        persistent Agent repository already contains the Supervisor-created V0
        commit and the accepted Framework Baseline descendant.  Export that
        exact ancestry once instead of letting each trajectory independently
        create an initial commit whose timestamp would change its identity.
        """
        source = self.source(bootstrap)
        source_sha256 = hashlib.sha256(source).hexdigest()
        bundle = self._canonical_agent_bundle
        receipt_path = self._canonical_agent_receipt
        existing_receipt = None
        if receipt_path.is_file():
            existing_receipt = read_state(receipt_path)
        if bundle.is_file() and receipt_path.is_file():
            receipt = existing_receipt
            if not isinstance(receipt, dict):
                raise RuntimeError("Canonical Agent root receipt is invalid")
            if receipt.get("source_sha256") != source_sha256:
                raise RuntimeError(
                    "Canonical Agent root source differs from the frozen Framework Baseline"
                )
            baseline_commit = receipt.get("framework_baseline_commit_id")
            v0_commit = receipt.get("agent_v0_commit_id")
            if not isinstance(baseline_commit, str) or not isinstance(v0_commit, str):
                raise RuntimeError("Canonical Agent root receipt is incomplete")
            if not self._bundle_contains_commit(bundle, baseline_commit):
                raise RuntimeError("Canonical Agent root bundle lost its Framework Baseline")
            if not self._bundle_contains_commit(bundle, v0_commit):
                raise RuntimeError("Canonical Agent root bundle lost its V0 commit")
            return receipt

        runtime_root = supervisor_campaign_root(self.campaign.workspace)
        root_agent_workspace = runtime_root / "agent-workspace"
        root_agent_bundle = runtime_root / "adaptive-state" / "agent-git.bundle"
        source_repository: Path | None = None
        baseline_commit: str | None = None

        if (root_agent_workspace / ".git").is_dir():
            root_git = AgentGit(root_agent_workspace, root_agent_bundle)
            baseline_commit = root_git.find_kernel_commit(source)
            source_repository = root_agent_workspace

        recovery_directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            if baseline_commit is None and root_agent_bundle.is_file():
                recovery_directory = tempfile.TemporaryDirectory(
                    prefix="aka-framework-agent-recovery-"
                )
                recovered = Path(recovery_directory.name) / "repo"
                recovered.mkdir()
                git(recovered, "init", "-b", "work", "--quiet")
                git(
                    recovered,
                    "-c",
                    "protocol.file.allow=always",
                    "fetch",
                    "--quiet",
                    str(root_agent_bundle),
                    "+refs/heads/*:refs/atrex/framework-baseline-heads/*",
                    "+refs/tags/*:refs/tags/*",
                    "+refs/atrex/*:refs/atrex/*",
                )
                baseline_commit = AgentGit(
                    recovered, recovered / ".git" / "unused.bundle"
                ).find_kernel_commit(source)
                source_repository = recovered

            # Legacy/pinned baselines may predate an Agent repository.  Create
            # the canonical root exactly once here; trajectories still copy the
            # resulting commit rather than independently manufacturing one.
            if baseline_commit is None or source_repository is None:
                recovery_directory = tempfile.TemporaryDirectory(
                    prefix="aka-canonical-agent-v0-"
                )
                source_repository = Path(recovery_directory.name) / "repo"
                from aka.light.workspace.sources import install

                install(source_repository, source)
                repository = AgentGit(source_repository, bundle)
                repository.prepare()
                baseline_commit = repository.persist()
                if baseline_commit is None:
                    raise RuntimeError("Supervisor could not create the canonical Agent V0")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="aka-canonical-agent-root-"
                ) as temporary:
                    canonical_workspace = Path(temporary) / "repo"
                    canonical_workspace.mkdir()
                    git(canonical_workspace, "init", "-b", "work", "--quiet")
                    git(
                        canonical_workspace,
                        "-c",
                        "protocol.file.allow=always",
                        "fetch",
                        "--quiet",
                        str(source_repository),
                        "+refs/heads/*:refs/atrex/framework-baseline-heads/*",
                        "+refs/tags/*:refs/tags/*",
                        "+refs/atrex/*:refs/atrex/*",
                        "+refs/notes/*:refs/notes/*",
                        f"+{baseline_commit}:refs/atrex/framework-baseline",
                    )
                    git(
                        canonical_workspace,
                        "checkout",
                        "--quiet",
                        "-B",
                        "work",
                        "refs/atrex/framework-baseline",
                    )
                    canonical_git = AgentGit(canonical_workspace, bundle)
                    persisted = canonical_git.persist()
                    if persisted != baseline_commit:
                        raise RuntimeError(
                            "Canonical Agent root persistence changed the Framework Baseline commit"
                        )

            roots = git(
                source_repository,
                "rev-list",
                "--max-parents=0",
                baseline_commit,
            ).decode().splitlines()
            if len(roots) != 1:
                raise RuntimeError("Canonical Agent history must have exactly one V0 root")
            v0_commit = roots[0]
            if not self._bundle_contains_commit(bundle, baseline_commit):
                raise RuntimeError("Canonical Agent bundle does not contain its baseline commit")
            if not self._bundle_contains_commit(bundle, v0_commit):
                raise RuntimeError("Canonical Agent bundle does not contain its V0 root")
            receipt = {
                "schema_version": 1,
                "source_sha256": source_sha256,
                "agent_v0_commit_id": v0_commit,
                "framework_baseline_commit_id": baseline_commit,
                "framework_baseline_controller_commit_id": bootstrap["commit"],
            }
            if existing_receipt is not None and existing_receipt != receipt:
                raise RuntimeError(
                    "Recovered canonical Agent root differs from its durable receipt"
                )
            durable_write_json(receipt_path, receipt)
            return receipt
        finally:
            if recovery_directory is not None:
                recovery_directory.cleanup()

    def _seed_trajectory_agent_root(self, child, canonical: dict) -> None:
        """Install the immutable canonical bundle before a child Runtime starts."""
        target_root = supervisor_campaign_root(child.workspace)
        target = target_root / "adaptive-state" / "agent-git.bundle"
        baseline_commit = canonical["framework_baseline_commit_id"]
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

    def _framework_baseline_research_seed(self) -> dict:
        """Return Episode-zero research as immutable pooled-search evidence."""
        source_root = supervisor_campaign_root(self.campaign.workspace)
        state = ResearchStore(source_root / "research").read_through_episode(0)
        directions = {}
        for direction_id, raw in state.get("directions", {}).items():
            item = copy.deepcopy(raw)
            item.pop("origin", None)
            item["access"] = "peer_read_only"
            item["proposed_by"] = "Framework Baseline"
            if item.get("status") == "in_progress":
                item["owner"] = "Framework Baseline"
            else:
                item.pop("owner", None)
            directions[direction_id] = item

        experiments = []
        for experiment_id in sorted(state.get("experiments", {})):
            item = copy.deepcopy(state["experiments"][experiment_id])
            item.pop("origin", None)
            item["access"] = "peer_read_only"
            item["recorded_by"] = "Framework Baseline"
            experiments.append(item)

        queue = []
        for direction_id in (*state.get("queue", []), *state.get("shared_queue", [])):
            direction = directions.get(direction_id)
            if (
                direction_id not in queue
                and isinstance(direction, dict)
                and direction.get("status") in _OPEN_DIRECTION_STATUSES
            ):
                queue.append(direction_id)

        record_ids = set()
        for item in (*directions.values(), *experiments):
            values = item.get("gateway_record_ids", [])
            if isinstance(values, list):
                record_ids.update(value for value in values if isinstance(value, str))

        public = {
            "schema_version": 1,
            "through_epoch": 0,
            "directions": directions,
            "experiments": experiments,
            "research_queue": queue,
        }
        public["revision"] = evidence_digest({
            "directions": directions,
            "experiments": experiments,
            "research_queue": queue,
        })
        return {
            "public": public,
            "record_ids": sorted(record_ids),
            "source_root": source_root,
        }

    @staticmethod
    def _merge_framework_baseline_research(existing: dict, seed: dict) -> dict:
        """Merge an Episode-zero seed without replacing later shared evidence."""
        directions = copy.deepcopy(existing.get("directions", {}))
        for direction_id, item in seed["directions"].items():
            previous = directions.get(direction_id)
            if previous is not None and previous != item:
                raise RuntimeError(
                    f"Framework Baseline Direction identity changed: {direction_id}"
                )
            directions.setdefault(direction_id, copy.deepcopy(item))

        experiments_by_id = {
            item["experiment_id"]: copy.deepcopy(item)
            for item in existing.get("experiments", [])
        }
        for item in seed["experiments"]:
            experiment_id = item["experiment_id"]
            previous = experiments_by_id.get(experiment_id)
            if previous is not None and previous != item:
                raise RuntimeError(
                    f"Framework Baseline Experiment identity changed: {experiment_id}"
                )
            experiments_by_id.setdefault(experiment_id, copy.deepcopy(item))

        queue = []
        for direction_id in (
            *existing.get("research_queue", []),
            *seed.get("research_queue", []),
        ):
            if direction_id in directions and direction_id not in queue:
                queue.append(direction_id)
        result = {
            "schema_version": 1,
            "through_epoch": max(existing.get("through_epoch", 0), 0),
            "directions": directions,
            "experiments": [experiments_by_id[key] for key in sorted(experiments_by_id)],
            "research_queue": queue,
        }
        result["revision"] = evidence_digest({
            "directions": result["directions"],
            "experiments": result["experiments"],
            "research_queue": result["research_queue"],
        })
        return result

    def _seed_framework_baseline_search_state(self, seed: dict) -> None:
        """Pin common Baseline evidence and provenance before trajectory fan-out."""
        public = seed["public"]
        if not public["directions"] and not public["experiments"]:
            return
        source_root = seed["source_root"]
        source_measurements = MeasurementStore(source_root / "measurements")
        central = MeasurementStore(self.root / "shared" / "measurements")
        for record_id in seed["record_ids"]:
            central.import_record(source_measurements, record_id)

        manifest_path = self.root / "shared" / "manifest.json"
        try:
            manifest = read_state(manifest_path)
        except FileNotFoundError:
            manifest = {"directions": {}, "experiments": {}, "records": {}}
        origin = {
            "source": "framework_baseline",
            "epoch": 0,
            "trajectory": 0,
            "episode": 0,
        }
        def pin_identity(kind, identifier):
            identities = manifest.setdefault(kind, {})
            previous = identities.get(identifier)
            if previous is not None and previous != origin:
                raise RuntimeError(
                    f"Framework Baseline {kind[:-1]} identity conflicts with "
                    f"pooled search state: {identifier}"
                )
            identities.setdefault(identifier, copy.deepcopy(origin))

        for direction_id in public["directions"]:
            pin_identity("directions", direction_id)
        for experiment in public["experiments"]:
            pin_identity("experiments", experiment["experiment_id"])
        for record_id in seed["record_ids"]:
            pin_identity("records", record_id)
        inherited = manifest.get("research_queue", [])
        manifest["research_queue"] = list(dict.fromkeys(
            [*inherited, *public["research_queue"]]
        ))
        private_write(manifest_path, manifest)

    def _seed_trajectory_framework_baseline_research(self, child, seed: dict) -> None:
        """Install common Baseline claims, attachments and Records in one child."""
        public = seed["public"]
        if not public["directions"] and not public["experiments"]:
            return
        source_root = seed["source_root"]
        destination_root = supervisor_campaign_root(child.workspace)
        source_measurements = MeasurementStore(source_root / "measurements")
        destination_measurements = MeasurementStore(destination_root / "measurements")
        for record_id in seed["record_ids"]:
            destination_measurements.import_record(source_measurements, record_id)
        for experiment in public["experiments"]:
            if not experiment.get("references"):
                continue
            copy_experiment_reference_archive(
                source_root / "journals",
                destination_root / "journals",
                experiment["experiment_id"],
                int(experiment.get("reference_revision", 1)),
            )

        path = shared_research_path(destination_root / "journals")
        try:
            existing = read_state(path)
        except FileNotFoundError:
            existing = {
                "schema_version": 1,
                "revision": "",
                "through_epoch": 0,
                "directions": {},
                "experiments": [],
                "research_queue": [],
            }
        private_write(
            path,
            self._merge_framework_baseline_research(existing, public),
        )

    def _kernel(
        self, workspace, commit, record, trajectory, episode, kernel_id=None
    ):
        source = from_git(workspace, commit)
        value = {"workspace": str(workspace), "commit": commit,
                 "kernel_id": kernel_id or commit,
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
            if origin.get("source") == "framework_baseline":
                serialized = {
                    json.dumps(item, sort_keys=True, allow_nan=False)
                    for item in candidates.values()
                }
                if len(serialized) != 1:
                    raise RuntimeError(
                        f"Conflicting inherited Framework Baseline {label}: {identifier}"
                    )
                selected = candidates[min(candidates)]
            elif owner in candidates:
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
            if experiment_origin.get("source") == "framework_baseline":
                # Baseline attachments were copied to every child before fan-out.
                continue
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
                        if (
                            value.get("status") == "in_progress"
                            and direction_origins[key].get("source")
                            != "framework_baseline"
                        )
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
        }
        private_write(manifest_path, manifest)
        receipt = {
            "schema_version": 1,
            "epoch": epoch,
            "revision": revision,
            "direction_ids": sorted(directions),
            "experiment_ids": sorted(experiments),
            "gateway_record_ids": central.record_ids(),
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
                audit_path=self.root / "shared" / "git" / f"T{number}" / f"e{epoch:08d}.json",
            )

    def initialize(self, bootstrap, trajectories):
        baseline = read_baseline_record(self.campaign.workspace)
        if not isinstance(baseline, dict):
            raise RuntimeError("Pooled search requires private V0 baseline evidence")
        canonical = self._canonical_agent_root(bootstrap)
        baseline_research = self._framework_baseline_research_seed()
        self._seed_framework_baseline_search_state(baseline_research)
        if bootstrap.get("kernel_id") != canonical["framework_baseline_commit_id"]:
            raise RuntimeError(
                "Pooled search checkpoint predates the canonical shared Agent V0; "
                "start a new pooled Campaign rather than rewriting trajectory Git history"
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
            self._seed_trajectory_framework_baseline_research(
                child, baseline_research
            )
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
            record.update(version=f"v{version}", git_commit_hash=kernel["kernel_id"],
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

    def prepare_epoch(self, epoch, kernels):
        search_state = read_state(self.root / "state.json")
        official = search_state.get("global_best")
        if not isinstance(official, dict):
            raise RuntimeError("Search checkpoint has no global best Kernel")
        for number, child in self.children.items():
            store = self._state(child)
            expected = (epoch - 1) * self.campaign.module_options.search.episodes_per_trajectory
            if store["episodes"] != expected or (child.workspace / ".atrex_long_horizon/active_episode.json").exists():
                raise RuntimeError("Cannot broadcast while a trajectory has an unfinished or unexpected Episode")
            expected_kernel = kernels.get(str(number))
            if not isinstance(expected_kernel, dict):
                raise RuntimeError("Search checkpoint has no working Kernel for a trajectory")
            # Persistent island semantics: each trajectory keeps its own HEAD,
            # WIP and lineage best. Epoch preparation validates identity but
            # never broadcasts another trajectory's winner into this checkout.
            current_source = source_snapshot(child.workspace)
            if hashlib.sha256(current_source).hexdigest() != expected_kernel["sha256"]:
                self._publish(
                    child.workspace,
                    epoch,
                    expected_kernel,
                    f"trajectory-{number:04d}-lineage-restore",
                )
            child.start_runtime()
            kernel_identity = child._supervisor_runtime.measurements.kernel(
                self.source(expected_kernel), expected_kernel["kernel_id"]
            )
            official_identity = child._supervisor_runtime.measurements.kernel(
                self.source(official), official["kernel_id"]
            )
            child._supervisor_runtime.research.checkpoint_lineage(
                {
                    "working_kernel_id": kernel_identity["kernel_id"],
                    "lineage_best_kernel_id": kernel_identity["kernel_id"],
                    "lineage_best_latency_us": expected_kernel["latency_us_geomean"],
                    "last_correct_kernel_id": kernel_identity["kernel_id"],
                },
                origin={"epoch": epoch, "trajectory": number},
            )
            child._supervisor_runtime.research.append(
                "promotion.accepted",
                {
                    "kernel_id": official_identity["kernel_id"],
                    "latency_us_geomean": official["latency_us_geomean"],
                },
                origin={"epoch": epoch, "trajectory": number},
                dedupe_key=f"epoch-incumbent:{epoch}:{official['sha256']}",
            )
            if epoch > 1:
                exports = []
                exported_peers = []
                for peer in sorted(self.children):
                    if peer == number:
                        continue
                    root = self.root / "shared" / "git" / f"T{peer}"
                    export = root / f"e{epoch - 1:08d}.bundle"
                    if export.is_file():
                        exports.append(export)
                        exported_peers.append(peer)
                        continue
                    receipt_path = root / f"e{epoch - 1:08d}.json"
                    try:
                        receipt = read_state(receipt_path)
                    except FileNotFoundError as error:
                        raise RuntimeError(
                            f"Epoch peer Git history has neither export nor audit receipt: {export}"
                        ) from error
                    if receipt.get("status") != "unavailable":
                        raise RuntimeError(f"Epoch peer Git export is missing: {export}")
                notes = {}
                for peer, kernel in kernels.items():
                    if int(peer) not in exported_peers:
                        continue
                    ref = f"refs/atrex/shared/T{peer}/e{epoch - 1:04d}/head"
                    notes[ref] = {
                        "measurement_status": "measured",
                        "kernel_sha256": kernel["sha256"],
                        "correctness": "pass",
                        "latency_us_geomean": kernel["latency_us_geomean"],
                    }
                runtime_root = supervisor_campaign_root(child.workspace)
                audit_path = (
                    runtime_root / "adaptive-state" / "git-imports" / f"e{epoch:08d}.json"
                )
                if exports:
                    import_exports(
                        runtime_root / "adaptive-state" / "agent-git.bundle",
                        exports,
                        notes=notes,
                        audit_path=audit_path,
                    )
                else:
                    private_write(audit_path, {
                        "schema_version": 1,
                        "imports": [],
                        "status": "unavailable",
                        "reason": "all_peer_episodes_predate_agent_git",
                    })

    def prepare_trajectory(
        self,
        epoch,
        trajectory,
        episode,
        expected_kernel,
        official,
        shared_publication,
        peer_kernels,
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
        if not isinstance(expected_kernel, dict) or not isinstance(official, dict):
            raise RuntimeError("Asynchronous search checkpoint has no valid Kernel identity")
        if type(shared_publication) is not int or shared_publication < 0:
            raise RuntimeError("Asynchronous shared publication is invalid")
        if not isinstance(peer_kernels, dict):
            raise RuntimeError("Asynchronous peer Kernel snapshot is invalid")

        current_source = source_snapshot(child.workspace)
        if hashlib.sha256(current_source).hexdigest() != expected_kernel["sha256"]:
            self._publish(
                child.workspace,
                episode,
                expected_kernel,
                f"trajectory-{trajectory:04d}-lineage-restore",
            )
        child.start_runtime()
        kernel_identity = child._supervisor_runtime.measurements.kernel(
            self.source(expected_kernel), expected_kernel["kernel_id"]
        )
        official_identity = child._supervisor_runtime.measurements.kernel(
            self.source(official), official["kernel_id"]
        )
        child._supervisor_runtime.research.checkpoint_lineage(
            {
                "working_kernel_id": kernel_identity["kernel_id"],
                "lineage_best_kernel_id": kernel_identity["kernel_id"],
                "lineage_best_latency_us": expected_kernel["latency_us_geomean"],
                "last_correct_kernel_id": kernel_identity["kernel_id"],
            },
            origin={"epoch": epoch, "trajectory": trajectory, "episode": episode},
        )
        child._supervisor_runtime.research.append(
            "promotion.accepted",
            {
                "kernel_id": official_identity["kernel_id"],
                "latency_us_geomean": official["latency_us_geomean"],
            },
            origin={"epoch": epoch, "trajectory": trajectory, "episode": episode},
            dedupe_key=(
                f"async-incumbent:{trajectory}:{episode}:{official['sha256']}"
            ),
        )
        if shared_publication == 0:
            return

        exports = []
        exported_peers = []
        for peer in sorted(self.children):
            if peer == trajectory:
                continue
            root = self.root / "shared" / "git" / f"T{peer}"
            export = root / f"e{shared_publication:08d}.bundle"
            if export.is_file():
                exports.append(export)
                exported_peers.append(peer)
                continue
            receipt_path = root / f"e{shared_publication:08d}.json"
            try:
                receipt = read_state(receipt_path)
            except FileNotFoundError:
                # This peer had not completed an Episode in that cumulative
                # publication, so it legitimately has no exported history.
                continue
            if receipt.get("status") != "unavailable":
                raise RuntimeError(f"Asynchronous peer Git export is missing: {export}")
        notes = {}
        for peer, kernel in peer_kernels.items():
            if int(peer) not in exported_peers:
                continue
            ref = f"refs/atrex/shared/T{peer}/e{shared_publication:04d}/head"
            notes[ref] = {
                "measurement_status": "measured",
                "kernel_sha256": kernel["sha256"],
                "correctness": "pass",
                "latency_us_geomean": kernel["latency_us_geomean"],
            }
        runtime_root = supervisor_campaign_root(child.workspace)
        audit_path = (
            runtime_root
            / "adaptive-state"
            / "git-imports"
            / f"episode-{episode:08d}-publication-{shared_publication:08d}.json"
        )
        if exports:
            import_exports(
                runtime_root / "adaptive-state" / "agent-git.bundle",
                exports,
                notes=notes,
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
            verifier = child.acceptance_verifier(
                execute=child.measure_for_acceptance, hardware=child.sandbox_hardware,
                profile=child.sandbox_profile, url=child.sandbox_url, ssh=child.sandbox_ssh,
                ssh_init=child.sandbox_ssh_init, health_command=child.sandbox_health_command,
                timeout=child.sandbox_timeout, repeats=child.verify_repeats,
                per_run_timeout=child.evaluation_step_timeout, min_improvement_pct=child.min_improvement_pct,
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
            agent_kernel_id = attempt.get(
                "candidate_kernel_commit_id", attempt.get("candidate_kernel_id")
            )
            if isinstance(agent_kernel_id, str):
                measured_source = MeasurementStore(
                    supervisor_campaign_root(child.workspace) / "measurements"
                ).read_kernel(agent_kernel_id)
                if from_git(child.workspace, commit) != measured_source:
                    raise RuntimeError(
                        "Promoted search Kernel differs from the accepted Agent commit"
                    )
            retained = self._kernel(
                child.workspace,
                commit,
                record,
                trajectory,
                episode,
                kernel_id=agent_kernel_id,
            )
        return {"epoch": epoch, "trajectory": trajectory, "episode": episode,
                "status": attempt["status"], "retained_kernel": retained,
                "archive": str(child.workspace / f".atrex_long_horizon/episodes/e{episode:04d}"),
                "tokens": attempt.get("tokens", 0),
                "session_id": attempt.get("session_id")}

    def publish_epoch(self, epoch, kernel):
        self._publish(self.campaign.workspace, epoch, kernel, "campaign")

    def tokens(self):
        return sum(self._state(child)["tokens"] for child in self.children.values())

    def stop_reason(self, state):
        target = self.campaign.target_util
        if target > 0 and main_adapter.peak_util(state["global_best"]["record"]) >= target:
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
