"""Reuse the serial Session engine in isolated persistent Agent trajectories.

Completed Epochs publish immutable research, measurements and peer Agent Git.
Lineage/global winners are derived from accepted ``submit-kernel`` commits with
passing full Evaluate evidence and broadcast as observations. They never select
the next checkout: every trajectory keeps its own HEAD and provider Session.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Lock

import aka.workflow.light.long_horizon.main_adapter as main_adapter
from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
from aka.task.git import git_head
from aka.core.filesystem import durable_write_json, durable_replace
from .search import (
    COORDINATOR_STATE,
    lineage_state_path,
    merge_lineage_state,
    publication_import_path,
    publication_path,
    publication_source_root,
    read_state,
    update_publication,
)
from aka.task.git import git_text as git
from aka.task.source.formats import from_git
from aka.experiment.measurement_store import MeasurementStore, digest as evidence_digest, private_write
from aka.experiment.frontier import (
    kernel_full_evaluation_evidence,
    kernel_pareto_frontier,
    standard_full_evaluation,
)
from aka.workflow.light.support.paths import (
    legacy_supervisor_campaign_root,
    supervisor_campaign_base_root,
    supervisor_campaign_root,
    supervisor_measurement_root,
    supervisor_research_catalog_root,
    supervisor_submission_root,
    runtime_audit_files,
)
from aka.workflow.light.host.journal import (
    _public_direction,
    _public_experiment,
    project_campaign_research_library,
    shared_research_path,
)
from aka.workflow.light.long_horizon.submissions import accepted_submission_kernel_ids
from aka.workflow.light.search_git import (
    export_publication,
    import_exports_into_repository,
)
from aka.workflow.light.research import ResearchStore


_OPEN_DIRECTION_STATUSES = {"proposed", "in_progress", "deferred", "blocked"}


def _visible_claim_record_ids(value: object) -> set[str]:
    """Return Record IDs referenced by one claim visible to a lineage."""
    if not isinstance(value, dict):
        return set()
    record_ids = value.get("gateway_record_ids", [])
    if not isinstance(record_ids, list):
        return set()
    return {record_id for record_id in record_ids if isinstance(record_id, str)}


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

    @property
    def measurement_root(self) -> Path:
        """One canonical Measurement Store for every trajectory in this Campaign."""
        return supervisor_measurement_root(self.campaign.workspace)

    def _migrate_trajectory_private_root(self, workspace: Path) -> None:
        """Move a pre-centralization trajectory root into its Campaign lineage."""
        legacy = legacy_supervisor_campaign_root(workspace)
        destination = supervisor_campaign_root(workspace)
        if legacy.resolve() == destination.resolve() or not legacy.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists():
            raise RuntimeError(
                "Both legacy trajectory-private state and the Campaign lineage "
                f"namespace exist; reconcile them before recovery: {legacy} and "
                f"{destination}"
            )
        os.replace(legacy, destination)

    def _migrate_campaign_measurements(self) -> None:
        """Import pre-centralization immutable Records into the Campaign Store."""
        destination = MeasurementStore(self.measurement_root)
        legacy_roots = [self.root / "shared" / "measurements"]
        for child in self.children.values():
            legacy_roots.append(supervisor_campaign_root(child.workspace) / "measurements")
        seen: set[Path] = set()
        for root in legacy_roots:
            resolved = root.resolve()
            if resolved in seen or resolved == self.measurement_root.resolve():
                continue
            seen.add(resolved)
            if not (root / "identity.key").is_file():
                continue
            source = MeasurementStore(root)
            for record_id in source.record_ids():
                destination.import_record(source, record_id)

    def bootstrap(self):
        main_adapter.prepare_campaign(self.campaign)
        # Do not reinterpret a previously serial campaign as pooled search.
        from aka.workflow.light.long_horizon.store import CampaignStore

        segment_store = CampaignStore(self.campaign.workspace)
        state_file = segment_store.state_path
        recovering_pooled = (self.root / COORDINATOR_STATE).is_file()
        if (
            state_file.exists()
            and read_state(state_file).get("episodes", 0)
            and not recovering_pooled
        ):
            raise ValueError("Start pooled search in a new Campaign; serial Episode checkpoints are not migrated")
        active = segment_store.active_path
        if active.exists() and not recovering_pooled:
            raise ValueError("Cannot enable pooled search with an active serial Episode")
        from .initial_source import initial_kernel

        seed = initial_kernel(self.campaign)
        canonical = self._canonical_agent_root(seed)
        seed["kernel_id"] = canonical["agent_v0_commit_id"]
        return seed

    @property
    def _canonical_agent_bundle(self) -> Path:
        return (
            supervisor_campaign_base_root(self.campaign.workspace)
            / "bootstrap"
            / "agent-v0.bundle"
        )

    @property
    def _canonical_agent_receipt(self) -> Path:
        return (
            supervisor_campaign_base_root(self.campaign.workspace)
            / "bootstrap"
            / "agent-v0.json"
        )

    def _publication_source_root(self, trajectory: int, publication: int) -> Path:
        campaign_root = supervisor_campaign_base_root(self.campaign.workspace)
        return publication_source_root(campaign_root, publication, trajectory)

    def _publication_import_receipt(
        self,
        trajectory: int,
        publication: int,
    ) -> Path:
        campaign_root = supervisor_campaign_base_root(self.campaign.workspace)
        return publication_import_path(campaign_root, publication, trajectory)

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
        """Archive the T1 V0 repository used to initialize every trajectory."""
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
                    "Canonical Agent root source differs from the initial source"
                )
            v0_commit = receipt.get("agent_v0_commit_id")
            if not isinstance(v0_commit, str):
                raise RuntimeError("Canonical Agent root receipt is incomplete")
            if not self._bundle_contains_commit(bundle, v0_commit):
                raise RuntimeError("Canonical Agent root bundle lost its V0 commit")
            return receipt
        if bundle.exists() != receipt_path.exists():
            raise RuntimeError("Canonical Agent V0 bundle and receipt are incomplete")

        v0_commit = bootstrap["commit"]
        # Recovery may find an Agent HEAD beyond the initial source. The shared
        # root stays the archived initial commit; no checkout is performed.
        git(self.campaign.workspace, "cat-file", "-e", f"{v0_commit}^{{commit}}")
        bundle.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = bundle.with_name(f".{bundle.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            ["git", "bundle", "create", str(temporary), "--all"],
            cwd=self.campaign.workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Canonical Agent V0 bundle failed: " + result.stderr.strip()
            )
        durable_replace(temporary, bundle)
        if not self._bundle_contains_commit(bundle, v0_commit):
            raise RuntimeError("Canonical Agent bundle does not contain its V0 root")
        receipt = {
            "schema_version": 2,
            "source_sha256": source_sha256,
            "agent_v0_commit_id": v0_commit,
        }
        durable_write_json(receipt_path, receipt)
        return receipt

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
        state = ResearchStore(root / "research").read_through_episode(episode)
        return {
            "directions": {
                key: _public_direction(value)
                for key, value in state.get("directions", {}).items()
                if isinstance(key, str) and isinstance(value, dict)
            },
            "experiments": [
                _public_experiment(value)
                for value in state.get("experiments", {}).values()
                if isinstance(value, dict)
            ],
        }

    @staticmethod
    def _local_research_origins(child, terminal_episode, trajectory):
        root = supervisor_campaign_root(child.workspace) / "research"
        state = ResearchStore(root).read_through_episode(terminal_episode)
        result = {"directions": {}, "experiments": {}}
        for collection in ("directions", "experiments"):
            values = state.get(collection, {})
            if not isinstance(values, dict):
                continue
            for identifier, value in values.items():
                if not isinstance(identifier, str) or not isinstance(value, dict):
                    continue
                origin = value.get("origin")
                owner = value.get("owner_lineage")
                origin_trajectory = (
                    origin.get("trajectory") if isinstance(origin, dict) else None
                )
                episode = origin.get("episode") if isinstance(origin, dict) else None
                if (
                    owner not in {None, f"T{trajectory}"}
                    or origin_trajectory not in {None, trajectory}
                    or value.get("access") == "peer_read_only"
                    or type(episode) is not int
                ):
                    continue
                result[collection][identifier] = episode
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
        try:
            publication = read_state(publication_path(self.root, epoch))
        except FileNotFoundError:
            return False
        return (
            publication.get("synchronization") == "synchronous"
            and isinstance(publication.get("evidence"), dict)
        )

    def seal_epoch(self, epoch, jobs):
        return self._seal_epoch(epoch, jobs, terminal_only=False)

    def seal_publication(self, publication, jobs):
        """Seal an asynchronous snapshot without reading newer live Episodes."""
        return self._seal_epoch(publication, jobs, terminal_only=True)

    def _local_record_ids(self, child, trajectory: int) -> list[str]:
        """Return Records observed by one lineage without owning a private store."""
        runtime_root = supervisor_campaign_root(child.workspace)
        record_ids: set[str] = set()

        state = ResearchStore(runtime_root / "research").read()
        lineage = state.get("lineages", {}).get(f"T{trajectory}", {})
        values = lineage.get("gateway_record_ids", []) if isinstance(lineage, dict) else []
        if isinstance(values, list):
            record_ids.update(value for value in values if isinstance(value, str))

        # Runtime request receipts are the durable ownership fallback for
        # campaigns created before lineage checkpoints recorded Record IDs.
        for path in runtime_audit_files(runtime_root):
            try:
                audit = read_state(path)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                continue
            record_id = audit.get("gateway_record_id")
            if isinstance(record_id, str):
                record_ids.add(record_id)

        # Resume compatibility: import old trajectory-local immutable facts
        # once into the Campaign store. New pooled Runtime instances never
        # create or write this legacy directory.
        legacy_root = runtime_root / "measurements"
        if (
            legacy_root.resolve() != self.measurement_root.resolve()
            and (legacy_root / "identity.key").is_file()
        ):
            legacy = MeasurementStore(legacy_root)
            central = MeasurementStore(self.measurement_root)
            for record_id in legacy.record_ids():
                central.import_record(legacy, record_id)
                record_ids.add(record_id)
        return sorted(record_ids)

    def _terminal_record_ids(self, child, terminal_episode, projection):
        record_ids = set()
        for experiment in projection["experiments"]:
            values = experiment.get("gateway_record_ids")
            if isinstance(values, list):
                record_ids.update(item for item in values if isinstance(item, str))
        trajectory = next(
            (
                number
                for number, candidate in self.children.items()
                if candidate is child
            ),
            1,
        )
        submitted_commits = accepted_submission_kernel_ids(
            supervisor_submission_root(child.workspace),
            lineage=f"T{trajectory}",
            through_episode=terminal_episode,
        )
        if submitted_commits:
            measurements = MeasurementStore(self.measurement_root)
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
        from aka.workflow.light.long_horizon.store import CampaignStore

        attempt_path = (
            CampaignStore(child.workspace).episode_dir(terminal_episode)
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

    def _submitted_winner(self, child, terminal_episode, trajectory):
        """Derive one lineage winner only from accepted submit-kernel commits."""
        submitted = accepted_submission_kernel_ids(
            supervisor_submission_root(child.workspace),
            lineage=f"T{trajectory}",
            through_episode=terminal_episode,
        )
        if not submitted:
            return None
        frontier = kernel_pareto_frontier(
            MeasurementStore(self.measurement_root),
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
            number: self._local_research_origins(
                self.children[number], terminal[number], number
            )
            for number in sorted(terminal)
        }
        accepted_submissions_by_lineage = {}
        for number in sorted(terminal):
            commit_ids = accepted_submission_kernel_ids(
                supervisor_submission_root(self.children[number].workspace),
                lineage=f"T{number}",
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
        research_catalog_root = supervisor_research_catalog_root(
            self.campaign.workspace
        )
        research_catalog_through_sequence = ResearchStore(
            research_catalog_root / "events"
        ).read()["sequence"]
        synchronization = getattr(
            getattr(getattr(self.campaign, "module_options", None), "search", None),
            "synchronization",
            "synchronous",
        )
        manifest_path = (
            supervisor_research_catalog_root(self.campaign.workspace)
            / "publication.json"
        )
        try:
            previous = read_state(manifest_path)
        except FileNotFoundError:
            previous = {"directions": {}, "experiments": {}, "records": {}}
        direction_rows = {
            number: dict(projection["directions"])
            for number, projection in projections.items()
        }
        experiment_rows = {
            number: {
                item["experiment_id"]: item
                for item in projection["experiments"]
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
                direction["owner_lineage"] = f"T{trajectory}"
                direction.pop("proposed_by", None)
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
                experiment["owner_lineage"] = f"T{trajectory}"
                experiment.pop("recorded_by", None)
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

        central = MeasurementStore(self.measurement_root)
        persistent_record_reasons = {
            "passing_full_evaluate",
            "accepted_submission_full_evaluate",
            "lineage_winner_full_evaluate",
            "global_winner_full_evaluate",
        }
        # Pre-policy manifests did not record why a Record was shared and may
        # contain every local Runtime result. Do not perpetuate that visibility:
        # recompute legacy entries from the whitelist below. New manifests keep
        # a reason so later publications retain immutable qualifying evidence.
        record_origins = {
            record_id: copy.deepcopy(origin)
            for record_id, origin in previous.get("records", {}).items()
            if isinstance(origin, dict)
            and origin.get("reason") in persistent_record_reasons
        }

        def publish_record(record_id: object, number: int, reason: str) -> None:
            if not isinstance(record_id, str) or record_id in record_origins:
                return
            # The Runtime wrote this Record into the canonical Campaign store.
            # Sharing publishes only the immutable identity; each Agent receives
            # a normal-file projection from that one authoritative copy.
            central.read(record_id)
            record_origins[record_id] = {
                "epoch": epoch,
                "trajectory": number,
                "reason": reason,
            }

        # The manifest audits the union of current claim-linked evidence. Each
        # receiving lineage is filtered again below using its own visible claim
        # projection, so this union never becomes an Agent visibility grant.
        for direction_id, direction in directions.items():
            owner = direction_origins.get(direction_id, {}).get("trajectory")
            if type(owner) is not int or owner <= 0:
                continue
            for record_id in _visible_claim_record_ids(direction):
                publish_record(record_id, owner, "visible_direction_reference")
        for experiment_id, experiment in experiments.items():
            owner = experiment_origins.get(experiment_id, {}).get("trajectory")
            if type(owner) is not int or owner <= 0:
                continue
            for record_id in _visible_claim_record_ids(experiment):
                publish_record(record_id, owner, "visible_experiment_reference")

        # Also share standard full Evaluates that passed correctness and carry
        # complete per-Shape performance. Direction/Experiment references do
        # not widen visibility, and diagnostic/partial results remain local.
        for number, state in research_states.items():
            lineage = state.get("lineages", {}).get(f"T{number}", {})
            local_record_ids = (
                lineage.get("gateway_record_ids", [])
                if isinstance(lineage, dict)
                and isinstance(lineage.get("gateway_record_ids", []), list)
                else []
            )
            for record_id in local_record_ids:
                try:
                    parsed = standard_full_evaluation(central.read(record_id))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if isinstance(parsed, dict) and parsed.get("passed") is True:
                    publish_record(record_id, number, "passing_full_evaluate")

        # Every accepted submission publishes complete passing full-Evaluate
        # evidence. The selected lineage/global winner Record is included too,
        # because a commit may have several passing Evaluates and the winner can
        # be backed by a faster Record than the newest one.
        for lineage, commit_ids in accepted_submissions_by_lineage.items():
            try:
                owner = int(lineage.removeprefix("T"))
            except ValueError:
                continue
            for commit_id in commit_ids:
                evidence = kernel_full_evaluation_evidence(central, commit_id)
                if evidence.get("status") == "passed":
                    publish_record(
                        evidence.get("gateway_record_id"),
                        owner,
                        "accepted_submission_full_evaluate",
                    )
        for lineage, winner in lineage_winners.items():
            try:
                owner = int(lineage.removeprefix("T"))
            except ValueError:
                continue
            publish_record(
                winner.get("gateway_record_id"),
                owner,
                "lineage_winner_full_evaluate",
            )
        if isinstance(global_best, dict):
            owner = global_best.get("trajectory")
            if type(owner) is int and owner > 0:
                publish_record(
                    global_best.get("gateway_record_id"),
                    owner,
                    "global_winner_full_evaluate",
                )

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
            "gateway_record_ids": sorted(record_origins),
            "gateway_record_reasons": {
                record_id: record_origins[record_id]["reason"]
                for record_id in sorted(record_origins)
            },
            "research_queue": research_queue,
            "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
            "lineage_winners": lineage_winners,
            "global_best": global_best,
            "research_catalog_through_sequence": research_catalog_through_sequence,
        }
        revision = evidence_digest(research_value)
        globally_visible_record_ids = {
            record_id
            for record_id, origin in record_origins.items()
            if origin.get("reason") in persistent_record_reasons
        }
        for number, child in sorted(self.children.items()):
            destination_root = supervisor_campaign_root(child.workspace)
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
            visible_claim_record_ids = set()
            for direction in visible_directions.values():
                visible_claim_record_ids.update(
                    _visible_claim_record_ids(direction)
                )
            for experiment in visible_experiments:
                visible_claim_record_ids.update(
                    _visible_claim_record_ids(experiment)
                )
            visible_record_ids = (
                globally_visible_record_ids | visible_claim_record_ids
            )
            shared = {
                "schema_version": 1,
                "revision": revision,
                "through_epoch": epoch,
                "research_catalog_through_sequence": research_catalog_through_sequence,
                "directions": visible_directions,
                "experiments": visible_experiments,
                "gateway_record_ids": sorted(visible_record_ids),
                "gateway_record_reasons": {
                    record_id: record_origins[record_id]["reason"]
                    for record_id in sorted(visible_record_ids)
                },
                "research_queue": research_queue,
                "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
                "lineage_winners": lineage_winners,
                "global_best": global_best,
            }
            private_write(shared_research_path(destination_root / "research"), shared)
            agent_workspace = destination_root / "agent-workspace"
            if agent_workspace.is_dir():
                project_campaign_research_library(
                    catalog_root=research_catalog_root,
                    workspace=agent_workspace,
                    trajectory=number,
                    synchronization=synchronization,
                    peer_through_sequence=research_catalog_through_sequence,
                )

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
            "research_catalog_through_sequence": research_catalog_through_sequence,
        }
        private_write(manifest_path, manifest)
        receipt = {
            "revision": revision,
            "direction_ids": sorted(directions),
            "experiment_ids": sorted(experiments),
            "gateway_record_ids": sorted(record_origins),
            "accepted_submission_kernel_ids": accepted_submissions_by_lineage,
            "lineage_winners": lineage_winners,
            "global_best": global_best,
        }
        sources = []
        for number, episode in sorted(terminal.items()):
            source_root = self._publication_source_root(number, epoch)
            agent_workspace = (
                supervisor_campaign_root(self.children[number].workspace)
                / "agent-workspace"
            )
            if not (agent_workspace / ".git").is_dir():
                raise RuntimeError(
                    f"T{number} cannot publish Git history without its persistent "
                    "Agent repository"
                )
            exported = export_publication(
                agent_workspace,
                source_root / "history.bundle",
                trajectory=number,
                publication=epoch,
                lineage_winner_commit_id=(
                    lineage_winners.get(f"T{number}", {}).get("kernel_commit_id")
                ),
                audit_path=source_root / "receipt.json",
            )
            sources.append({
                "trajectory": f"T{number}",
                "through_episode": episode,
                "git_status": "available",
                "git_ref": f"atrex/T{number}",
                "agent_head_commit_id": exported["original_agent_commit"],
            })

        update_publication(
            self.root,
            epoch,
            synchronization,
            watermark={
                "trajectory_episodes": {
                    f"T{number}": episode
                    for number, episode in sorted(terminal.items())
                },
                "research_catalog_sequence": research_catalog_through_sequence,
            },
            sources=sources,
            evidence=receipt,
        )

    def initialize(self, bootstrap, trajectories):
        canonical = self._canonical_agent_root(bootstrap)
        if bootstrap.get("kernel_id") != canonical["agent_v0_commit_id"]:
            raise RuntimeError(
                "Pooled search checkpoint has a different canonical Agent V0; "
                "start a new Campaign rather than rewriting Agent Git history"
            )
        for number in range(1, trajectories + 1):
            campaign_root = supervisor_campaign_base_root(self.campaign.workspace)
            lineage_root = campaign_root / "lineages" / f"t{number:04d}"
            workspace = lineage_root / "agent-workspace"
            expected = {
                "schema_version": 1,
                "trajectory": number,
                "agent_v0_commit_id": canonical["agent_v0_commit_id"],
            }
            state_path = lineage_state_path(campaign_root, number)
            if number == 1:
                if workspace.resolve() != self.campaign.workspace.resolve():
                    raise RuntimeError("T1 Agent workspace does not match the Campaign seed")
            elif not (workspace / ".git").is_dir():
                lineage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                with tempfile.TemporaryDirectory(
                    prefix=".preparing-", dir=lineage_root
                ) as temporary:
                    staged = Path(temporary) / "agent-workspace"
                    result = subprocess.run(
                        [
                            "git", "clone", "--quiet", "--no-hardlinks",
                            str(self._canonical_agent_bundle), str(staged),
                        ],
                        capture_output=True, text=True, check=False,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            "Trajectory Agent Git initialization failed: "
                            + result.stderr.strip()
                        )
                    git(staged, "checkout", "--quiet", "-B", "work", bootstrap["commit"])
                    git(staged, "remote", "remove", "origin")
                    durable_replace(staged, workspace)
            present = subprocess.run(
                ["git", "cat-file", "-e", f"{bootstrap['commit']}^{{commit}}"],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
            if present.returncode != 0:
                raise RuntimeError("Trajectory Agent Git has a different V0 root")
            try:
                existing_state = read_state(state_path)
            except FileNotFoundError:
                existing_state = {}
            if existing_state.get("agent_v0_commit_id") not in {
                None,
                expected["agent_v0_commit_id"],
            }:
                raise RuntimeError("Trajectory identity differs from this Campaign")
            merge_lineage_state(
                state_path,
                number,
                agent_v0_commit_id=expected["agent_v0_commit_id"],
            )
            child = copy.copy(self.campaign)
            child.request = replace(self.campaign.request, workspace=workspace)
            # Every trajectory resolves the one immutable source binding from
            # campaign.json. No lineage owns or copies a detached source/ tree.
            from aka.task.source.formats.private_state import read as read_source_state

            if read_source_state(workspace) is None:
                raise RuntimeError("Pooled search has no Campaign source contract")
            child._supervisor_runtime = None
            child._production_review_cache = {}
            child._account_lock = Lock()
            child.workspace_suffix = f"trajectory-{number:04d}"
            child.target_util = 0.0  # Search budget, not a local utilization stop, owns admission.
            child.tokens_spent = 0
            child.search_coordinate = None
            child.search_cancel = self.cancel_event
            child._measurement_store_root = self.measurement_root
            # Initial source identity is Campaign-scoped; every lineage starts from the
            # same Agent commit without a Controller repository or clone.
            self.children[number] = child
        # Import immutable evidence from pre-central-store Campaigns only after
        # every legacy trajectory root has moved below the Campaign namespace.
        self._migrate_campaign_measurements()
        for number, child in self.children.items():
            self._local_record_ids(child, number)

    def _import_peer_history(self, child, *, publication, trajectory):
        """Import peer bundles into Runtime-owned refs without touching Agent HEAD."""
        if publication <= 0:
            return
        exports = []
        for peer in sorted(self.children):
            if peer == trajectory:
                continue
            root = self._publication_source_root(peer, publication)
            export = root / "history.bundle"
            if export.is_file():
                exports.append(export)
                continue
            receipt_path = root / "receipt.json"
            try:
                read_state(receipt_path)
            except FileNotFoundError:
                # An asynchronous cumulative publication may predate this
                # peer's first completed segment.
                continue
            raise RuntimeError(f"Peer Git export is missing: {export}")
        audit_path = self._publication_import_receipt(trajectory, publication)
        if exports:
            import_exports_into_repository(
                child.workspace,
                exports,
                audit_path=audit_path,
            )
        else:
            private_write(
                audit_path,
                {
                    "schema_version": 3,
                    "imports": [],
                    "commit_mappings": [],
                    "shared_refs": [],
                    "status": "unavailable",
                    "reason": "no_peer_history_in_selected_publication",
                },
            )

    def prepare_epoch(self, epoch):
        from aka.workflow.light.long_horizon.store import CampaignStore

        for number, child in self.children.items():
            store = self._state(child)
            expected = (epoch - 1) * self.campaign.module_options.search.episodes_per_trajectory
            if store["episodes"] != expected or CampaignStore(
                child.workspace
            ).active_path.exists():
                raise RuntimeError(
                    "Cannot publish peers while a trajectory has an unfinished or "
                    "unexpected Episode"
                )
            self._import_peer_history(
                child,
                publication=epoch - 1,
                trajectory=number,
            )
            self._set_search_coordinate(
                child,
                epoch=epoch,
                trajectory=number,
                episode=expected + 1,
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
        from aka.workflow.light.long_horizon.store import CampaignStore

        child = self.children[trajectory]
        self._set_search_coordinate(
            child,
            epoch=epoch,
            trajectory=trajectory,
            episode=episode,
        )
        store = self._state(child)
        active = CampaignStore(child.workspace).active_path
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
            trajectory=trajectory,
        )
        child.start_runtime()

    def _state(self, child):
        from aka.workflow.light.long_horizon.store import CampaignStore

        campaign_store = CampaignStore(child.workspace)
        path = campaign_store.state_path
        try:
            state = read_state(path)
        except FileNotFoundError:
            # The very first Episode writes its active marker/brief before its
            # first usage checkpoint. That interrupted preparation is resumable;
            # lost accounting for terminal/later Episodes is not a fresh start.
            control = campaign_store.root
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

    def _set_search_coordinate(
        self,
        child,
        *,
        epoch: int,
        trajectory: int,
        episode: int,
    ) -> None:
        """Bind pooled-search semantics before the child Runtime is opened.

        RuntimeConfig snapshots research synchronization when ``start_runtime``
        first opens the service. Setting this only in ``run_episode`` is too
        late because synchronous/asynchronous preparation opens the Runtime at
        the publication boundary. In that ordering every pooled lineage was
        silently registered as ``independent`` for the life of the process.
        """
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

    def run_episode(self, epoch, trajectory, episode):
        from aka.workflow.light.long_horizon.store import CampaignStore

        child = self.children[trajectory]
        current = self._state(child)["episodes"]
        if current not in {episode - 1, episode}:
            raise RuntimeError("Trajectory Episode sequence differs from the Epoch schedule")
        self._set_search_coordinate(
            child,
            epoch=epoch,
            trajectory=trajectory,
            episode=episode,
        )
        # An archived terminal Episode is not run again after coordinator restart.
        campaign_store = CampaignStore(child.workspace)
        if current < episode or campaign_store.active_path.exists():
            child.start_runtime()
            from aka.workflow.light.long_horizon.session import LongSessionRunner
            runner = LongSessionRunner(agent=child.agent, agent_cli=child.agent_cli, agent_model=child.agent_model,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0)
            runner.cancel_event = self.cancel_event
            LongHorizonCampaign(base_campaign=child, max_episodes=episode,
                                session_runner=runner).run()
        if self._state(child)["episodes"] != episode:
            raise RuntimeError("Episode did not reach a durable terminal outcome")
        attempt_path = campaign_store.episode_dir(episode) / "attempt.json"
        attempt = read_state(attempt_path)
        if attempt.get("episode") != episode:
            raise RuntimeError("Episode archive identity mismatch")
        agent_head = attempt.get("agent_head")
        if not isinstance(agent_head, str) or not agent_head:
            raise RuntimeError("Completed audit segment has no Agent-owned HEAD commit identity")
        submitted = accepted_submission_kernel_ids(
            supervisor_submission_root(child.workspace),
            lineage=f"T{trajectory}",
        )
        frontier = kernel_pareto_frontier(
            MeasurementStore(self.measurement_root),
            kernel_ids=submitted,
        )
        best_submitted = frontier.get("best_full_evaluation")
        return {"epoch": epoch, "trajectory": trajectory, "episode": episode,
                "status": attempt["status"],
                "agent_head_commit_id": agent_head,
                "best_submitted_evaluation": best_submitted,
                "archive": str(campaign_store.episode_dir(episode)),
                "tokens": attempt.get("tokens", 0),
                "session_id": attempt.get("session_id")}

    def tokens(self):
        return sum(self._state(child)["tokens"] for child in self.children.values())

    def interrupt(self):
        # Runtime cancellation also stops its bound Agent invocation through
        # the public owner grant. Keep unfinished workspaces for recovery.
        self.cancel_event.set()
        with ExitStack() as cleanup:
            for child in self.children.values():
                cleanup.callback(child.close_runtime)

    def close(self):
        self.interrupt()
