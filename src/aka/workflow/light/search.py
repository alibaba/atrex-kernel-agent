"""Optional pooled search over persistent Agent-owned trajectories.

Every trajectory owns one durable Agent Git repository and one resumable provider
Session. An Episode is only an audit segment created when that Agent voluntarily
stops. The coordinator admits segments and publishes immutable peer Git,
Direction, Experiment and Gateway Record evidence. It derives and broadcasts
lineage/global winners from accepted ``submit-kernel`` commits with passing full
Evaluate evidence, but a winner never rewrites an Agent checkout or rolls a
trajectory back.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path

from aka.core.filesystem import open_private_directory, durable_write_json, read_regular_bytes


COORDINATOR_STATE = "coordinator.json"
COORDINATOR_LOCK = "coordinator.lock"


@dataclass(frozen=True)
class SearchOptions:
    epochs: int = 10
    trajectories: int = 3
    episodes_per_trajectory: int = 1
    max_parallel_episodes: int = 3
    synchronization: str = "asynchronous"
    role_specialization: bool = False

    def __post_init__(self):
        for name, maximum in (
            ("epochs", 10000),
            ("trajectories", 64),
            ("episodes_per_trajectory", 100),
            ("max_parallel_episodes", 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"search.{name} must be an integer in 1..{maximum}")
        if self.max_parallel_episodes > self.trajectories:
            raise ValueError("search.max_parallel_episodes cannot exceed trajectories")
        if self.synchronization not in {"synchronous", "asynchronous"}:
            raise ValueError(
                "search.synchronization must be synchronous or asynchronous"
            )
        if type(self.role_specialization) is not bool:
            raise TypeError("search.role_specialization must be a boolean")


def read_state(path: Path) -> dict:
    data = read_regular_bytes(path, limit=16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Search checkpoint exceeds 16 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Search checkpoint must be an object")
    return value


def publication_root(root: Path, publication: int) -> Path:
    if type(publication) is not int or publication < 1:
        raise ValueError("Search publication sequence must be a positive integer")
    return root / "publications" / f"p{publication:08d}"


def publication_path(root: Path, publication: int) -> Path:
    return publication_root(root, publication) / "publication.json"


def publication_source_root(
    root: Path,
    publication: int,
    trajectory: int,
) -> Path:
    if type(trajectory) is not int or trajectory < 1:
        raise ValueError("Search trajectory must be a positive integer")
    return publication_root(root, publication) / "sources" / f"t{trajectory:04d}"


def publication_import_path(
    root: Path,
    publication: int,
    trajectory: int,
) -> Path:
    if type(trajectory) is not int or trajectory < 1:
        raise ValueError("Search trajectory must be a positive integer")
    return publication_root(root, publication) / "imports" / f"t{trajectory:04d}.json"


def lineage_state_path(root: Path, trajectory: int) -> Path:
    if type(trajectory) is not int or trajectory < 1:
        raise ValueError("Search trajectory must be a positive integer")
    return root / "lineages" / f"t{trajectory:04d}" / "state.json"


def merge_lineage_state(
    path: Path,
    trajectory: int,
    **changes: object,
) -> dict:
    """Merge scheduler/runtime facts into one lineage-owned checkpoint."""
    if type(trajectory) is not int or trajectory < 1:
        raise ValueError("Search trajectory must be a positive integer")
    try:
        value = read_state(path)
    except FileNotFoundError:
        value = {}
    if value.get("trajectory") not in {None, trajectory} or value.get("lineage") not in {
        None,
        f"T{trajectory}",
    }:
        raise RuntimeError("Lineage checkpoint identity changed during recovery")
    value.update(
        schema_version=1,
        lineage=f"T{trajectory}",
        trajectory=trajectory,
    )
    value.update(changes)
    durable_write_json(path, value)
    return value


def update_publication(
    root: Path,
    publication: int,
    synchronization: str,
    **components: object,
) -> dict:
    """Atomically add immutable components to one publication record."""
    if synchronization not in {"synchronous", "asynchronous"}:
        raise ValueError("Search publication synchronization is invalid")
    path = publication_path(root, publication)
    try:
        value = read_state(path)
    except FileNotFoundError:
        value = {
            "schema_version": 1,
            "publication": publication,
            "synchronization": synchronization,
        }
    if (
        value.get("schema_version") != 1
        or value.get("publication") != publication
        or value.get("synchronization") != synchronization
    ):
        raise RuntimeError("Search publication identity changed during recovery")
    allowed = {"watermark", "sources", "evidence", "scheduler"}
    unknown = set(components) - allowed
    if unknown:
        raise ValueError(
            "Unknown Search publication components: " + ", ".join(sorted(unknown))
        )
    for name, component in components.items():
        if name in value and value[name] != component:
            raise RuntimeError(
                f"Search publication {name} changed during recovery"
            )
        value[name] = component
    durable_write_json(path, value)
    return value


class PooledSearch:
    """Durable scheduling and peer-evidence publication for Agent trajectories."""

    SCHEMA_VERSION = 5

    def __init__(self, root: Path, options: SearchOptions, driver, *, token_budget=0):
        self.root, self.options, self.driver = root, options, driver
        self.token_budget = token_budget
        self.path = root / COORDINATOR_STATE

    def _save(self, state):
        # coordinator.json is the authoritative scheduler checkpoint.  Keep the
        # immutable launch options beside its mutable progress so read-only
        # observability does not need to infer the Campaign configuration from
        # whichever jobs happen to be active at the time of inspection.
        state["options"] = asdict(self.options)
        state["token_budget"] = self.token_budget
        durable_write_json(self.path, state)

    def run(self) -> str:
        with open_private_directory(self.root) as directory:
            fd = os.open(
                COORDINATOR_LOCK,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError(
                        "This pooled Campaign already has an active coordinator"
                    ) from None
                return self._run()
            finally:
                os.close(fd)

    def _lineage(self, trajectory: int) -> dict:
        path = lineage_state_path(self.root, trajectory)
        try:
            value = read_state(path)
        except FileNotFoundError:
            return {
                "schema_version": 1,
                "lineage": f"T{trajectory}",
                "trajectory": trajectory,
                "agent_head_commit_id": None,
                "session_id": None,
                "completed_episodes": 0,
                "lineage_winner": None,
            }
        if value.get("trajectory") not in {None, trajectory} or value.get(
            "lineage"
        ) not in {None, f"T{trajectory}"}:
            raise ValueError("Search lineage checkpoint has an invalid identity")
        value.setdefault("agent_head_commit_id", None)
        value.setdefault("session_id", None)
        value.setdefault("completed_episodes", 0)
        value.setdefault("lineage_winner", None)
        return value

    def _save_lineage(self, trajectory: int, lineage: dict) -> None:
        merge_lineage_state(
            lineage_state_path(self.root, trajectory),
            trajectory,
            **{
                name: lineage.get(name)
                for name in (
                    "agent_head_commit_id",
                    "session_id",
                    "completed_episodes",
                    "lineage_winner",
                )
            },
        )

    def _ensure_lineages(self) -> None:
        for trajectory in range(1, self.options.trajectories + 1):
            self._save_lineage(trajectory, self._lineage(trajectory))

    def _record_completed_episode(self, job: dict) -> None:
        """Project one durable Episode result into its lineage immediately.

        Publication is an Epoch-level operation and may legitimately wait for
        peer trajectories.  Episode progress is lineage-local, so delaying it
        until publication makes an already stopped Session look stalled.  The
        coordinator job is persisted before this projection; replaying this
        method during recovery is therefore idempotent.
        """
        if job.get("status") not in {"completed", "sealing"}:
            return
        trajectory = job.get("trajectory")
        episode = job.get("episode")
        result = job.get("result")
        if type(trajectory) is not int or trajectory < 1:
            raise ValueError("Completed search job has an invalid trajectory")
        if type(episode) is not int or episode < 1:
            raise ValueError("Completed search job has an invalid Episode")
        if not isinstance(result, dict):
            raise TypeError("Completed search job has no result")

        lineage = self._lineage(trajectory)
        previous = lineage.get("completed_episodes", 0)
        if type(previous) is not int or previous < 0:
            raise ValueError("Search lineage has invalid Episode progress")
        lineage["completed_episodes"] = max(previous, episode)
        # Never let recovery of an older result replace the identity attached
        # to a later completed Episode.
        if episode >= previous:
            head = result.get("agent_head_commit_id")
            if isinstance(head, str) and head:
                lineage["agent_head_commit_id"] = head
            session_id = result.get("session_id")
            if isinstance(session_id, str) and session_id:
                lineage["session_id"] = session_id
            observed = result.get("best_submitted_evaluation")
            if isinstance(observed, dict):
                lineage["lineage_winner"] = observed
        self._save_lineage(trajectory, lineage)

    def _run(self):
        try:
            state = read_state(self.path)
        except FileNotFoundError:
            bootstrap = self.driver.bootstrap()
            state = {
                "schema_version": self.SCHEMA_VERSION,
                "epoch": 1,
                "active": None,
                "completed_epochs": 0,
                "global_best": None,
            }
            self._save(state)
        else:
            bootstrap = None
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                "This checkpoint predates persistent submitted-winner sharing; "
                "start a new pooled Campaign with the current search contract"
            )
        # campaign.json already freezes the launch composition and V0 identity;
        # coordinator.json contains only mutable scheduler state. Reconstruct
        # the bootstrap view from those Campaign authorities on recovery.
        if bootstrap is None:
            bootstrap = self.driver.bootstrap()
        self.driver.initialize(bootstrap, self.options.trajectories)
        self._ensure_lineages()
        if self.options.synchronization == "asynchronous":
            return self._run_asynchronous(state)
        return self._run_synchronous(state)

    def _run_synchronous(self, state):
        while state["epoch"] <= self.options.epochs:
            epoch = state["epoch"]
            if state["active"] is None:
                if self._budget_reached():
                    return "budget: token-budget"
                jobs = []
                for step in range(1, self.options.episodes_per_trajectory + 1):
                    for trajectory in range(1, self.options.trajectories + 1):
                        episode = (
                            (epoch - 1) * self.options.episodes_per_trajectory + step
                        )
                        jobs.append(
                            {
                                "epoch": epoch,
                                "trajectory": trajectory,
                                "step": step,
                                "episode": episode,
                                "status": "pending",
                            }
                        )
                state["active"] = {
                    "epoch": epoch,
                    "phase": "preparing",
                    "jobs": jobs,
                }
                self._save(state)
            active = state["active"]
            if active["phase"] == "preparing":
                self.driver.prepare_epoch(epoch)
                active["phase"] = "running"
                self._save(state)
            if active["phase"] == "running":
                if not self._run_jobs(state):
                    return "budget: token-budget (Epoch incomplete; no peer publication)"
                active["phase"] = "sealing"
                self._save(state)
            # Reconcile completed jobs before potentially slow peer sealing.
            # This also repairs a crash after the coordinator job was saved but
            # before its lineage projection was written.
            for job in active["jobs"]:
                self._record_completed_episode(job)
            self._refresh_observed_winners(state)
            if active["phase"] == "sealing":
                self.driver.seal_epoch(epoch, active["jobs"])
                active["phase"] = "publishing"
                self._save(state)
            elif active["phase"] == "publishing" and not self.driver.epoch_sealed(epoch):
                self.driver.seal_epoch(epoch, active["jobs"])

            summary = dict(active, phase="completed", tokens=self.driver.tokens())
            update_publication(
                self.root,
                epoch,
                "synchronous",
                scheduler=summary,
            )
            state.update(completed_epochs=epoch, epoch=epoch + 1, active=None)
            self._save(state)
            print(
                f"[pooled-search] epoch={epoch} completed "
                f"segments={len(active['jobs'])} tokens={summary['tokens']}",
                flush=True,
            )
        return "search: completed Epoch budget"

    def _budget_reached(self):
        return self.token_budget > 0 and self.driver.tokens() >= self.token_budget

    def _refresh_observed_winners(self, state):
        """Update observational winners without changing search execution."""
        candidates = [
            dict(item, trajectory=int(trajectory))
            for trajectory in range(1, self.options.trajectories + 1)
            for lineage in (self._lineage(trajectory),)
            if isinstance(item := lineage.get("lineage_winner"), dict)
        ]
        candidates = [
            item
            for item in candidates
            if type(item.get("latency_us_geomean")) in {int, float}
            and float(item["latency_us_geomean"]) > 0
        ]
        state["global_best"] = (
            min(candidates, key=lambda item: float(item["latency_us_geomean"]))
            if candidates
            else None
        )

    def _run_asynchronous(self, state):
        """Run persistent trajectories and publish cumulative immutable snapshots."""
        total = self.options.epochs * self.options.episodes_per_trajectory
        asynchronous = state.get("asynchronous")
        if asynchronous is None:
            if state.get("active") is not None or state.get("completed_epochs", 0):
                raise ValueError(
                    "Asynchronous search requires a fresh asynchronous checkpoint"
                )
            asynchronous = {
                "jobs": {
                    str(trajectory): None
                    for trajectory in range(1, self.options.trajectories + 1)
                },
                "next_admission_trajectory": 1,
                "next_publication": 1,
                "latest_publication": 0,
            }
            state["asynchronous"] = asynchronous
            self._save(state)
        jobs = asynchronous.get("jobs")
        expected = {str(item) for item in range(1, self.options.trajectories + 1)}
        if not isinstance(jobs, dict) or set(jobs) != expected:
            raise ValueError("Asynchronous search checkpoint has invalid trajectory jobs")

        pool = ThreadPoolExecutor(
            max_workers=self.options.max_parallel_episodes,
            thread_name_prefix="aka-trajectory",
        )
        running = {}
        try:
            while True:
                for trajectory in range(1, self.options.trajectories + 1):
                    job = jobs[str(trajectory)]
                    if isinstance(job, dict) and job.get("status") in {
                        "completed",
                        "sealing",
                    }:
                        self._finalize_asynchronous(state, trajectory, job)

                complete = all(
                    self._lineage(trajectory).get("completed_episodes") == total
                    for trajectory in range(1, self.options.trajectories + 1)
                )
                if complete and not running:
                    return "search: completed asynchronous Episode budget"

                first = asynchronous.get("next_admission_trajectory", 1)
                if type(first) is not int or not 1 <= first <= self.options.trajectories:
                    raise ValueError("Asynchronous admission cursor is invalid")
                admission_order = [
                    (first - 1 + offset) % self.options.trajectories + 1
                    for offset in range(self.options.trajectories)
                ]
                for trajectory in admission_order:
                    if len(running) >= self.options.max_parallel_episodes:
                        break
                    key = str(trajectory)
                    if trajectory in running.values():
                        continue
                    lineage = self._lineage(trajectory)
                    completed = lineage.get("completed_episodes")
                    if type(completed) is not int or not 0 <= completed <= total:
                        raise ValueError(
                            "Asynchronous search checkpoint has invalid Episode progress"
                        )
                    if completed >= total:
                        continue
                    job = jobs[key]
                    if job is None:
                        if self._budget_reached():
                            continue
                        episode = completed + 1
                        job = {
                            "epoch": (
                                (episode - 1) // self.options.episodes_per_trajectory + 1
                            ),
                            "trajectory": trajectory,
                            "episode": episode,
                            "status": "preparing",
                            "shared_publication": asynchronous["latest_publication"],
                        }
                        jobs[key] = job
                        self._save(state)
                    if job.get("status") not in {"preparing", "running"}:
                        continue
                    self.driver.prepare_trajectory(
                        job["epoch"],
                        trajectory,
                        job["episode"],
                        job["shared_publication"],
                    )
                    job["status"] = "running"
                    asynchronous["next_admission_trajectory"] = (
                        trajectory % self.options.trajectories + 1
                    )
                    self._save(state)
                    future = pool.submit(
                        self.driver.run_episode,
                        job["epoch"],
                        trajectory,
                        job["episode"],
                    )
                    running[future] = trajectory

                if not running:
                    if self._budget_reached():
                        return "budget: token-budget (asynchronous search partially complete)"
                    raise RuntimeError(
                        "Asynchronous search has unfinished work but no runnable Episode"
                    )

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    trajectory = running.pop(future)
                    job = jobs[str(trajectory)]
                    try:
                        result = future.result()
                        if (
                            result.get("epoch"),
                            result.get("trajectory"),
                            result.get("episode"),
                        ) != (job["epoch"], trajectory, job["episode"]):
                            raise RuntimeError(
                                "Episode result does not match its scheduled identity"
                            )
                    except Exception:
                        self.driver.interrupt()
                        raise
                    job.update(status="completed", result=result)
                    self._save(state)
                    self._record_completed_episode(job)
        except BaseException:
            self.driver.interrupt()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _finalize_asynchronous(self, state, trajectory, job):
        asynchronous = state["asynchronous"]
        key = str(trajectory)
        # A recovered completed/sealing job may predate its lineage projection.
        self._record_completed_episode(job)
        if job["status"] == "completed":
            result = job.get("result")
            if not isinstance(result, dict):
                raise ValueError("Completed asynchronous Episode has no result")
            publication = asynchronous["next_publication"]
            if type(publication) is not int or publication < 1:
                raise ValueError("Asynchronous publication sequence is invalid")
            terminal_episodes = {
                str(number): self._lineage(number).get("completed_episodes", 0)
                for number in range(1, self.options.trajectories + 1)
            }
            terminal_episodes[key] = job["episode"]
            job.update(
                status="sealing",
                publication=publication,
                terminal_episodes=terminal_episodes,
            )
            self._save(state)

        publication = job["publication"]
        terminal_jobs = [
            {
                "epoch": publication,
                "trajectory": number,
                "episode": episode,
                "status": "completed",
            }
            for number, episode in sorted(
                (int(number), episode)
                for number, episode in job["terminal_episodes"].items()
                if type(episode) is int and episode > 0
            )
        ]
        seal = getattr(self.driver, "seal_publication", self.driver.seal_epoch)
        seal(publication, terminal_jobs)
        summary = {
            "trajectory": trajectory,
            "episode": job["episode"],
            "epoch": job["epoch"],
            "result": job["result"],
            "tokens": self.driver.tokens(),
        }
        update_publication(
            self.root,
            publication,
            "asynchronous",
            scheduler=summary,
        )

        self._refresh_observed_winners(state)
        asynchronous.update(
            latest_publication=publication,
            next_publication=publication + 1,
        )
        asynchronous["jobs"][key] = None

        completed = [
            self._lineage(number)["completed_episodes"]
            for number in range(1, self.options.trajectories + 1)
        ]
        common = min(completed) // self.options.episodes_per_trajectory
        state["completed_epochs"] = common
        state["epoch"] = min(common + 1, self.options.epochs + 1)
        self._save(state)
        print(
            f"[pooled-search] async publication={publication} trajectory={trajectory} "
            f"episode={job['episode']} tokens={summary['tokens']}",
            flush=True,
        )

    def _run_jobs(self, state):
        jobs = state["active"]["jobs"]
        pool = ThreadPoolExecutor(
            max_workers=self.options.max_parallel_episodes,
            thread_name_prefix="aka-trajectory",
        )
        running = {}
        failure = None
        try:
            while True:
                busy = {job["trajectory"] for job in running.values()}
                for job in jobs:
                    if failure is not None or len(running) >= self.options.max_parallel_episodes:
                        break
                    if job["status"] == "completed" or job["trajectory"] in busy:
                        continue
                    if any(
                        previous["status"] != "completed"
                        for previous in jobs
                        if previous["trajectory"] == job["trajectory"]
                        and previous["episode"] < job["episode"]
                    ):
                        continue
                    if job["status"] != "running" and self._budget_reached():
                        continue
                    job["status"] = "running"
                    self._save(state)
                    future = pool.submit(
                        self.driver.run_episode,
                        job["epoch"],
                        job["trajectory"],
                        job["episode"],
                    )
                    running[future] = job
                    busy.add(job["trajectory"])
                if not running:
                    if failure is not None:
                        raise failure
                    return all(job["status"] == "completed" for job in jobs)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    job = running.pop(future)
                    try:
                        result = future.result()
                        if (
                            result.get("epoch"),
                            result.get("trajectory"),
                            result.get("episode"),
                        ) != (job["epoch"], job["trajectory"], job["episode"]):
                            raise RuntimeError(
                                "Episode result does not match its scheduled identity"
                            )
                        job.update(status="completed", result=result)
                        self._save(state)
                        self._record_completed_episode(job)
                    except Exception as error:
                        failure = failure or error
                if failure is not None and running:
                    continue
        except BaseException:
            self.driver.interrupt()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
