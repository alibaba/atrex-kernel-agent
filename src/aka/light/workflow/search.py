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

from aka.light.agent.home import open_private_directory
from aka.legacy.task.persistence import durable_write_json
from aka.light.agent.session_tail import read_regular_bytes


@dataclass(frozen=True)
class SearchOptions:
    epochs: int = 10
    trajectories: int = 3
    episodes_per_trajectory: int = 1
    max_parallel_episodes: int = 3
    synchronization: str = "synchronous"
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


class PooledSearch:
    """Durable scheduling and peer-evidence publication for Agent trajectories."""

    SCHEMA_VERSION = 4

    def __init__(self, root: Path, options: SearchOptions, driver, *, token_budget=0):
        self.root, self.options, self.driver = root, options, driver
        self.token_budget = token_budget
        self.path = root / "state.json"

    def _save(self, state):
        durable_write_json(self.path, state)

    def run(self) -> str:
        with open_private_directory(self.root) as directory:
            fd = os.open(
                "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory
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

    def _new_lineages(self):
        return {
            str(trajectory): {
                "agent_head_commit_id": None,
                "session_id": None,
                "completed_episodes": 0,
                "lineage_winner": None,
            }
            for trajectory in range(1, self.options.trajectories + 1)
        }

    def _run(self):
        binding = self.driver.identity()
        try:
            state = read_state(self.path)
        except FileNotFoundError:
            bootstrap = self.driver.bootstrap()
            state = {
                "schema_version": self.SCHEMA_VERSION,
                "options": asdict(self.options),
                "binding": binding,
                "token_budget": self.token_budget,
                "bootstrap": bootstrap,
                "lineages": self._new_lineages(),
                "epoch": 1,
                "active": None,
                "completed_epochs": 0,
                "global_best": None,
            }
            self._save(state)
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                "This checkpoint predates persistent submitted-winner sharing; "
                "start a new pooled Campaign with the current search contract"
            )
        if (
            state.get("options") != asdict(self.options)
            or state.get("binding") != binding
            or state.get("token_budget") != self.token_budget
        ):
            raise ValueError(
                "Search configuration or Campaign contract changed; resume with the "
                "original configuration"
            )
        self.driver.initialize(state["bootstrap"], self.options.trajectories)
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
            if active["phase"] == "sealing":
                self.driver.seal_epoch(epoch, active["jobs"])
                active["phase"] = "publishing"
                self._save(state)
            elif active["phase"] == "publishing" and not self.driver.epoch_sealed(epoch):
                self.driver.seal_epoch(epoch, active["jobs"])

            for job in active["jobs"]:
                lineage = state["lineages"][str(job["trajectory"])]
                lineage["completed_episodes"] = max(
                    lineage["completed_episodes"], job["episode"]
                )
                head = job["result"].get("agent_head_commit_id")
                if isinstance(head, str) and head:
                    lineage["agent_head_commit_id"] = head
                session_id = job["result"].get("session_id")
                if isinstance(session_id, str) and session_id:
                    lineage["session_id"] = session_id
                observed = job["result"].get("best_submitted_evaluation")
                if isinstance(observed, dict):
                    lineage["lineage_winner"] = observed

            self._refresh_observed_winners(state)

            summary = dict(active, phase="completed", tokens=self.driver.tokens())
            durable_write_json(self.root / "epochs" / f"e{epoch:08d}.json", summary)
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

    @staticmethod
    def _refresh_observed_winners(state):
        """Update observational winners without changing search execution."""
        candidates = [
            dict(item, trajectory=int(trajectory))
            for trajectory, lineage in state["lineages"].items()
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
                    state["lineages"][str(trajectory)].get("completed_episodes") == total
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
                    lineage = state["lineages"][key]
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
        except BaseException:
            self.driver.interrupt()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _finalize_asynchronous(self, state, trajectory, job):
        asynchronous = state["asynchronous"]
        key = str(trajectory)
        lineage = state["lineages"][key]
        if job["status"] == "completed":
            result = job.get("result")
            if not isinstance(result, dict):
                raise ValueError("Completed asynchronous Episode has no result")
            publication = asynchronous["next_publication"]
            if type(publication) is not int or publication < 1:
                raise ValueError("Asynchronous publication sequence is invalid")
            terminal_episodes = {
                str(number): state["lineages"][str(number)].get(
                    "completed_episodes", 0
                )
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
            "schema_version": 2,
            "synchronization": "asynchronous",
            "publication": publication,
            "trajectory": trajectory,
            "episode": job["episode"],
            "epoch": job["epoch"],
            "result": job["result"],
            "tokens": self.driver.tokens(),
        }
        durable_write_json(
            self.root / "async-publications" / f"p{publication:08d}.json", summary
        )

        lineage["completed_episodes"] = job["episode"]
        head = job["result"].get("agent_head_commit_id")
        if isinstance(head, str) and head:
            lineage["agent_head_commit_id"] = head
        session_id = job["result"].get("session_id")
        if isinstance(session_id, str) and session_id:
            lineage["session_id"] = session_id
        observed = job["result"].get("best_submitted_evaluation")
        if isinstance(observed, dict):
            lineage["lineage_winner"] = observed
        self._refresh_observed_winners(state)
        asynchronous.update(
            latest_publication=publication,
            next_publication=publication + 1,
        )
        asynchronous["jobs"][key] = None

        completed = [
            state["lineages"][str(number)]["completed_episodes"]
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
                    except Exception as error:
                        failure = failure or error
                if failure is not None and running:
                    continue
        except BaseException:
            self.driver.interrupt()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
