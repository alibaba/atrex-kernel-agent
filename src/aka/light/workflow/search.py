"""Optional pooled search: a Runtime Attempt is one ordinary Light Episode.

Only the outer search topology changes. The driver retains Episode reporting,
correctness, ABBA, sealed candidates and recovery. A barrier publishes a Kernel,
never a live draft or another trajectory's adaptive state.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path

from aka.light.orchestrator.agent_home import open_private_directory
from aka.light.orchestrator.durable_state import durable_write_json
from aka.light.orchestrator.session_tail import read_regular_bytes


@dataclass(frozen=True)
class SearchOptions:
    epochs: int = 10
    trajectories: int = 2
    episodes_per_trajectory: int = 1
    max_parallel_episodes: int = 2

    def __post_init__(self):
        for name, maximum in (("epochs", 10000), ("trajectories", 64),
                              ("episodes_per_trajectory", 100), ("max_parallel_episodes", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"search.{name} must be an integer in 1..{maximum}")
        if self.max_parallel_episodes > self.trajectories:
            raise ValueError("search.max_parallel_episodes cannot exceed trajectories")


def read_state(path: Path) -> dict:
    data = read_regular_bytes(path, limit=16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Search checkpoint exceeds 16 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Search checkpoint must be an object")
    return value


def latency(kernel: dict) -> float:
    value = kernel["latency_us_geomean"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Search selection requires a finite positive authoritative GeoMean latency")
    return float(value)


class PooledSearch:
    """Durable Epoch barrier and bounded admission; storage/execution are injected.

    Driver methods run ordinary Episodes in private trajectory controller workspaces.
    All checkpoint writes belong to this coordinator thread, not executor threads.
    """
    def __init__(self, root: Path, options: SearchOptions, driver, *, token_budget=0):
        self.root, self.options, self.driver = root, options, driver
        self.token_budget = token_budget
        self.path = root / "state.json"

    def _save(self, state):
        durable_write_json(self.path, state)

    def run(self) -> str:
        with open_private_directory(self.root) as directory:
            fd = os.open("lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("This pooled Campaign already has an active coordinator") from None
                return self._run()
            finally:
                os.close(fd)

    def _run(self):
        binding = self.driver.identity()
        try:
            state = read_state(self.path)
        except FileNotFoundError:
            seed = self.driver.bootstrap()
            latency(seed)
            state = {"schema_version": 1, "options": asdict(self.options), "binding": binding,
                     "token_budget": self.token_budget, "bootstrap": seed,
                     "seed": seed, "epoch": 1, "active": None, "completed_epochs": 0}
            self._save(state)
        if (state.get("schema_version") != 1 or state.get("options") != asdict(self.options)
                or state.get("binding") != binding or state.get("token_budget") != self.token_budget):
            raise ValueError("Search configuration or Campaign contract changed; resume with the original configuration")
        self.driver.initialize(state["bootstrap"], self.options.trajectories)
        if state.get("stop_reason"):
            return state["stop_reason"]
        while state["epoch"] <= self.options.epochs:
            epoch = state["epoch"]
            if state["active"] is None:
                if self._budget_reached():
                    return "budget: token-budget"
                jobs = []
                for step in range(1, self.options.episodes_per_trajectory + 1):
                    for trajectory in range(1, self.options.trajectories + 1):
                        episode = (epoch - 1) * self.options.episodes_per_trajectory + step
                        jobs.append({"epoch": epoch, "trajectory": trajectory, "step": step,
                                     "episode": episode, "status": "pending"})
                state["active"] = {"epoch": epoch, "seed": state["seed"], "phase": "preparing", "jobs": jobs}
                self._save(state)
            active = state["active"]
            if active["phase"] == "preparing":
                self.driver.prepare_epoch(epoch, active["seed"])
                active["phase"] = "running"
                self._save(state)
            if active["phase"] == "running":
                if not self._run_jobs(state):
                    # Partial Epochs are retained, but never broadcast as complete.
                    return "budget: token-budget (Epoch incomplete; no broadcast)"
                candidates = [active["seed"]]
                for job in active["jobs"]:
                    candidate = job["result"].get("retained_kernel")
                    if candidate is not None:
                        latency(candidate)
                        candidates.append(candidate)
                # The incumbent wins exact ties; ordering is independent of completion order.
                active["winner"] = min(candidates, key=latency)
                active["phase"] = "publishing"
                self._save(state)
            self.driver.publish_epoch(epoch, active["winner"])
            summary = dict(active, phase="completed", tokens=self.driver.tokens())
            durable_write_json(self.root / "epochs" / f"e{epoch:08d}.json", summary)
            stalls = state.get("consecutive_without_improvement", 0)
            state["consecutive_without_improvement"] = (
                stalls + 1 if latency(active["winner"]) >= latency(active["seed"]) else 0)
            state.update(seed=active["winner"], completed_epochs=epoch, epoch=epoch + 1, active=None)
            stop = getattr(self.driver, "stop_reason", lambda state: "")(state)
            if stop:
                state["stop_reason"] = stop
            self._save(state)
            print(f"[pooled-search] epoch={epoch} completed "
                  f"episodes={len(active['jobs'])} best_geomean_us={latency(active['winner']):.6f} "
                  f"tokens={summary['tokens']}", flush=True)
            if stop:
                return stop
        return "search: completed Epoch budget"

    def _budget_reached(self):
        return self.token_budget > 0 and self.driver.tokens() >= self.token_budget

    def _run_jobs(self, state):
        jobs = state["active"]["jobs"]
        pool = ThreadPoolExecutor(max_workers=self.options.max_parallel_episodes, thread_name_prefix="aka-trajectory")
        running = {}
        failure = None
        try:
            while True:
                busy = {job["trajectory"] for job in running.values()}
                # A restarted running job resumes its exact Episode even at the token limit.
                for job in jobs:
                    if failure is not None or len(running) >= self.options.max_parallel_episodes:
                        break
                    if job["status"] == "completed" or job["trajectory"] in busy:
                        continue
                    if any(previous["status"] != "completed" for previous in jobs
                           if previous["trajectory"] == job["trajectory"] and previous["episode"] < job["episode"]):
                        continue
                    if job["status"] != "running" and self._budget_reached():
                        continue
                    job["status"] = "running"
                    self._save(state)  # Write-ahead identity, before starting any Agent.
                    future = pool.submit(self.driver.run_episode, job["epoch"], job["trajectory"], job["episode"])
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
                        if (result.get("epoch"), result.get("trajectory"), result.get("episode")) != (
                                job["epoch"], job["trajectory"], job["episode"]):
                            raise RuntimeError("Episode result does not match its scheduled identity")
                        job.update(status="completed", result=result)
                        self._save(state)
                    except Exception as error:
                        # Keep 'running': underlying Episode recovery owns unknown outcomes.
                        failure = failure or error
                if failure is not None and running:
                    # Already-dispatched siblings may finish; never dispatch later Episodes.
                    continue
        except BaseException:
            self.driver.interrupt()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
