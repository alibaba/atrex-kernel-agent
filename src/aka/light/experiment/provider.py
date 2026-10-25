"""Compose Evaluation with narrow execution/publication callbacks, not the HTTP host."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
import subprocess

from aka.light.experiment.store import MeasurementStore


@dataclass(frozen=True)
class MeasurementEffects:
    mode: str
    repetitions: int
    private_reference_dir: Path | None
    private_paths: tuple[str, ...]
    run: Callable[[list[str], Path, dict[str, str]], subprocess.CompletedProcess]
    audit: Callable[[list[str], subprocess.CompletedProcess, str], None]
    publish_log: Callable[[Path], None]
    publish_outputs: Callable[[Path, object], None]
    deadline: float | None = None
    request_id: str = ""


class RecordedExperiments:
    def __init__(self, root: Path, evaluation):
        self.store = MeasurementStore(root)
        self.evaluation = evaluation

    def query(self, args, workspace):
        from .records import query
        return query(self.store, args, workspace)

    def task_identity(self, staged, args, environment, *, mode, repetitions):
        from aka.light.runtime.request_errors import AgentRequestError
        try:
            return self.evaluation.prepare(
                args, staged, environment, identity_key=self.store.identity_key,
                mode=mode, repetitions=repetitions,
            ).request
        except ValueError as error:
            raise AgentRequestError(str(error)) from error

    def execute(self, staged, args, argv, environment, command, *, effects,
                reuse_completed=False, reuse_correctness=False):
        from .records import execute
        return execute(self.store, self.evaluation, effects, staged, args, argv, environment, command,
                       reuse_completed=reuse_completed, reuse_correctness=reuse_correctness)


@dataclass(frozen=True)
class RecordedExperimentFactory:
    evaluation: object

    def open(self, root: Path):
        return RecordedExperiments(root, self.evaluation)
