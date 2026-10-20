"""Compose Evaluation with narrow execution/publication callbacks, not the HTTP host."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
import subprocess

from aka.light.supervisor.measurement_records import MeasurementStore


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


class RecordedExperiments:
    def __init__(self, root: Path, evaluation):
        self.store = MeasurementStore(root)
        self.evaluation = evaluation

    def query(self, args, workspace):
        from .records import query
        return query(self.store, args, workspace)

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
