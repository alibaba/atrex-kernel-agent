"""Version-local optional service boundaries; not new upstream-stable protocols.

Factories are composition-owned and stateless. Experiment stores and Journals
are Campaign/Episode-scoped; services borrow only explicit callbacks. Transport
authorization and lifetime never enter the evaluation or acceptance objects.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from .evaluation.provider import EvaluationPlan
from .experiment.provider import MeasurementEffects
from .acceptance.provider import CandidateValidation

if TYPE_CHECKING:
    from aka.light.experiment.store import MeasurementStore


class Evaluation(Protocol):
    def prepare(self, args, staged, environment, *, identity_key, mode, repetitions) -> EvaluationPlan: ...
    def project(self, process, plan, *, generalized, environment, private_paths) -> dict: ...
    def result(self, response, operation) -> dict | None: ...
    def aggregate(self, samples, plan) -> dict: ...


class Experiments(Protocol):
    # Existing Journal evidence readers share this private store; replacing its
    # storage schema is not implied by replacing the experiment coordinator.
    store: MeasurementStore

    def query(self, args, workspace: Path) -> dict: ...
    def task_identity(self, staged, args, environment, *, mode, repetitions) -> dict: ...
    def execute(self, staged, args, argv, environment, command, *, effects: MeasurementEffects,
                reuse_completed=False, reuse_correctness=False) -> dict: ...


class ExperimentFactory(Protocol):
    def open(self, root: Path) -> Experiments: ...


class Acceptance(Protocol):
    def validate_candidate(self, context: CandidateValidation, source: bytes) -> dict: ...
    def verifier(self, **options): ...


class JournalFactory(Protocol):
    research_records: bool

    def initialize(self, path, *, episode, base_commit, branch, memory_version): ...
    def open(self, **options): ...
