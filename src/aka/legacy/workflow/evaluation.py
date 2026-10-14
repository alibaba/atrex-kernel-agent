"""Consume explicit measurement and record dependencies at Workflow boundaries."""

import json

from aka.contracts.acceptance import AcceptanceRequirements
from aka.contracts.evaluation import EvaluationRequest, VerificationResult
from aka.contracts.experiment import ExperimentExecutionError


class FormalVerification:
    def __init__(self, experiment, acceptance, threshold):
        self.experiment = experiment
        self.acceptance = acceptance
        self.threshold = threshold

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        request = EvaluationRequest(
            workspace,
            base_commit=base_commit,
            candidate_commit=candidate_commit,
            changed_paths=tuple(changed_paths),
        )
        try:
            measurement = self.experiment.run(request)
        except ExperimentExecutionError as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"gateway ABBA verification failed: {exc}",
            )
        return self.acceptance.decide(
            measurement, AcceptanceRequirements(min_improvement_pct=self.threshold)
        ).verification


class EpisodeMeasurements:
    def __init__(
        self,
        workspace,
        private_reference_dir,
        threshold,
        *,
        evaluator,
        acceptance,
        records,
        command,
    ):
        self.workspace = workspace
        self.private_reference_dir = private_reference_dir
        self.threshold = threshold
        self.evaluator = evaluator
        self.acceptance = acceptance
        self.records = records
        self.command = command

    def expected_shape_ids(self):
        if self.private_reference_dir is None:
            return None
        return set(
            json.loads(
                (self.private_reference_dir / "shapes.json").read_text(encoding="utf-8")
            )
        )

    def fast_evaluator_command(self, version):
        return self.command(
            ["python", "test_kernel.py", "--version", f"v{version}", "--no-memory"]
        )

    def fast_verification_result(self, workspace, *, memory_version):
        shapes = self.expected_shape_ids()
        request = EvaluationRequest(
            workspace,
            mode="fast",
            memory_version=memory_version,
            canonical_workspace=self.workspace,
            expected_shape_ids=frozenset(shapes) if shapes is not None else None,
        )
        prepared = self.evaluator.prepare(request)
        measurement = self.evaluator.interpret(prepared, ())
        return self.acceptance.decide(
            measurement, AcceptanceRequirements(self.threshold, mode="fast")
        ).verification

    def conversion_parity_passes(self, verification):
        return self.acceptance.conversion_parity_passes(verification)

    def iso_timestamp(self, value):
        return self.records.iso_timestamp(value)

    def representative_candidate_result(self, verification):
        return self.records.representative_candidate_result(verification)

    def candidate_shape_latencies(self, verification):
        return self.records.candidate_shape_latencies(verification)

    def latest_complete_canonical_performance(self, workspace, **options):
        return self.records.latest_complete_canonical_performance(workspace, **options)

    def latest_complete_episode_performance(self, workspace, **options):
        return self.records.latest_complete_episode_performance(workspace, **options)

    def episode_evaluation_count(self, workspace):
        return self.records.episode_evaluation_count(workspace)
