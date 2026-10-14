"""Translate original episode calls to the public Evaluation/Acceptance contracts."""

from __future__ import annotations

import json
from pathlib import Path

from aka.contracts.acceptance import AcceptanceRequirements
from aka.contracts.evaluation import EvaluationRequest
from aka.legacy.acceptance import LegacyAcceptancePolicy
from aka.legacy.evaluation import LegacyEvaluator
from aka.legacy.evaluation import results


class EpisodeEvaluationAdapter:
    def __init__(
        self,
        workspace,
        private_reference_dir,
        threshold,
        *,
        evaluator,
        acceptance,
        command,
    ):
        self.workspace = workspace
        self.private_reference_dir = private_reference_dir
        self.threshold = threshold
        self.evaluator = evaluator
        self.acceptance = acceptance
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
        return self.command(version)

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

    def __getattr__(self, name):
        return getattr(results, "_" + name)


def episode_adapter(implementation, campaign, *, settings=None, base=None):
    from .workflow import bind_evaluation, EvaluationContext
    from long_horizon.legacy_evaluation.episode import LegacyEpisodeEvaluation

    if implementation is not None:
        return bind_evaluation(implementation, campaign, settings=settings, base=base)
    context = EvaluationContext(campaign, LegacyEpisodeEvaluation(), settings, base)
    return EpisodeEvaluationAdapter(
        campaign.workspace,
        base.private_reference_dir,
        float(base.min_improvement_pct),
        evaluator=LegacyEvaluator(),
        acceptance=LegacyAcceptancePolicy(),
        command=lambda version: LegacyEpisodeEvaluation.fast_evaluator_command(
            context, version
        ),
    )
