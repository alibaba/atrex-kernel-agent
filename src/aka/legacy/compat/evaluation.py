"""Translate original episode calls to the public Evaluation/Acceptance contracts."""

from __future__ import annotations

from aka.legacy.acceptance import LegacyAcceptancePolicy
from aka.legacy.evaluation import LegacyEvaluator


def episode_adapter(implementation, campaign, *, settings=None, base=None):
    from .workflow import bind_evaluation

    if implementation is not None:
        return bind_evaluation(implementation, campaign, settings=settings, base=base)
    from aka.legacy.workflow.evaluation import EpisodeMeasurements
    from aka.legacy.evaluation import records
    from .configuration import EvaluationSettings

    s = settings or getattr(base, "settings", None) or EvaluationSettings()

    def command(argv):
        import shlex

        prefix = ["python", "tools/sandbox.py", "--kind", "run"]
        hardware = getattr(base, "sandbox_hardware", s.hardware)
        if hardware:
            prefix += ["--hardware", hardware]
        for attribute, flag in (
            ("ssh", "--ssh"),
            ("url", "--url"),
            ("profile", "--gateway-profile"),
        ):
            value = getattr(base, "sandbox_" + attribute, getattr(s, attribute))
            if value:
                prefix += [flag, value]
                break
        return shlex.join([*prefix, "--no-sync", "--", *argv])

    return EpisodeMeasurements(
        campaign.workspace,
        base.private_reference_dir,
        float(getattr(base, "min_improvement_pct", s.min_improvement_pct)),
        evaluator=LegacyEvaluator(),
        acceptance=LegacyAcceptancePolicy(),
        records=records,
        command=command,
    )
