from __future__ import annotations

import json
import shlex
from pathlib import Path


from aka.legacy.compat.legacy_evaluation.models import VerificationResult, VerificationRun
from aka.legacy.compat.legacy_evaluation.results import _latest_complete_canonical_performance, _latest_complete_episode_performance, EPISODE_EVALUATIONS_PATH


class LegacyEpisodeEvaluation:
    @staticmethod
    def expected_shape_ids(self) -> set[str] | None:
        private_reference_dir = self.base_campaign.private_reference_dir
        if private_reference_dir is None:
            return None
        return set(
            json.loads(
                (private_reference_dir / "shapes.json").read_text(encoding="utf-8")
            )
        )

    @staticmethod
    def fast_evaluator_command(self, version: int) -> str:
        command = ["python", "tools/sandbox.py", "--kind", "run"]
        if self.base_campaign.sandbox_hardware:
            command += ["--hardware", self.base_campaign.sandbox_hardware]
        if self.base_campaign.sandbox_ssh:
            command += ["--ssh", self.base_campaign.sandbox_ssh]
        elif self.base_campaign.sandbox_url:
            command += ["--url", self.base_campaign.sandbox_url]
        elif self.base_campaign.sandbox_profile:
            command += ["--gateway-profile", self.base_campaign.sandbox_profile]
        command += [
            "--no-sync",
            "--",
            "python",
            "test_kernel.py",
            "--version",
            f"v{version}",
            "--no-memory",
        ]
        return shlex.join(command)

    @staticmethod
    def fast_verification_result(self, episode_workspace, *, memory_version):
        from aka.contracts.acceptance import AcceptanceRequirements
        from aka.contracts.evaluation import EvaluationRequest
        from aka.legacy.acceptance import LegacyAcceptancePolicy
        from aka.legacy.evaluation import LegacyEvaluator

        evaluator = LegacyEvaluator()
        expected = self._expected_shape_ids()
        request = EvaluationRequest(
            episode_workspace,
            mode="fast",
            memory_version=memory_version,
            canonical_workspace=self.workspace,
            expected_shape_ids=frozenset(expected) if expected is not None else None,
        )
        result = evaluator.interpret(evaluator.prepare(request), ())
        return (
            LegacyAcceptancePolicy()
            .decide(
                result,
                AcceptanceRequirements(
                    float(self.base_campaign.min_improvement_pct), mode="fast"
                ),
            )
            .verification
        )
