from __future__ import annotations

import json
import shlex
from pathlib import Path


from .models import VerificationResult, VerificationRun
from .results import (
    _latest_complete_canonical_performance,
    _latest_complete_episode_performance,
    EPISODE_EVALUATIONS_PATH,
)


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
    def fast_verification_result(
        self, episode_workspace: Path, *, memory_version: int
    ) -> VerificationResult:
        """Score the final recorded evaluator result without launching ABBA.

        ``tools/sandbox.py`` fingerprints ``kernel.py`` and ``solution.json`` in
        every episode result. The reader below selects a complete passing trial result whose fingerprint matches
        the final selected candidate, then compares that measurement with canonical
        incumbent memory.  This deliberately trades statistical rigor for turnaround.
        """
        artifact = str(episode_workspace / EPISODE_EVALUATIONS_PATH)
        expected_shape_ids = self._expected_shape_ids()
        candidate_result = _latest_complete_episode_performance(
            episode_workspace,
            expected_shape_ids=expected_shape_ids,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if candidate_result is None:
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                error=(
                    "fast mode requires one complete passing evaluator result for the "
                    "final kernel.py and solution.json pair"
                ),
                artifact=artifact,
            )
        candidate_latency = float(candidate_result["latency_us_geomean"])
        candidate_score = float(candidate_result["performance_score"])
        run = VerificationRun(
            revision="candidate",
            repeat=0,
            exit_code=0,
            result=candidate_result,
        )
        incumbent = _latest_complete_canonical_performance(
            self.workspace,
            before_version=memory_version,
            expected_shape_ids=expected_shape_ids,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if incumbent is None:
            return VerificationResult(
                "FAIL",
                candidate_latency,
                None,
                None,
                runs=[run],
                error="fast mode could not find complete canonical incumbent performance",
                artifact=artifact,
            )
        incumbent_performance, _incumbent_version = incumbent
        incumbent_latency = float(incumbent_performance["latency_us_geomean"])
        incumbent_score = float(incumbent_performance["performance_score"])
        improvement = (candidate_score / incumbent_score - 1.0) * 100.0
        threshold = float(self.base_campaign.min_improvement_pct)
        if improvement <= threshold:
            return VerificationResult(
                "FAIL",
                candidate_latency,
                incumbent_latency,
                improvement,
                runs=[run],
                error=(
                    "fast evaluator performance-score improvement "
                    f"{improvement:.6f}% did not exceed {threshold:.3f}%"
                ),
                artifact=artifact,
                candidate_performance_score=candidate_score,
                incumbent_performance_score=incumbent_score,
            )
        return VerificationResult(
            "PASS",
            candidate_latency,
            incumbent_latency,
            improvement,
            runs=[run],
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
        )
