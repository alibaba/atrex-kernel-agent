"""Accept recorded, policy-matched ABBA evidence for Full Episodes."""
from __future__ import annotations

import tempfile
from pathlib import Path

from supervisor.measurements import result_from_response
from supervisor.workspace import read_input, publish
from orchestrator.episode_workspace import PUBLIC_FILES

from .git_episode import git_blob
from .models import VerificationResult
from .verifier import GatewayABBAValidator, score_verification_payload, verification_schedule


class RecordedABBAValidator(GatewayABBAValidator):
    def __init__(self, *, execute, **kwargs):
        super().__init__(**kwargs)
        self.execute = execute

    def verify(self, workspace: Path, *, base_commit: str, candidate_commit: str,
               changed_paths: list[str]) -> VerificationResult:
        try:
            if changed_paths != ["kernel.py"]:
                raise ValueError("Recorded acceptance requires a kernel-only candidate")
            candidate = git_blob(workspace, candidate_commit, "kernel.py")
            baseline = git_blob(workspace, base_commit, "kernel.py")
            if read_input(workspace, "kernel.py") != candidate:
                raise ValueError("Candidate source changed after its Supervisor commit")
            with tempfile.TemporaryDirectory(prefix="aka-acceptance-") as temporary:
                staged = Path(temporary).resolve()
                for name in PUBLIC_FILES:
                    try:
                        content = read_input(workspace, name)
                    except FileNotFoundError:
                        continue
                    publish(staged, name, content)
                publish(staged, "kernel.py", candidate)
                publish(staged, "scratch/incumbent.py", baseline)
                response = self.execute(staged, [
                    "--kind", "run", "--mode", "full", "--no-sync",
                    "--baseline-path", "scratch/incumbent.py",
                    "--comparison-repeats", str(self.repeats),
                    "--comparison-run-timeout", str(self.per_run_timeout),
                    "--shape-batch-size", str(self.shape_batch_size),
                ])
            result = result_from_response(response, "same_allocation_abba")
            if not result or not result.get("gateway_record_id"):
                raise ValueError("No authoritative ABBA record: " + response.get("stderr", "")[-1000:])
            if result.get("comparison") != {"method": "abba", "repeats": self.repeats}:
                raise ValueError("ABBA record does not match the acceptance schedule")
            # Gateway validates all schedule rows and Shape coverage. Score its
            # aggregate sides using the existing objective/threshold, not Evaluate.
            schedule = verification_schedule(1)
            rows = []
            for step, side in zip(schedule, ("baseline", "candidate"), strict=True):
                metrics = result[side]
                rows.append(dict(step, exit_code=0, result=dict(
                    metrics, all_pass=result.get("correct") is True and metrics.get("correct") is True)))
            scored = score_verification_payload(
                {"schema_version": 1, "runs": rows}, schedule=schedule, repeats=1,
                min_improvement_pct=self.min_improvement_pct, artifact=result["gateway_record_id"],
            )
            scored.gateway_record_id = result["gateway_record_id"]
            scored.reused = response.get("reused") is True
            return scored
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            return VerificationResult("ERROR", None, None, None, error=f"Recorded ABBA verification failed: {error}")
