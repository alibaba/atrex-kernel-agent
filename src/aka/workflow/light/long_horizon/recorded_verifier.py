"""Accept recorded, policy-matched ABBA evidence for Full Episodes."""
from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
import logging
from aka.contracts.content import WorkspaceFile
from aka.contracts.experiment import ExperimentRecordError

from aka.evaluation.tool_results import result_from_response
from aka.workflow.light.host.workspace import read_input, publish
from aka.workflow.light.orchestration.episode_workspace import PUBLIC_FILES
from aka.workflow.light.orchestration.infrastructure_retry import InfrastructureUnavailable
from aka.workflow.light.support.errors import GatewayExecutionUnavailable, RequestDispatchTimeout

from .models import VerificationResult
from .verifier import GatewayABBAValidator, score_verification_payload, verification_schedule


class RecordedABBAValidator(GatewayABBAValidator):
    BASELINE_PATH = "scratch/incumbent.py"

    def __init__(self, *, execute, workspace_provider, **kwargs):
        super().__init__(**kwargs)
        self.execute = execute
        self.workspace_provider = workspace_provider

    def request_argv(self) -> list[str]:
        """One canonical request for trusted acceptance and the Agent prompt."""
        return [
            "--kind", "run", "--mode", "full", "--no-sync",
            "--baseline-path", self.BASELINE_PATH,
            "--comparison-repeats", str(self.repeats),
            "--comparison-run-timeout", str(self.per_run_timeout),
            "--shape-batch-size", str(self.shape_batch_size),
            "--multi-seed", "0",
        ]

    def agent_instructions(self, *, baseline_kernel_id: str, measurement_repetitions: int) -> str:
        read = shlex.join([
            "python3", "tools/sandbox.py", "--kind", "kernel-read", "--kernel-id",
            baseline_kernel_id, "--output-path", self.BASELINE_PATH,
        ])
        measure = shlex.join(["python3", "tools/sandbox.py", *self.request_argv()])
        return f"""## This Episode's acceptance request

ABBA is optional during exploration. To make a comparison eligible for acceptance reuse,
leave the complete candidate in place and run these exact commands
(for a source-tree task, `kernel-read` creates a baseline directory, not a Python file):

```bash
{read}
{measure}
```

The first command copies this Episode's committed incumbent, not the current draft; it is
safe to use after edits or resume. Do not modify `{self.BASELINE_PATH}` before the comparison.
The second command matches the Supervisor's configured ABBA repeats, per-run timeout and
Shape batch size. Retention ABBA uses one correctness case, matching Runtime;
the separate candidate correctness gate covers six cases. Do not add
command/input/Shape/seed/benchmark/dependency overrides.
The Runtime supplies the evaluator, full workload and target; private inputs stay private.

Supervisor measurement repetitions: **{measurement_repetitions}** per request (one result if 1;
per-Shape median if 3).
These are Runtime-owned settings, already applied to this request; do not loop the command
or set environment variables to reproduce them.

Reuse requires the same candidate, incumbent, inputs, evaluator and complete task policy.
A duplicate-task error names an existing Record: use `record-read`, not another submission.
Acceptance reuses a completed matching Record or measures when none exists; it is not a
zero-submission guarantee. ABBA does not replace the standard full Evaluate required by
`episode-report`, and the Supervisor still applies policy and performance gates.
"""

    def verify(self, workspace: Path, *, base_commit: str, candidate_commit: str,
               changed_paths: list[str]) -> VerificationResult:
        try:
            owner = self.workspace_provider
            candidate = owner.read_source(workspace, revision=candidate_commit)
            baseline = owner.read_source(workspace, revision=base_commit)
            single_file = set(baseline.files) == {"kernel.py"}
            if (changed_paths != ["kernel.py"] if single_file else
                    not set(changed_paths) <= candidate.files.keys() | baseline.files.keys()):
                raise ValueError("Recorded acceptance requires a kernel-only candidate")
            try:
                mismatch = owner.read_source(workspace).files != candidate.files
            except (OSError, ValueError, RuntimeError):
                mismatch = True
            if mismatch:
                logging.getLogger(__name__).warning("Candidate working source differs from sealed revision; acceptance uses committed bytes")
            with tempfile.TemporaryDirectory(prefix="aka-acceptance-") as temporary:
                staged = Path(temporary).resolve()
                for name in PUBLIC_FILES:
                    try:
                        content = read_input(workspace, name)
                    except FileNotFoundError:
                        continue
                    publish(staged, name, content)
                owner.install_source(staged, candidate)
                if not single_file:
                    owner.install_source(staged / self.BASELINE_PATH, baseline)
                else:
                    owner.install_files(staged, (WorkspaceFile(self.BASELINE_PATH, text=baseline.files["kernel.py"].decode("utf-8")),))
                response = self.execute(staged, self.request_argv())
            result = result_from_response(response, "same_allocation_abba")
            if not result or not result.get("gateway_record_id"):
                raise ValueError("No authoritative ABBA record: " + response.get("stderr", "")[-1000:])
            if result.get("comparison") != {"method": "abba", "repeats": self.repeats}:
                raise ValueError("ABBA record does not match the acceptance schedule")
            if result.get("complete") is not True or result.get("comparison_run_timeout") != self.per_run_timeout:
                raise ValueError("ABBA record has incomplete evidence or a different per-run budget")
            if type(result.get("correct")) is not bool:
                raise ValueError("ABBA correctness is unknown")
            # Gateway validates every schedule row and Shape. Promotion uses
            # latency reduction, never the reference-relative performance score.
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
        except InfrastructureUnavailable:
            raise
        except (GatewayExecutionUnavailable, RequestDispatchTimeout) as error:
            raise InfrastructureUnavailable(str(error)) from error
        except ExperimentRecordError as error:
            raise InfrastructureUnavailable(str(error)) from error
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            return VerificationResult("ERROR", None, None, None, error=f"Recorded ABBA verification failed: {error}")
