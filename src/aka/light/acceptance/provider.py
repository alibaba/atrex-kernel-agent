"""Acceptance interprets evidence; it neither authorizes HTTP nor seals Git."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import json

from aka.light.orchestrator.constants import SOL_HARNESS
from aka.light.supervisor.workspace import read_input
from aka.light.evaluation.policy import CORRECTNESS_CASES
from aka.light.supervisor.errors import AgentRequestError, RuntimeStateError

SOL_ACCEPTANCE_INPUTS = frozenset({
    "reference.py",
    "definition.json",
    "workload.jsonl",
    "solution.json",
    "test_kernel.py",
})


def candidate_acceptance_argv(*, atrex_bench: bool) -> list[str]:
    """Return the evaluator-specific, Supervisor-owned acceptance request.

    Native Atrex-Bench has a typed five-case correctness mode. SOL-ExecBench
    instead owns correctness through its full workload evaluator; typed controls
    cannot represent that contract and would prevent the required Dev fallback.
    """
    if atrex_bench:
        return [
            "--kind", "run", "--mode", "correctness_only",
            "--multi-seed", str(CORRECTNESS_CASES - 1), "--no-sync",
        ]
    return ["--kind", "run", "--no-sync"]


def authoritative_candidate_record(record: dict, source: bytes, *, atrex_bench: bool) -> bool:
    """Validate evaluator identity without trusting the projected result alone."""
    request = record.get("request")
    if not isinstance(request, dict):
        return False
    inputs = request.get("inputs")
    options = request.get("options")
    if not isinstance(inputs, dict) or not isinstance(options, dict):
        return False
    common = (
        record.get("cacheable") is True
        and record.get("operation") == "evaluate"
        and (options.get("source_snapshots", {}).get("candidate", inputs.get("kernel.py"))
             == hashlib.sha256(source).hexdigest())
    )
    if not common:
        return False
    if atrex_bench:
        policy = options.get("evaluation_policy")
        return (
            isinstance(policy, dict)
            and policy.get("mode") in {"full", "correctness_only"}
            and policy.get("num_correctness_cases") == CORRECTNESS_CASES
        )
    return (
        SOL_ACCEPTANCE_INPUTS <= inputs.keys()
        and inputs.get("test_kernel.py") == hashlib.sha256(
            read_input(SOL_HARNESS.parent, SOL_HARNESS.name)
        ).hexdigest()
        and options.get("command") == ["python3", "test_kernel.py"]
        and not options.get("evaluation_input_path")
        and not options.get("evaluation_shapes_path")
    )



@dataclass(frozen=True)
class CandidateValidation:
    atrex_bench: bool
    production: bool
    measure: Callable[[list[str], bytes, bool], dict]
    read_record: Callable[[str], dict]
    ensure_live: Callable[[], None]
    supplemental: Callable[[bytes], str] | None = None
    research_records: bool = False


class RecordedAcceptance:
    def verifier(self, **options):
        from aka.light.long_horizon.recorded_verifier import RecordedABBAValidator
        return RecordedABBAValidator(**options)

    def validate_candidate(self, context: CandidateValidation, source: bytes) -> dict:
        """Reuse or execute the evaluator-specific mandatory correctness gate.

        Atrex-Bench uses its typed five-case gate. SOL-ExecBench executes the
        official evaluator over all workload.jsonl records with their own
        tolerances. Run under the caller's Session/deadline, never a capability
        surviving its revocation.
        """
        from aka.light.evaluation.provider import result_from_response

        atrex_bench = context.atrex_bench
        response = context.measure(candidate_acceptance_argv(atrex_bench=atrex_bench),
                                   source, atrex_bench)
        context.ensure_live()
        from aka.light.supervisor.projection import source_error_from_stdout
        source_error = source_error_from_stdout(response["stdout"])
        if source_error is not None:
            error = source_error["error"]
            raise AgentRequestError(
                error["message"] + " The report was not accepted and no candidate commit was created.",
                code=error["code"], next_action=error["next_action"],
                error_class=error["error_class"], reason=error["reason"],
                job_submitted=False, violations=error["violations"],
            )
        try:
            result = result_from_response(response, "evaluate")
            identity = result or next((json.loads(line.split("=", 1)[1])
                for line in reversed(response["stdout"].splitlines())
                if line.startswith("[sandbox] RECORD_JSON=")), {})
            record_id = identity["gateway_record_id"]
            record = context.read_record(record_id)
            authoritative = authoritative_candidate_record(
                record, source, atrex_bench=atrex_bench,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeStateError(
                "Supervisor candidate correctness evidence is unavailable; report was not accepted",
                code="candidate_validation_unavailable",
            ) from error
        if not authoritative:
            raise RuntimeStateError(
                "Supervisor candidate correctness did not produce authoritative evidence; report was not accepted",
                code="candidate_validation_unavailable",
            )
        # Some confirmed candidate failures (e.g. compile errors) have no Eval
        # payload, only a cacheable terminal Record. Unknown/infra outcomes are
        # never cacheable and cannot reach this repairable rejection path.
        if (result is None and response["exit_code"] != 0) or (result and result.get("all_pass") is False):
            gate_name = (
                "multi-seed correctness (base case plus four additional seeds)"
                if atrex_bench else
                "SOL-ExecBench full-workload correctness"
            )
            raise AgentRequestError(
                f"The candidate failed Supervisor {gate_name}. "
                "The report was not accepted and no candidate commit was created.",
                code="candidate_correctness_failed", gateway_record_id=record_id,
                next_action=(
                    f"Inspect python3 tools/sandbox.py --kind record-read --record-id {record_id}. "
                    + ("Repair kernel.py, record a passing full Evaluate and its Experiment, then resubmit "
                       "episode-report with the matching selected_experiment_id. "
                       if context.research_records else
                       "Repair kernel.py, obtain a passing full Evaluate, then resubmit episode-report "
                       "with its gateway_record_ids. ") +
                    "The Supervisor runs its "
                    "acceptance check automatically; do not repeat the failed GPU request unchanged. "
                    "If abandoning the candidate, submit an evidence-backed pivot instead."
                ),
            )
        result_complete = (
            result
            and result.get("all_pass") is True
            and response["exit_code"] == 0
            and not result.get("error")
        )
        if not atrex_bench:
            total = result.get("correctness_total") if result else None
            result_complete = (
                result_complete
                and type(total) is int
                and total > 0
                and result.get("correctness_passed") == total
            )
        if not result_complete:
            raise RuntimeStateError(
                "Supervisor candidate correctness result is inconsistent; report was not accepted"
            )
        evidence = {"gateway_record_id": record_id, "reused": response.get("reused") is True}
        if atrex_bench:
            evidence.update(additional_seeds=CORRECTNESS_CASES - 1, correctness_cases=CORRECTNESS_CASES, evaluator="atrex_bench")
        else:
            evidence.update(additional_seeds=0, workloads=total, evaluator="sol_execbench")
        validator = context.supplemental
        if context.production:
            if validator is None:
                raise RuntimeStateError("Production numerical validator is not configured")
            feedback = validator(source)
            context.ensure_live()
            if feedback:
                from aka.light.orchestrator.constants import SUPPLEMENTAL_REPAIR_PREFIX
                if feedback.startswith(SUPPLEMENTAL_REPAIR_PREFIX):
                    raise AgentRequestError(
                        "The candidate failed a measured supplemental numerical probe; report not accepted.",
                        code="candidate_numerical_failed", next_action=feedback,
                    )
                raise RuntimeStateError(
                    "Supplemental numerical validation is incomplete; report not accepted.",
                    code="candidate_numerical_pending", next_action=feedback,
                )
        return evidence
