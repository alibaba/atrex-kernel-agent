"""Light's report gate consumes public measurement facts, without owning evaluation."""
from dataclasses import dataclass
from collections.abc import Callable

from aka.contracts.workspace import RevisionContent
from aka.light.supervisor.errors import AgentRequestError, RuntimeStateError
from .measurement_policy import CORRECTNESS_CASES


def candidate_acceptance_argv(*, atrex_bench):
    if atrex_bench:
        return ["--kind", "run", "--mode", "correctness_only", "--multi-seed", str(CORRECTNESS_CASES - 1), "--no-sync"]
    return ["--kind", "run", "--no-sync"]


@dataclass(frozen=True)
class CandidateValidation:
    atrex_bench: bool
    production: bool
    measure: Callable
    read_record: Callable
    ensure_live: Callable
    source_identity: Callable
    supplemental: Callable | None = None
    research_records: bool = False


class CandidateGate:
    def verifier(self, **options):
        from aka.light.long_horizon.recorded_verifier import RecordedABBAValidator
        return RecordedABBAValidator(**options)

    def validate_candidate(self, context, source):
        if not isinstance(source, RevisionContent):
            raise TypeError("Report acceptance requires public frozen source")
        response = context.measure(candidate_acceptance_argv(atrex_bench=context.atrex_bench), source, context.atrex_bench)
        context.ensure_live()
        record_id = (response.get("agent_result", {}).get("identity") or {}).get("gateway_record_id")
        if not record_id:
            raise RuntimeStateError("Candidate measurement did not return retained evidence", code="candidate_validation_unavailable")
        try:
            evidence = context.read_record(record_id)
        except (LookupError, OSError, ValueError) as error:
            raise RuntimeStateError("Candidate measurement evidence is unavailable", code="candidate_validation_unavailable") from error
        authoritative = (evidence.get("source_sha256") == context.source_identity(source)
            and evidence.get("execution_status") == "completed"
            and (evidence.get("typed_six_case_correctness") if context.atrex_bench else evidence.get("standard_full_evaluation")))
        if not authoritative:
            raise RuntimeStateError("Candidate measurement has insufficient identity or coverage", code="candidate_validation_unavailable")
        result = evidence.get("result", {})
        if result.get("all_pass") is False:
            raise AgentRequestError("The candidate failed the selected correctness gate; the report was not accepted and no candidate commit was created.",
                code="candidate_correctness_failed", gateway_record_id=record_id,
                next_action=f"Inspect record-read --record-id {record_id}, repair the candidate and submit a passing full evaluation with the report.")
        if result.get("all_pass") is not True or result.get("error"):
            raise RuntimeStateError("Candidate measurement is inconsistent", code="candidate_validation_unavailable")
        retained = {"gateway_record_id": record_id, "reused": response.get("reused") is True,
                    "evaluator": "atrex_bench" if context.atrex_bench else "sol_execbench"}
        if context.atrex_bench:
            retained.update(additional_seeds=CORRECTNESS_CASES - 1, correctness_cases=CORRECTNESS_CASES)
        if context.production:
            if context.supplemental is None:
                raise RuntimeStateError("Production numerical validator is not configured")
            feedback = context.supplemental(source)
            context.ensure_live()
            if feedback:
                from aka.light.orchestrator.constants import SUPPLEMENTAL_REPAIR_PREFIX
                if feedback.startswith(SUPPLEMENTAL_REPAIR_PREFIX):
                    raise AgentRequestError("The candidate failed a measured supplemental probe; report not accepted.", code="candidate_numerical_failed", next_action=feedback)
                raise RuntimeStateError("Supplemental numerical validation is incomplete; report not accepted.", code="candidate_numerical_pending", next_action=feedback)
        return retained
