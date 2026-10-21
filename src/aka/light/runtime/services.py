"""Explicit borrowed dependencies of the HTTP tool host."""
from dataclasses import dataclass
from aka.light.contracts import Acceptance, ExperimentFactory, JournalFactory


@dataclass(frozen=True)
class RuntimeServices:
    executor: object
    experiments: ExperimentFactory
    acceptance: Acceptance
    journal: JournalFactory


def builtin_services() -> RuntimeServices:
    """For direct embedding; composed runs always inject their selected services."""
    from aka.light.execution.provider import ReliableExecutor
    from aka.light.evaluation.provider import GatewayEvaluation
    from aka.light.experiment.provider import RecordedExperimentFactory
    from aka.light.acceptance.provider import RecordedAcceptance
    from aka.light.workflow.journal import JournalFactory
    return RuntimeServices(ReliableExecutor(), RecordedExperimentFactory(GatewayEvaluation()),
                           RecordedAcceptance(), JournalFactory())
