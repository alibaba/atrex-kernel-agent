from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .optimize import Campaign


class StopDecisionStatus(str, Enum):
    CONTINUE = "CONTINUE"
    SUCCESS = "SUCCESS"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True)
class StopDecision:
    status: StopDecisionStatus
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, StopDecisionStatus):
            raise TypeError("status must be a StopDecisionStatus")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if self.status in {StopDecisionStatus.SUCCESS, StopDecisionStatus.INFRA_ERROR} and not self.reason:
            raise ValueError("a terminal or infrastructure decision requires a reason")

    @classmethod
    def continue_(cls) -> "StopDecision":
        return cls(StopDecisionStatus.CONTINUE)


class StopPolicy(Protocol):
    def evaluate_accepted_iteration(
        self,
        campaign: "Campaign",
        version: int,
        memory: dict,
    ) -> StopDecision:
        ...


class DefaultStopPolicy:
    """Preserve the orchestrator's historical peak-utilization stop condition."""

    def evaluate_accepted_iteration(
        self,
        campaign: "Campaign",
        version: int,
        memory: dict,
    ) -> StopDecision:
        del version
        performance = memory.get("performance") or {}
        values = [
            performance.get("tflops_peak_utilization_pct"),
            performance.get("bandwidth_peak_utilization_pct"),
        ]
        peak = max(
            [float(value) for value in values if isinstance(value, (int, float))]
            or [0.0]
        )
        if peak < campaign.target_util:
            return StopDecision.continue_()
        return StopDecision(
            StopDecisionStatus.SUCCESS,
            f"success: peak_util {peak:.1f}% >= {campaign.target_util:.0f}%",
        )
