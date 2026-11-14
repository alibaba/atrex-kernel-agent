"""Shared experiment correlation identity, without runner or evaluation imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRef:
    run_id: str
    experiment_id: str

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "experiment_id": self.experiment_id}
