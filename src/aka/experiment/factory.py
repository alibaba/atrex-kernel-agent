"""Compose public execution and evaluation with explicitly selected storage."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from aka.contracts.execution import validate_executor
from .file_records import FileExperimentRecords
from .runner import Experiment


class ExperimentFactory:
    def __init__(self, *, measurement_reader=None, records_root=None, command_evaluator=None, retry_policy=None,
                 measurement_store=None):
        from aka.contracts.execution_recovery import InfrastructureRetryPolicy
        self.retry_policy = retry_policy if retry_policy is not None else InfrastructureRetryPolicy()
        self.measurement_reader = measurement_reader
        self.records_root = Path(records_root).expanduser().resolve() if records_root is not None else None
        if measurement_store is not None and records_root is not None:
            raise ValueError("Choose a Campaign Measurement Store or standalone records_root, not both")
        self.measurement_store = measurement_store
        if command_evaluator is not None and any(not callable(getattr(command_evaluator, method, None))
                for method in ("prepare", "observe_step", "fallback", "interpret")):
            raise TypeError("provider does not implement CommandEvaluator")
        self.command_evaluator = command_evaluator
        self._closed = False

    def for_measurement_store(self, store):
        """Bind a Campaign without changing the selected execution/evaluation policy."""
        from .measurement_store import MeasurementStore

        if not isinstance(store, MeasurementStore):
            raise TypeError("Campaign storage requires MeasurementStore")
        return type(self)(measurement_reader=self.measurement_reader,
            command_evaluator=self.command_evaluator, retry_policy=self.retry_policy,
            measurement_store=store)

    def for_campaign(self, measurement_root: Path):
        from .measurement_store import MeasurementStore

        return self.for_measurement_store(MeasurementStore(measurement_root))

    def close(self):
        self._closed = True

    @contextmanager
    def open(self, *, workspace, purpose, executor, evaluator=None, evaluation_input=None,
             command_evaluator=None):
        if self._closed:
            raise RuntimeError("Experiment selection has been closed")
        if purpose not in {"development", "formal"}:
            raise ValueError("unsupported experiment purpose")
        validate_executor(executor)
        selected_command = command_evaluator if command_evaluator is not None else self.command_evaluator
        if evaluator is None:
            if purpose == "formal":
                raise TypeError("a formal scope requires an injected Evaluator")
            if selected_command is None:
                raise TypeError("a command-only development scope requires an injected CommandEvaluator")
        if evaluator is not None and any(not callable(getattr(evaluator, method, None))
               for method in ("prepare", "observe_step", "interpret")):
            raise TypeError("provider does not implement Evaluator")
        if selected_command is not None and any(not callable(getattr(selected_command, method, None))
                for method in ("prepare", "observe_step", "fallback", "interpret")):
            raise TypeError("provider does not implement CommandEvaluator")
        from aka.evaluation.gateway_command import GatewayCommandEvaluator
        from .retry import retry_infrastructure

        workspace = Path(workspace).resolve()
        # The workspace identifies the retained record namespace. Episode evidence
        # carries the separate actual operation workspace without moving records.
        records = (self.measurement_store.execution_records() if self.measurement_store is not None else
                   FileExperimentRecords(self.records_root or workspace.parent / ".aka-experiments", str(workspace)))
        yield Experiment(executor, evaluator, records=records, purpose=purpose,
            submission_retry_policy=self.retry_policy,
            command_evaluator=selected_command if selected_command is not None else
                GatewayCommandEvaluator(measurement_reader=self.measurement_reader,
                    execution_capabilities=executor.capabilities, evaluation_input=evaluation_input),
            retry=partial(retry_infrastructure, policy=self.retry_policy,
                state_root=self.measurement_store.root / "recovery" if self.measurement_store is not None else None))
