"""Own Legacy command evaluation and storage for one borrowed operation."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from aka.contracts.execution import validate_executor
from .file_records import FileExperimentRecords
from .runner import LegacyExperiment, evaluation_stage


class LegacyExperimentFactory:
    def __init__(self, *, measurement_reader=None, records_root=None, command_evaluator=None, retry_policy=None):
        from aka.contracts.execution_recovery import InfrastructureRetryPolicy
        self.retry_policy = retry_policy if retry_policy is not None else InfrastructureRetryPolicy()
        self.measurement_reader = measurement_reader
        self.records_root = Path(records_root).expanduser().resolve() if records_root is not None else None
        if command_evaluator is not None and any(not callable(getattr(command_evaluator, method, None))
                for method in ("prepare", "observe_step", "fallback", "interpret")):
            raise TypeError("provider does not implement CommandEvaluator")
        self.command_evaluator = command_evaluator
        self._closed = False

    def close(self):
        self._closed = True

    @contextmanager
    def open(self, *, workspace, purpose, executor, evaluator=None, evaluation_input=None, episode_evidence=None,
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
        records = FileExperimentRecords(self.records_root or workspace.parent / ".aka-experiments", str(workspace))
        yield LegacyExperiment(executor, evaluator, records=records, purpose=purpose,
            command_evaluator=selected_command if selected_command is not None else
                GatewayCommandEvaluator(measurement_reader=self.measurement_reader,
                    execution_capabilities=executor.capabilities, evaluation_input=evaluation_input),
            episode_evidence=episode_evidence,
            retry=partial(retry_infrastructure, policy=self.retry_policy), stage=evaluation_stage)
