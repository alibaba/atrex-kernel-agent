"""Compatibility verifier exposing domain helpers and Experiment execution."""

from __future__ import annotations
import subprocess
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from threading import Event

from orchestrator._bootstrap import task_modules

task_modules()
from orchestrator.infrastructure_retry import (
    InfrastructureUnavailable,
    check_transport,
    retry_infrastructure,
)
from aka.contracts.acceptance import AcceptanceRequirements
from aka.contracts.evaluation import (
    EvaluationRequest,
    VerificationResult,
    VerificationRun,
)
from aka.contracts.execution import JobResult
from aka.legacy.acceptance import LegacyAcceptancePolicy
from aka.legacy.evaluation import EvaluationConfiguration, LegacyEvaluator
from aka.legacy.evaluation.abba import (
    ABBA_RESULT_PREFIX,
    DEFAULT_SHAPE_BATCH_SIZE,
    DEFAULT_SHAPE_BATCH_WORKERS,
    _geomean,
    _merge_batch_results,
    _merge_batch_payloads,
    _payload_from_stdout,
    _result_latency,
    _result_performance_score,
    _verification_shape_batches,
    interpret_verification_payload,
    verification_schedule,
)
from . import main_adapter


def score_verification_payload(
    payload, *, schedule, repeats, min_improvement_pct, artifact=""
):
    result = interpret_verification_payload(
        payload, schedule=schedule, repeats=repeats, artifact=artifact
    )
    return (
        LegacyAcceptancePolicy()
        .decide(result, AcceptanceRequirements(min_improvement_pct))
        .verification
    )


def _snapshot_revision(
    workspace, revision, changed_paths, destination, label, *, provider=None
):
    from aka.legacy.application.workspace import candidate_workspace

    with candidate_workspace(provider=provider) as provider:
        return provider.write_revision(
            workspace, revision, changed_paths, destination, label
        )


from aka.legacy.compat.experiment import GatewayABBAValidator
