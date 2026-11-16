"""Regress the successful L20N receipt that discovery left unqualified."""
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import Mock, patch

import pytest

from aka.contracts.execution import CommandSpec, JobResult
from aka.contracts.execution_environment import EnvironmentSnapshot, EnvironmentSupport
from aka.contracts.execution_recovery import JobReference
from aka.execution.gateway.environment import with_job_environment
from aka.execution.gateway.executor import GatewayExecutor
from aka.execution.gateway.jobs import result_from_job


# Actual result.environment fields from ev_a16d47820512, without timing noise.
ENVIRONMENT = {"gpu_arch": "sm_120", "gpu_name": "NVIDIA RTX PRO 5000 72GB Blackwell",
    "clock_locked": False, "clock_sm_mhz": 2557, "torch_version": "2.9.0+cu130",
    "driver_version": "580.126.09", "python_version": "3.12.13",
    "triton_version": "3.7.1", "runtime_version": "13.0", "clock_sm_max_mhz": 3090,
    "clock_lock_source": None, "accelerator_backend": "cuda", "clock_lock_detected": False,
    "clock_sm_pct_of_max": 82.8, "clock_throttle_reasons": "0x0000000000000000"}


def job(environment=ENVIRONMENT):
    return {"job_id": "ev_test", "status": "succeeded", "lock_clocks": True,
            "result": {"environment": environment}}


def support(identity=None):
    return EnvironmentSupport("unknown", "unknown",
        EnvironmentSnapshot("gateway", None, "L20N", identity), {}, (), "")


def test_actual_receipt_replaces_discovery_for_normal_and_recovery_execution(tmp_path):
    raw = JobResult(0, stdout=json.dumps(job()), transport_status="succeeded", hardware="L20N")
    executor = GatewayExecutor()
    executor._bound_context = True
    executor._environment_backend = Mock(return_value=Mock(check=Mock(return_value=support("stale-env"))))
    executor._job_transport = Mock(return_value=Mock(environment={}, endpoint="", profile=""))
    observed = []
    def run(*args, **kwargs):
        kwargs["on_observation"](raw)
        return raw
    with patch("aka.execution.outcomes.run_owned", side_effect=run):
        result = executor.run(CommandSpec(tmp_path, ("true",), operation="run", on_observation=observed.append))
    recovered = result_from_job(job(), JobReference("gateway", "ev_test"))
    assert result.environment_identity == recovered.environment_identity == observed[0].environment_identity
    assert result.environment_identity.startswith("agate-job-environment:")
    assert result.environment_snapshot.source == "agate-job-result"
    assert result.environment_snapshot.observed["clock_locked"] is False
    assert result.stdout == raw.stdout
    # Reapplying support during replay must not replace this with discovery.
    assert with_job_environment(result, support=support("stale-env")).environment_identity == result.environment_identity


def test_volatile_fields_do_not_split_identical_execution_environments():
    first = with_job_environment(JobResult(0, stdout=json.dumps(job())))
    changed = deepcopy(ENVIRONMENT)
    changed.update(clock_sm_mhz=2000, clock_sm_pct_of_max=60, clock_throttle_reasons="0x1")
    second = job(changed)
    second.update(job_id="ev_other", updated_at="2026-11-18T00:00:00Z")
    assert with_job_environment(JobResult(0, stdout=json.dumps(second))).environment_identity == first.environment_identity


@pytest.mark.parametrize("field", ["gpu_name", "gpu_arch", "driver_version", "runtime_version", "torch_version"])
def test_changed_hardware_or_toolchain_changes_identity(field):
    changed = {**ENVIRONMENT, field: "different"}
    assert with_job_environment(JobResult(0, stdout=json.dumps(job(changed)))).environment_identity != \
        with_job_environment(JobResult(0, stdout=json.dumps(job()))).environment_identity


@pytest.mark.parametrize("environment", [None, {}, {"gpu_name": "L20N"}])
def test_missing_worker_evidence_remains_unknown(environment):
    result = with_job_environment(JobResult(0, stdout=json.dumps(job(environment))), support=support())
    assert result.environment_identity is None


def test_environment_does_not_change_failed_result():
    failed = {**job(), "status": "failed", "error": {"error_class": "candidate", "reason": "compile_failed"}}
    result = result_from_job(failed, JobReference("gateway", "ev_test"))
    assert result.transport_status == "failed"
    assert result.returncode != 0
    assert json.loads(result.stdout)["error"] == failed["error"]
