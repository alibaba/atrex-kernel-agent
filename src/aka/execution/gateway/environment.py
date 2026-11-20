"""Environment identity from the worker that actually executed an Agate Job."""
from dataclasses import replace
import hashlib
import json

from aka.contracts.execution_environment import EnvironmentSnapshot


def with_job_environment(result, *, support=None):
    """Prefer the Job receipt over discovery of a potentially different worker.

    Hash stable GPU/toolchain facts, never Job IDs, timestamps or instantaneous
    clock readings. Requesting clock locking remains best effort. A receipt
    without sufficient observed identity stays unknown.
    """
    try:
        job = json.loads(result.stdout)
    except (TypeError, ValueError):
        job = None
    environment = None
    if isinstance(job, dict) and job.get("job_id"):
        payload = job.get("result")
        if isinstance(payload, dict):
            environment = payload.get("environment")
        if not isinstance(environment, dict):
            environment = job.get("environment")
    snapshot = result.environment_snapshot
    identity = result.environment_identity
    if isinstance(environment, dict):
        keys = ("accelerator_backend", "gpu_name", "gpu_arch", "driver_version",
                "runtime_version", "torch_version", "python_version", "triton_version",
                "image_digest", "environment_identity", "clock_locked", "clock_lock_source")
        facts = {key: environment[key] for key in keys if environment.get(key) is not None}
        text = lambda key: isinstance(facts.get(key), str) and bool(facts[key].strip())
        identified = text("environment_identity") or (
            all(text(key) for key in ("gpu_name", "gpu_arch", "driver_version"))
            and any(text(key) for key in ("runtime_version", "torch_version")))
        identity = ("agate-job-environment:" + hashlib.sha256(json.dumps(
            facts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()) if identified else None
        previous = support.environment if support is not None else snapshot
        snapshot = EnvironmentSnapshot(
            backend=result.resource.backend if result.resource else "gateway",
            endpoint=previous.endpoint if previous else None,
            hardware=result.hardware or (previous.hardware if previous else ""),
            identity=identity, observed=environment, raw=environment,
            source="agate-job-result", probed_at=job.get("finished_at") or job.get("updated_at"),
        )
    elif snapshot is None and identity is None and support is not None:
        snapshot, identity = support.environment, support.environment.identity
    return replace(result, environment_snapshot=snapshot, environment_identity=identity,
                   environment_support=support if support is not None else result.environment_support)
