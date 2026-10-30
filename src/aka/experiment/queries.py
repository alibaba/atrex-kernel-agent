"""Qualify saved common measurement facts without execution or a second index."""
from collections.abc import Mapping
import json
import math

from aka.contracts.experiment import ExperimentRecordError, ShapeMinimum


_SOURCE_KEYS = {"candidate_source_sha256", "candidate_source_artifact_sha256", "candidate_source_format", "candidate_source_identity"}


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _document(value):
    if isinstance(value, Mapping):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_document(item) for item in value]
    return value


def _ordinary(identity, *, correctness=False):
    return (identity.method == "atrex-bench/eval" and identity.parameters.get("workload_scope") == "full-task"
            and identity.parameters.get("mode") in ({"full", "correctness_only"} if correctness else {"full"})
            and identity.points and all(point.side == "candidate" and point.shape_id is not None for point in identity.points)
            and identity.dependency_policy != "unknown"
            and _sha256(identity.parameters.get("workload_digest")))


def _performance_key(identity):
    parameters = {key: _document(value) for key, value in identity.parameters.items()
                  if key not in _SOURCE_KEYS | {"correctness_digest", "measurement_repetition"}}
    return json.dumps({"method": identity.method, "parameters": parameters,
        "requirements": list(identity.requirements), "dependency_policy": identity.dependency_policy,
        "points": sorted((point.repeat, point.shape_id) for point in identity.points)},
        sort_keys=True, separators=(",", ":"), allow_nan=False)


def _measurements(record):
    latest = {attempt.task_id: attempt for attempt in record.attempts}
    steps = [step for step in record.steps if step.measurement_identity is not None]
    if not steps:
        return ()
    result = []
    for step in steps:
        attempt = latest.get(step.task_id)
        if attempt is None or attempt.disposition != "terminal" or attempt.result is None or attempt.assessment is None:
            return ()
        assessment = attempt.assessment
        if assessment.identity.digest != step.measurement_identity.digest:
            raise ExperimentRecordError("measurement_identity_mismatch", record.ref)
        if (assessment.completeness != "complete" or assessment.validity != "valid" or assessment.comparability != "comparable"
                or not assessment.environment_identity or attempt.result.facts.disposition != "terminal"):
            return ()
        result.append(assessment)
    return tuple(result)


def _eligible(record):
    return (record.lifecycle == "terminal" and record.phase == "complete" and record.outcome == "completed"
            and record.evaluation is not None and record.evaluation.gate == "PASS" and not record.evaluation.error)


def _source(records, record, identities):
    digests = {identity.parameters.get("candidate_source_artifact_sha256") for identity in identities}
    if len(digests) != 1 or not all(_sha256(digest) for digest in digests):
        return None
    expected = next(iter(digests))
    sources = [artifact for artifact in record.artifacts if artifact.role == "candidate-source" and artifact.sha256 == expected]
    if not sources:
        return None
    artifact = min(sources, key=lambda artifact: artifact.artifact_id)
    records.read_source(artifact)
    return artifact


def shape_minima(records, identities, environment_identity):
    if not isinstance(environment_identity, str) or not environment_identity or not identities or not all(_ordinary(identity) for identity in identities):
        return ()
    expected_keys = sorted(_performance_key(identity) for identity in identities)
    expected_shapes = {point.shape_id for identity in identities for point in identity.points}
    winners = {}
    for record in records.iter_records():
        if not _eligible(record):
            continue
        measurements = _measurements(record)
        observed = tuple(measurement.identity for measurement in measurements)
        if (not observed or not all(_ordinary(identity) for identity in observed)
                or {measurement.environment_identity for measurement in measurements} != {environment_identity}
                or sorted(_performance_key(identity) for identity in observed) != expected_keys):
            continue
        source = _source(records, record, observed)
        if source is None:
            continue
        latencies = record.evaluation.candidate_latency_us_by_shape
        if (not isinstance(latencies, Mapping) or set(latencies) != expected_shapes
                or any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0 for value in latencies.values())):
            continue
        for shape, latency in latencies.items():
            key = (float(latency), source.sha256, record.ref.run_id, record.ref.experiment_id, source.artifact_id)
            if shape not in winners or key < winners[shape][0]:
                winners[shape] = (key, ShapeMinimum(shape, float(latency), source, record.ref))
    return tuple(winners[shape][1] for shape in sorted(winners))


def correctness_reuse(records, identities, environment_identity):
    def key(identity):
        if not _ordinary(identity, correctness=True) or identity.parameters.get("correctness_reuse_policy") != "typed-six-case/1":
            return None
        digest = identity.parameters.get("correctness_digest")
        if not _sha256(digest):
            return None
        return (identity.method, digest, identity.requirements, identity.dependency_policy)
    if not isinstance(environment_identity, str) or not environment_identity or not identities:
        return None
    expected = {key(identity) for identity in identities}
    if None in expected:
        return None
    shapes = {point.shape_id for identity in identities for point in identity.points}
    candidates = []
    for record in records.iter_records():
        if not _eligible(record):
            continue
        measurements = _measurements(record)
        observed = tuple(measurement.identity for measurement in measurements)
        if (not observed or {key(identity) for identity in observed} != expected
                or {measurement.environment_identity for measurement in measurements} != {environment_identity}
                or {point.shape_id for identity in observed for point in identity.points} != shapes):
            continue
        if _source(records, record, observed) is not None:
            candidates.append(record)
    return max(candidates, key=lambda record: (record.created_at, record.ref.experiment_id)) if candidates else None
