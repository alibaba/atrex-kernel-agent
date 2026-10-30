"""Decode the documented candidate-source format after artifact validation."""
import base64
import json
from pathlib import PurePosixPath

from aka.contracts.experiment import ExperimentRecordError
from aka.contracts.workspace import RevisionContent


def read_source(records, artifact):
    if artifact.role != "candidate-source":
        raise ExperimentRecordError("not_candidate_source", artifact.experiment)
    record = records.get(artifact.experiment)
    formats = {step.measurement_identity.parameters.get("candidate_source_format")
               for step in record.steps if step.measurement_identity is not None
               and step.measurement_identity.parameters.get("candidate_source_artifact_sha256") == artifact.sha256}
    if len(formats) != 1 or None in formats:
        raise ExperimentRecordError("source_binding_mismatch", record.ref)
    content = records.read_artifact(artifact)
    format_name = next(iter(formats))
    if format_name == "raw-kernel/1":
        return RevisionContent({"kernel.py": content})
    if format_name != "source-map/1":
        raise ExperimentRecordError("unsupported_source_format", record.ref)
    try:
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"schema_version", "files"} or type(value["schema_version"]) is not int or value["schema_version"] != 1 or not isinstance(value["files"], dict):
            raise ValueError("invalid source map")
        files = {}
        for name, encoded in value["files"].items():
            if not isinstance(name, str):
                raise ValueError("unsafe source path")
            path = PurePosixPath(name)
            if (not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts
                    or path.as_posix() != name or path == PurePosixPath(".") or "\\" in name or any(ord(char) < 32 for char in name)):
                raise ValueError("unsafe source path")
            files[name] = None if encoded is None else base64.b64decode(encoded, validate=True)
        if not files:
            raise ValueError("empty source map")
        return RevisionContent(files)
    except (ValueError, TypeError, KeyError) as error:
        raise ExperimentRecordError("source_artifact_corrupt", record.ref) from error
