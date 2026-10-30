"""Read final command receipts and retained outputs through their record owner."""
from dataclasses import replace
import json

from aka.contracts.execution import ExecutionFile
from aka.contracts.experiment import ExperimentRecordError
from .file_records import _job


def completed_result(records, record):
    if record.kind != "command":
        raise LookupError("experiment has no command receipt")
    artifact = next((item for item in reversed(record.artifacts) if item.role == "command_result"), None)
    if artifact is None:
        raise LookupError("command final receipt is not recorded")
    try:
        result = _job(json.loads(records.read_artifact(artifact)))
        if result is None:
            raise ValueError("command receipt is invalid")
        files = []
        for receipt in result.file_receipts:
            content = next((item for item in record.artifacts
                if item.sha256 == receipt.sha256 and item.size_bytes == receipt.size_bytes), None)
            if content is None:
                raise ValueError("command output artifact is unavailable")
            files.append(ExecutionFile(receipt.path, records.read_artifact(content), receipt.mode))
        return replace(result, files=tuple(files))
    except (ValueError, TypeError, KeyError) as error:
        raise ExperimentRecordError("command_result_corrupt", record.ref) from error
