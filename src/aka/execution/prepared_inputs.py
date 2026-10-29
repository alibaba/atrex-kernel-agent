"""Transport an explicit closed execution view over the owned launcher channel."""
import base64
import io
import json
import os
from pathlib import Path
import tarfile

from aka.contracts.execution import ExecutionFile, PreparedExecutionInputs, UnsupportedExecution


def validate_prepared_inputs(command):
    inputs = command.prepared_inputs
    if inputs is None:
        return
    if command.filesystem_access is not None:
        raise UnsupportedExecution("closed prepared inputs cannot enforce a filesystem scope")
    if command.submission is not None:
        raise UnsupportedExecution("typed submissions cannot carry portable prepared inputs")
    if command.operation in {"preflight", "check", "disassemble"}:
        raise UnsupportedExecution("selected operation does not support closed prepared inputs")
    paths = [Path(item.path) for item in (*inputs.files, *inputs.runtime_files, *command.input_files)]
    if any(left != right and left in right.parents for left in paths for right in paths):
        raise UnsupportedExecution("closed execution input files cannot contain one another")


def payload(inputs, explicit_files=()):
    if inputs is None:
        return None
    def values(files):
        return [{"path": item.path, "content": base64.b64encode(item.content).decode("ascii"),
                 "mode": item.mode} for item in files]
    return json.dumps({"schema_version": 1, "files": values((*inputs.files, *explicit_files)),
                       "runtime_files": values(inputs.runtime_files),
                       "private_paths": inputs.private_paths}, separators=(",", ":")).encode("utf-8")


def read(descriptor):
    size = os.fstat(descriptor).st_size
    document = json.loads(os.pread(descriptor, size, 0))
    if (not isinstance(document, dict) or set(document) != {
            "schema_version", "files", "runtime_files", "private_paths"}
            or type(document["schema_version"]) is not int or document["schema_version"] != 1):
        raise ValueError("invalid closed execution input payload")
    def files(values):
        if not isinstance(values, list):
            raise ValueError("closed execution input members must be a list")
        result = []
        for value in values:
            if not isinstance(value, dict) or set(value) != {"path", "content", "mode"}:
                raise ValueError("invalid closed execution input member")
            result.append(ExecutionFile(value["path"], base64.b64decode(value["content"], validate=True), value["mode"]))
        return tuple(result)
    if not isinstance(document["private_paths"], list):
        raise ValueError("private execution input paths must be a list")
    return PreparedExecutionInputs(files(document["files"]), files(document["runtime_files"]),
                                   tuple(document["private_paths"]))


def archive(files, *, max_file_bytes):
    """Package only frozen members, preserving their exact bytes and permissions."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for item in files:
            if max_file_bytes is not None and len(item.content) > max_file_bytes:
                raise ValueError("closed execution input exceeds the selected launcher size limit")
            info = tarfile.TarInfo(Path(item.path).as_posix())
            info.size, info.mode = len(item.content), item.mode
            bundle.addfile(info, io.BytesIO(item.content))
    return base64.b64encode(output.getvalue()).decode("ascii")
