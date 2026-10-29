"""Freeze public command handoff inputs without owning execution policy."""
from dataclasses import asdict
import hashlib


def handoff_identity(command):
    """Omit absent additions so unchanged historical commands retain identity."""
    result = {}
    access = command.filesystem_access
    if access is not None:
        result["filesystem_access"] = {
            "mounts": [{"source": str(item.source), "target": str(item.target),
                        "writable": item.writable} for item in access.mounts],
            "hidden_paths": [str(path) for path in access.hidden_paths],
            "read_only_paths": [str(path) for path in access.read_only_paths],
        }
    if command.profile_options is not None:
        value = asdict(command.profile_options)
        value["counters"] = list(value["counters"])
        result["profile_options"] = value
    if command.output_options is not None:
        result["output_options"] = asdict(command.output_options)
    if command.input_files:
        result["input_files"] = [
            {"path": item.path, "mode": item.mode, "size_bytes": len(item.content),
             "sha256": hashlib.sha256(item.content).hexdigest()}
            for item in command.input_files
        ]
    if command.prepared_inputs is not None:
        result["prepared_inputs"] = {
            name: [{"path": item.path, "mode": item.mode, "size_bytes": len(item.content),
                    "sha256": hashlib.sha256(item.content).hexdigest()} for item in getattr(command.prepared_inputs, name)]
            for name in ("files", "runtime_files")
        }
        result["prepared_inputs"]["private_paths"] = list(command.prepared_inputs.private_paths)
    return result
