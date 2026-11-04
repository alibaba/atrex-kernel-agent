"""Retrieved file content travels over an owned launcher descriptor."""
import base64
import json
import os
from aka.contracts.execution import ExecutionFile
FILES_FD = "AKA_EXECUTION_FILES_FD"

def emit(files):
    if not files:
        return
    value = os.environ.get(FILES_FD)
    if value is None:
        raise RuntimeError("retrieved output requires an owning execution host")
    for path, content, mode in files:
        data = (json.dumps({"path": path, "content": base64.b64encode(content).decode(), "mode": mode},
                           separators=(",", ":")) + "\n").encode()
        descriptor = int(value)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])

def read(stream):
    stream.seek(0)
    return tuple(ExecutionFile(item["path"], base64.b64decode(item["content"], validate=True), item["mode"])
                 for item in (json.loads(line) for line in stream))
