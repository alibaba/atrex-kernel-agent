"""Encode operation-owned content using the common opaque Agent framing."""
from io import BytesIO
import json

from aka.agent.tools.protocol import receive, send, send_json
from aka.contracts.execution import ExecutionFile
from .file_records import _job


def _message(stream):
    channel, data = receive(stream)
    if channel != "message":
        raise ValueError("unexpected command message")
    return json.loads(data)


def encode_request(message, inputs=()):
    stream = BytesIO()
    send_json(stream, "message", message)
    for item in inputs:
        send(stream, "input", item.content)
    return stream.getvalue()


def decode_request(payload):
    stream = BytesIO(payload)
    message = _message(stream)
    if not isinstance(message, dict):
        raise ValueError("command tool request must be an object")
    inputs = []
    command = message.get("command", {})
    if not isinstance(command, dict):
        raise ValueError("invalid command inputs")
    for item in command.get("input_files", ()):
        if not isinstance(item, dict) or set(item) != {"path", "mode"}:
            raise ValueError("invalid command input descriptor")
        channel, content = receive(stream)
        if channel != "input":
            raise ValueError("invalid command input content")
        inputs.append(ExecutionFile(item["path"], content, item["mode"]))
    if stream.read(1):
        raise ValueError("unexpected command request content")
    return message, tuple(inputs)


def encode_response(message, result=None):
    stream = BytesIO()
    data = result.as_dict() if result is not None else None
    if data is not None:
        data["stdout"] = data["stderr"] = ""
    files = result.files if result is not None else ()
    send_json(stream, "message", {**message, "result": data,
        "files": [{"path": item.path, "mode": item.mode} for item in files]})
    send(stream, "stdout", result.stdout.encode() if result is not None else b"")
    send(stream, "stderr", result.stderr.encode() if result is not None else b"")
    for item in files:
        send(stream, "output", item.content)
    return stream.getvalue()


def decode_response(payload):
    stream = BytesIO(payload)
    message = _message(stream)
    result = _job(message["result"])
    texts = []
    for expected in ("stdout", "stderr"):
        channel, content = receive(stream)
        if channel != expected:
            raise ValueError("unexpected command output text")
        texts.append(content.decode())
    outputs = []
    for item in message["files"]:
        channel, content = receive(stream)
        if channel != "output":
            raise ValueError("unexpected command output content")
        outputs.append(ExecutionFile(item["path"], content, item["mode"]))
    if stream.read(1):
        raise ValueError("unexpected command reply content")
    return message, result, texts[0], texts[1], tuple(outputs)
