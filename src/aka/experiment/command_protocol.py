"""Bound each transport frame while streaming messages of workload-defined size."""
import json
import struct

FRAME_BYTES = 64 * 1024
ENDPOINT_ENV = "AKA_EXPERIMENT_COMMAND_ENDPOINT"
TOKEN_ENV = "AKA_EXPERIMENT_COMMAND_TOKEN"


def _exact(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise ConnectionError("Experiment command reply was lost; do not resubmit")
        chunks.extend(chunk)
    return bytes(chunks)


def _frame(stream):
    size = struct.unpack("!I", _exact(stream, 4))[0]
    if size < 1 or size > FRAME_BYTES:
        raise ValueError("invalid Experiment transport frame")
    return _exact(stream, size)


def _write_frame(stream, data):
    if not 0 < len(data) <= FRAME_BYTES:
        raise ValueError("invalid Experiment transport frame")
    for value in (struct.pack("!I", len(data)), data):
        remaining = memoryview(value)
        while remaining:
            written = stream.write(remaining)
            if written is None or written < 1 or written > len(remaining):
                raise ConnectionError("Experiment transport write stopped; do not resubmit")
            remaining = remaining[written:]


def send(stream, channel, data):
    """Send bytes in bounded frames; do not invent an overall workload limit."""
    header = json.dumps({"channel": channel, "bytes": len(data)}, separators=(",", ":")).encode()
    _write_frame(stream, header)
    for offset in range(0, len(data), FRAME_BYTES):
        _write_frame(stream, data[offset:offset + FRAME_BYTES])
    stream.flush()


def send_json(stream, channel, value):
    send(stream, channel, json.dumps(value, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")).encode())


def receive(stream, *, consume=None):
    header = json.loads(_frame(stream))
    if (not isinstance(header, dict) or set(header) != {"channel", "bytes"}
            or not isinstance(header["channel"], str) or type(header["bytes"]) is not int
            or header["bytes"] < 0):
        raise ValueError("invalid Experiment transport message")
    remaining = header["bytes"]
    chunks = bytearray() if consume is None else None
    while remaining:
        data = _frame(stream)
        if len(data) > remaining:
            raise ValueError("Experiment transport message exceeds its declared length")
        remaining -= len(data)
        if consume is None:
            chunks.extend(data)
        else:
            consume(header["channel"], data)
    return header["channel"], bytes(chunks) if chunks is not None else None


def receive_json(stream, expected):
    channel, data = receive(stream)
    if channel != expected:
        raise ValueError("unexpected Experiment transport message")
    return json.loads(data)
