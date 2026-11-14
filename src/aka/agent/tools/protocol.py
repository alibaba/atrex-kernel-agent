"""Bound transport frames while retaining workload-defined opaque message sizes."""
import json
import struct

FRAME_BYTES = 64 * 1024
ENDPOINT_ENV = "AKA_AGENT_TOOL_ENDPOINT"
TOKEN_ENV = "AKA_AGENT_TOOL_TOKEN"


def _exact(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise ConnectionError("Agent tool reply was lost; query original records before resubmitting")
        chunks.extend(chunk)
    return bytes(chunks)


def _frame(stream):
    size = struct.unpack("!I", _exact(stream, 4))[0]
    if not 0 < size <= FRAME_BYTES:
        raise ValueError("invalid Agent tool frame")
    return _exact(stream, size)


def _write_frame(stream, data):
    if not 0 < len(data) <= FRAME_BYTES:
        raise ValueError("invalid Agent tool frame")
    for value in (struct.pack("!I", len(data)), data):
        remaining = memoryview(value)
        while remaining:
            written = stream.write(remaining)
            if written is None or not 0 < written <= len(remaining):
                raise ConnectionError("Agent tool transport write stopped; do not resubmit")
            remaining = remaining[written:]


def send(stream, channel, data):
    if not isinstance(data, bytes):
        raise TypeError("Agent tool payload must be bytes")
    _write_frame(stream, json.dumps({"channel": channel, "bytes": len(data)}, separators=(",", ":")).encode())
    for offset in range(0, len(data), FRAME_BYTES):
        _write_frame(stream, data[offset:offset + FRAME_BYTES])
    stream.flush()


def send_json(stream, channel, value):
    send(stream, channel, json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())


def receive(stream, *, consume=None, max_bytes=None):
    header = json.loads(_frame(stream))
    if (not isinstance(header, dict) or set(header) != {"channel", "bytes"}
            or not isinstance(header["channel"], str) or type(header["bytes"]) is not int or header["bytes"] < 0):
        raise ValueError("invalid Agent tool message")
    if max_bytes is not None and header["bytes"] > max_bytes:
        raise ValueError("Agent tool metadata exceeds its bounded envelope")
    remaining = header["bytes"]
    chunks = bytearray() if consume is None else None
    while remaining:
        data = _frame(stream)
        if len(data) > remaining:
            raise ValueError("Agent tool message exceeds its declared length")
        remaining -= len(data)
        if consume is None:
            chunks.extend(data)
        else:
            consume(header["channel"], data)
    return header["channel"], bytes(chunks) if chunks is not None else None


def receive_json(stream, expected, *, max_bytes=None):
    channel, data = receive(stream, max_bytes=max_bytes)
    if channel != expected:
        raise ValueError("unexpected Agent tool message")
    return json.loads(data)
