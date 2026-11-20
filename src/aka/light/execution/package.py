"""JSON transport for one source package, including binary resources."""
from __future__ import annotations
import base64


def encode_files(files):
    result = {}
    for name, data in files.items():
        try:
            result[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            result[name] = {"base64": base64.b64encode(data).decode("ascii")}
    return result


def decode_files(files):
    return {name: value.encode("utf-8") if isinstance(value, str)
            else base64.b64decode(value["base64"], validate=True)
            for name, value in files.items()}
