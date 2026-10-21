"""Child-process entrypoint for frozen resource tool calls."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, is_dataclass


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported result type: {type(value).__name__}")


def main() -> int:
    try:
        request = json.load(sys.stdin)
        module_name, function_name = request["entrypoint"].split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        result = _jsonable(function(request["input"], request["context"]))
        sys.stdout.write(
            json.dumps(
                result, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - tool exceptions are returned through stderr and exit status.
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
