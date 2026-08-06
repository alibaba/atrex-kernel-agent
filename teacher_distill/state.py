from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .models import canonical_json


PRIVATE_STATE_FILE = "private_config.json"
PRIVATE_BENCHMARK_FILE = "benchmark_result.json"
PRIVATE_RESULT_FILE = "result.json"
PUBLIC_LOCK_FILE = "campaign_lock.json"
PUBLIC_TARGET_FILE = "teacher_target.json"


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("missing %s: %s" % (label, path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid %s: %s" % (label, exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("%s must be a JSON object" % label)
    return value


def hash_operator_inputs(op_dir: Path) -> str:
    names = (
        "definition.json",
        "reference.py",
        "workload.jsonl",
        "input.py",
        "shapes.json",
        "metadata.json",
        "roofline.json",
        "valid.py",
    )
    values: list[tuple[str, bytes]] = []
    for name in names:
        path = op_dir / name
        if path.is_file():
            values.append((name, path.read_bytes()))
    if not values:
        raise ValueError("operator directory has no recognized immutable inputs")
    digest = hashlib.sha256()
    for name, content in sorted(values):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def campaign_id_for(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return "campaign_" + digest[:24]
