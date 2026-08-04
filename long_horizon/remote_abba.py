#!/usr/bin/env python3
"""Execute a complete incumbent/candidate ABBA schedule in one GPU allocation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="


def _safe_relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("snapshot path must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ValueError(f"unsafe snapshot path: {value!r}")
    return path.as_posix()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _apply_revision(root: Path, snapshot_root: Path, manifest: dict[str, object]) -> None:
    for raw_path, raw_source in manifest.items():
        relative = _safe_relative(raw_path)
        target = root / relative
        if raw_source is None:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            continue
        source_relative = _safe_relative(raw_source)
        source = snapshot_root / source_relative
        if not source.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _parse_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line[len(RESULT_PREFIX) :])
            return value if isinstance(value, dict) else None
    return None


def run(request_path: Path, result_path: Path) -> int:
    runs: list[dict[str, Any]] = []
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("schema_version") != 1:
            raise ValueError("unsupported ABBA request schema")
        schedule = request.get("schedule")
        manifests = request.get("manifests")
        command = request.get("command")
        timeout = int(request.get("run_timeout_seconds", 120))
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("ABBA schedule must be a non-empty list")
        if not isinstance(manifests, dict):
            raise ValueError("ABBA manifests must be an object")
        if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
            raise ValueError("ABBA command must be a list of strings")
        if timeout <= 0:
            raise ValueError("run timeout must be positive")
        root = Path.cwd()
        snapshot_root = request_path.parent
        for step in schedule:
            if not isinstance(step, dict):
                raise ValueError("ABBA schedule entries must be objects")
            revision = step.get("revision")
            repeat = step.get("repeat")
            if revision not in {"incumbent", "candidate"} or not isinstance(repeat, int):
                raise ValueError("invalid ABBA schedule entry")
            manifest = manifests.get(revision)
            if not isinstance(manifest, dict):
                raise ValueError(f"missing {revision} snapshot manifest")
            _apply_revision(root, snapshot_root, manifest)
            try:
                process = subprocess.run(
                    command,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=os.environ.copy(),
                )
                result = _parse_result(process.stdout)
                row = {
                    "revision": revision,
                    "repeat": repeat,
                    "exit_code": process.returncode,
                    "result": result,
                    "stdout_tail": process.stdout[-3000:],
                    "stderr_tail": process.stderr[-3000:],
                }
            except subprocess.TimeoutExpired as exc:
                row = {
                    "revision": revision,
                    "repeat": repeat,
                    "exit_code": -1,
                    "result": None,
                    "stdout_tail": str(exc.stdout or "")[-3000:],
                    "stderr_tail": "evaluation timed out",
                }
            runs.append(row)
        payload = {"schema_version": 1, "runs": runs, "error": None}
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "runs": runs,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _atomic_json(result_path, payload)
    # Return the authoritative result through ordinary command stdout.  The
    # long-horizon verifier deliberately runs the sandbox with --no-sync so a
    # gateway cannot replace an artifact frame with an elision notice that is
    # not valid base64.  Keep result.json on the worker as debugging evidence.
    print(
        ABBA_RESULT_PREFIX
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: remote_abba.py REQUEST_JSON RESULT_JSON", file=sys.stderr)
        return 2
    return run(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
