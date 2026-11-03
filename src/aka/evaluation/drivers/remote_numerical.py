#!/usr/bin/env python3
"""Supervisor-owned distribution probes inside the existing GPU evaluator allocation.

The original generator supplies tensor metadata and unchanged structural inputs.
Only the requested tensor leaves are regenerated; tuple/list/dict ABIs are preserved.
Supplemental floating outputs use relative L2; structural and mutation checks
remain owned by the evaluator.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

if __package__:
    from .numerical_inputs import (
        GENERATORS, validate_suite, validation_schedule, matches_constraints,
        tensor_leaves, replace_leaves, numerical_inputs, install_inputs,
    )
else:
    from numerical_inputs import (
        GENERATORS, validate_suite, validation_schedule, matches_constraints,
        tensor_leaves, replace_leaves, numerical_inputs, install_inputs,
    )

PREFIX = "__ATREX_NUMERICAL_RESULT__="
MAX_REL_L2 = 1e-3


def _run_live(request_path, request=None):
    request = json.loads(request_path.read_text()) if request is None else request
    suite = validate_suite(request["suite"])
    root = Path.cwd()
    input_path = root / "input.py"
    harness = root / "test_kernel.py"
    original = input_path.read_bytes()
    original_harness = harness.read_bytes()
    rows = []
    try:
        shapes = json.loads((root / "shapes.json").read_text())
        schedule = validation_schedule(
            suite, shapes, request.get("rotation", ""), request.get("mode", "light")
        )
        # Native Atrex workspaces use the supervisor's current transport adapter.
        # This replacement exists only in this isolated allocation, not the workspace.
        snapshot = request_path.parent / "snapshots" / "evaluator.py"
        if snapshot.is_file():
            harness.write_bytes(snapshot.read_bytes())
        evaluator = suite.get("evaluator_command", [sys.executable, "test_kernel.py"])
        for plan in schedule:
            if plan["case_id"] not in request.get(
                "case_ids", [c["id"] for c in suite["cases"]]
            ):
                continue
            case = next(c for c in suite["cases"] if c["id"] == plan["case_id"])
            if plan.get("status") == "unsupported":
                rows.append(plan)
                continue
            # Evaluate one workload/seed at a time. Evaluators can stop a shape's
            # seed loop on its first mismatch; that is a counterexample, not a
            # broken probe plan. Bind each result to its own input receipt.
            expected = {
                (
                    json.dumps(shapes[sid].get("input_kwargs") or {}, sort_keys=True),
                    seed,
                    rank,
                )
                for sid in plan["shape_ids"]
                for seed in plan["seeds"]
                for rank in range(suite["world_size"])
            }
            seen = set()
            failed_probes = 0
            results = []
            diagnosis = ""
            nonfinite_outputs = set()
            deadline = time.monotonic() + request["per_case_timeout"]
            for shape_id in plan["shape_ids"]:
                for seed in plan["seeds"]:
                    receipts = request_path.parent / "receipts.jsonl"
                    receipts.unlink(missing_ok=True)
                    input_error = Path(str(receipts) + ".error")
                    input_error.unlink(missing_ok=True)
                    tail = (
                        "\nimport runpy as __numeric_runpy\n"
                        f"__numeric_runpy.run_path({str(Path(__file__).resolve())!r})['install_inputs'](globals(), {case!r}, {[seed]!r}, {str(receipts.resolve())!r})\n"
                    )
                    input_path.write_bytes(original + tail.encode())
                    for stem in ("input", "test_kernel"):
                        for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                            cached.unlink()
                    command = [
                        *evaluator,
                        "--version",
                        "vlong",
                        "--no-memory",
                        "--correctness-only",
                        "--correctness-max-rel-l2",
                        str(request.get("max_rel_l2", MAX_REL_L2)),
                        "--multi-seed",
                        "0",
                        "--seed",
                        str(seed),
                        "--shape-id",
                        shape_id,
                    ]
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(
                                command, request["per_case_timeout"]
                            )
                        process = subprocess.run(
                            command,
                            cwd=root,
                            capture_output=True,
                            text=True,
                            timeout=remaining,
                        )
                    except subprocess.TimeoutExpired:
                        diagnosis = "supplemental evaluator exceeded its case timeout"
                        break
                    result = None
                    for line in process.stdout.splitlines():
                        if line.startswith("[test_kernel] RESULT_JSON="):
                            result = json.loads(line.split("RESULT_JSON=", 1)[1])
                    observed = (
                        [json.loads(line) for line in receipts.read_text().splitlines()]
                        if receipts.exists()
                        else []
                    )
                    actual = {
                        (
                            json.dumps(r["input_kwargs"], sort_keys=True),
                            r["seed"],
                            r["rank"],
                        )
                        for r in observed
                    }
                    required = {
                        (
                            json.dumps(
                                shapes[shape_id].get("input_kwargs") or {},
                                sort_keys=True,
                            ),
                            seed,
                            rank,
                        )
                        for rank in range(suite["world_size"])
                    }
                    seen.update(actual & expected)
                    if input_error.exists():
                        diagnosis = input_error.read_text()
                    elif actual != required:
                        diagnosis = "evaluator did not generate the requested workload, seed and ranks"
                    elif not isinstance(result, dict):
                        diagnosis = (
                            f"evaluator exited {process.returncode} without a result"
                        )
                    elif "unknown" in result.get("nonfinite_outputs", []):
                        diagnosis = "non-finite output diagnostic has unknown roles"
                    elif "reference" in result.get("nonfinite_outputs", []):
                        diagnosis = "reference output is non-finite under the requested probe inputs"
                    elif result.get("all_pass") is False:
                        failed_probes += 1
                    elif result.get("all_pass") is not True or process.returncode != 0:
                        diagnosis = f"evaluator returned an inconsistent result (exit={process.returncode})"
                    if isinstance(result, dict):
                        results.append(result)
                        nonfinite_outputs.update(result.get("nonfinite_outputs", []))
                    if diagnosis or failed_probes:
                        break
                if diagnosis or failed_probes:
                    break
            passed = not diagnosis and not failed_probes and seen == expected
            metrics = {}
            for result in results:
                values = {
                    **{
                        name: result[name]
                        for name in ("max_abs_err", "max_rel_err")
                        if name in result
                    },
                    **result.get("numerical_metrics", {}),
                }
                for name, value in values.items():
                    if value is None or not math.isfinite(value):
                        metrics[name] = None
                    elif metrics.get(name, 0) is not None:
                        metrics[name] = max(metrics.get(name, 0), value)
            rows.append(
                {
                    "case_id": case["id"],
                    "passed": passed,
                    "exit_code": 0 if passed else 1,
                    "expected_probes": len(expected),
                    "observed_probes": len(seen),
                    "failed_probes": failed_probes,
                    "selection_digest": plan["selection_digest"],
                    "shape_count": len(plan["shape_ids"]),
                    "seeds": plan["seeds"],
                    "world_size": suite["world_size"],
                    "result": {
                        "all_pass": passed,
                        "numerical_metrics": metrics,
                        "nonfinite_outputs": sorted(nonfinite_outputs),
                    },
                    "input_error": diagnosis,
                }
            )
            if not passed:
                break
        payload = {
            "schema_version": 1,
            "runs": rows,
            "all_pass": bool(rows) and all(r.get("passed") is True for r in rows),
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "runs": rows,
            "all_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        input_path.write_bytes(original)
        harness.write_bytes(original_harness)
        for stem in ("input", "test_kernel"):
            for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                cached.unlink()
    print(PREFIX + json.dumps(payload, allow_nan=False), flush=True)
    return 0


def _safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("numerical snapshot path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("numerical snapshot path is invalid")
    return path


def _clear_cache(paths):
    for path in paths:
        if path.suffix == ".py":
            path.with_suffix(".pyc").unlink(missing_ok=True)
            for cached in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
                cached.unlink(missing_ok=True)


@contextmanager
def _snapshot_files(request, root, directory):
    if "candidate_manifest" not in request:
        yield  # Retained standalone requests preserve their original live-file route.
        return
    candidate, trusted = request["candidate_manifest"], request["trusted_manifest"]
    if not isinstance(candidate, list) or not candidate or not isinstance(trusted, list):
        raise ValueError("numerical snapshot manifests are invalid")
    rows, frozen, seen = [], [], set()
    trusted_paths = {row.get("path") for row in trusted if isinstance(row, dict)}
    for role, manifest in (("candidate", candidate), ("trusted", trusted)):
        for row in manifest:
            if not isinstance(row, dict) or set(row) != {"path", "snapshot", "mode", "sha256"}:
                raise ValueError("numerical snapshot entry is invalid")
            relative, snapshot = _safe_path(row["path"]), _safe_path(row["snapshot"])
            if snapshot.parts[0] != "snapshots" or type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o777:
                raise ValueError("numerical snapshot entry is invalid")
            if row["path"] in seen or role == "candidate" and row["path"] in trusted_paths:
                raise ValueError("numerical snapshot path overlaps trusted inputs")
            seen.add(row["path"])
            source, target = directory / snapshot, root / relative
            if not source.resolve().is_relative_to((directory / "snapshots").resolve()) or source.is_symlink():
                raise ValueError("numerical snapshot source escapes its frozen input root")
            if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root and parent.is_relative_to(root)):
                raise ValueError("numerical snapshot target escapes its allocation")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("numerical snapshot content identity mismatch")
            original = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) if target.exists() else None
            frozen.append((target, content, row["mode"], original))
            if role == "candidate":
                rows.append({key: row[key] for key in ("path", "mode", "sha256")})
    rows.sort(key=lambda row: row["path"])
    identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    if identity != request["candidate_identity"]:
        raise ValueError("numerical candidate snapshot identity mismatch")
    touched, created = [], []
    try:
        for target, content, mode, original in frozen:
            for parent in reversed(target.parents):
                if parent.is_relative_to(root) and not parent.exists():
                    parent.mkdir()
                    created.append(parent)
            touched.append((target, original))
            target.write_bytes(content)
            target.chmod(mode)
        _clear_cache(target for target, _ in touched)
        yield
    finally:
        for target, original in reversed(touched):
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original[0])
                target.chmod(original[1])
        _clear_cache(target for target, _ in touched)
        for path in reversed(created):
            try:
                path.rmdir()
            except OSError:
                pass  # Evaluator-owned diagnostics may remain in this isolated allocation.


def run(request_path):
    request = json.loads(request_path.read_text())
    if not isinstance(request, dict) or "candidate_manifest" not in request:
        return _run_live(request_path, request)
    try:
        with _snapshot_files(request, Path.cwd(), request_path.parent):
            return _run_live(request_path, request)
    except Exception as exc:
        print(PREFIX + json.dumps({"schema_version": 1, "runs": [], "all_pass": False,
            "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
