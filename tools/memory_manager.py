#!/usr/bin/env python3
# Copyright 2026 Alibaba Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
memory_manager.py - Structured memory JSON management tool for gpu-kernel-optimizer.

Manages per-iteration memory files (memory/v<N>.json) in the kernel optimization
workspace. Supports creation, reading, updating, masking, and summarizing iterations.

Usage:
    python tools/memory_manager.py <command> [options]

Commands:
    init          Create the memory/ directory in the workspace.
    create        Create a new iteration JSON file from the schema template.
    read          Read a specific iteration or all unmasked iterations.
    update        Update fields in a specific iteration JSON file.
    mask          Set masked=true on specified iteration(s).
    unmask        Set masked=false on specified iteration(s).
    summary       Print a performance summary table of all iterations.
    latest        Print the latest unmasked iteration version.
    list          List all iteration files with their masked status.

Examples:
    python tools/memory_manager.py init --workspace /tmp/kernel_opt_mla
    python tools/memory_manager.py create --workspace /tmp/kernel_opt_mla --version v0
    python tools/memory_manager.py read --workspace /tmp/kernel_opt_mla --version v1
    python tools/memory_manager.py read --workspace /tmp/kernel_opt_mla --version v1 --field performance.tflops
    python tools/memory_manager.py read --workspace /tmp/kernel_opt_mla --unmasked-only --field correctness.status
    python tools/memory_manager.py update --workspace /tmp/kernel_opt_mla --version v1 \\
        --set 'performance.tflops=150.2' --set 'correctness.status=PASS'
    python tools/memory_manager.py mask --workspace /tmp/kernel_opt_mla --version v2 v3
    python tools/memory_manager.py unmask --workspace /tmp/kernel_opt_mla --version v2
    python tools/memory_manager.py summary --workspace /tmp/kernel_opt_mla
    python tools/memory_manager.py latest --workspace /tmp/kernel_opt_mla
    python tools/memory_manager.py list --workspace /tmp/kernel_opt_mla
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_TEMPLATE = {
    "version": None,
    "masked": False,
    "timestamp": None,
    "performance": {
        "latency_us": None,
        "tflops": None,
        "bandwidth_gbps": None,
        "tflops_peak_utilization_pct": None,
        "bandwidth_peak_utilization_pct": None,
        "comparison_with_previous": {
            "latency_delta": None,
            "tflops_delta": None,
            "bandwidth_delta": None,
        },
    },
    "optimization": {
        "action_category": None,
        "action_description": None,
        "expected_impact": None,
        "risks_and_rollback": None,
    },
    "profile_evidence": {
        "tool_used": None,
        "evidence_summary": None,
        "bottleneck_type": None,
        "evidence_chain": None,
    },
    "correctness": {
        "rel_err": None,
        "status": None,
    },
    "isa_metric_progress": [],
    "search_log": [],
    "pitfalls_and_fixes": [],
    "references": [],
    "quality_gate": {
        "result": None,
        "failure_reason": None,
    },
    "git_commit_hash": None,
}


def get_memory_dir(workspace: str) -> Path:
    return Path(workspace) / "memory"


def get_iteration_path(workspace: str, version: str) -> Path:
    return get_memory_dir(workspace) / f"{version}.json"


def read_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(filepath: Path, data: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def parse_version_number(version_str: str) -> int:
    """Extract numeric part from version string like 'v0', 'v12'."""
    match = re.match(r"v(\d+)", version_str)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid version format: {version_str}. Expected 'v<N>' like v0, v1.")


def list_iteration_files(workspace: str) -> list[Path]:
    """List all memory/v*.json files sorted by version number."""
    memory_dir = get_memory_dir(workspace)
    if not memory_dir.exists():
        return []
    files = sorted(
        memory_dir.glob("v*.json"),
        key=lambda p: parse_version_number(p.stem),
    )
    return files


def get_nested_value(data: dict, dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation like 'performance.tflops'."""
    keys = dotted_key.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise KeyError(f"Key '{dotted_key}' not found (failed at '{key}').")
    return current


def set_nested_value(data: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation like 'performance.tflops'."""
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def parse_typed_value(raw_value: str) -> Any:
    """Parse a string value into the appropriate Python type."""
    if raw_value.lower() == "null" or raw_value.lower() == "none":
        return None
    if raw_value.lower() == "true":
        return True
    if raw_value.lower() == "false":
        return False
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> None:
    """Create memory/ directory in the workspace."""
    memory_dir = get_memory_dir(args.workspace)
    memory_dir.mkdir(parents=True, exist_ok=True)
    print(f"Initialized memory directory: {memory_dir}")


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new iteration JSON file from the schema template."""
    memory_dir = get_memory_dir(args.workspace)
    if not memory_dir.exists():
        memory_dir.mkdir(parents=True, exist_ok=True)

    version = args.version
    parse_version_number(version)  # validate format

    filepath = get_iteration_path(args.workspace, version)
    if filepath.exists() and not args.force:
        print(f"Error: {filepath} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    data = deepcopy(SCHEMA_TEMPLATE)
    data["version"] = version
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    write_json(filepath, data)
    print(f"Created: {filepath}")


def _extract_field(data: dict, field: Optional[str]) -> Any:
    """Extract a specific field from data, or return the full data if field is None."""
    if field is None:
        return data
    try:
        return get_nested_value(data, field)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_read(args: argparse.Namespace) -> None:
    """Read iteration JSON files, optionally extracting a specific field."""
    field = getattr(args, "field", None)

    if args.version:
        filepath = get_iteration_path(args.workspace, args.version)
        if not filepath.exists():
            print(f"Error: {filepath} not found.", file=sys.stderr)
            sys.exit(1)
        data = read_json(filepath)
        if data.get("masked", False):
            print(f"Error: {args.version} is masked and cannot be read. Use 'unmask' first.", file=sys.stderr)
            sys.exit(1)
        result = _extract_field(data, field)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    files = list_iteration_files(args.workspace)
    if not files:
        print("No iteration files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for filepath in files:
        data = read_json(filepath)
        if data.get("masked", False):
            continue
        if field is not None:
            try:
                extracted = get_nested_value(data, field)
            except KeyError:
                extracted = None
            results.append({"version": data.get("version", filepath.stem), field: extracted})
        else:
            results.append(data)

    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_update(args: argparse.Namespace) -> None:
    """Update fields in a specific iteration JSON file."""
    filepath = get_iteration_path(args.workspace, args.version)
    if not filepath.exists():
        print(f"Error: {filepath} not found.", file=sys.stderr)
        sys.exit(1)

    data = read_json(filepath)

    for assignment in args.set:
        if "=" not in assignment:
            print(f"Error: Invalid --set format '{assignment}'. Expected 'key=value'.", file=sys.stderr)
            sys.exit(1)
        dotted_key, raw_value = assignment.split("=", 1)
        typed_value = parse_typed_value(raw_value)
        set_nested_value(data, dotted_key.strip(), typed_value)

    write_json(filepath, data)
    print(f"Updated: {filepath}")


def cmd_mask(args: argparse.Namespace) -> None:
    """Set masked=true on specified iteration(s)."""
    for version in args.version:
        filepath = get_iteration_path(args.workspace, version)
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.", file=sys.stderr)
            continue
        data = read_json(filepath)
        data["masked"] = True
        write_json(filepath, data)
        print(f"Masked: {filepath}")


def cmd_unmask(args: argparse.Namespace) -> None:
    """Set masked=false on specified iteration(s)."""
    for version in args.version:
        filepath = get_iteration_path(args.workspace, version)
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.", file=sys.stderr)
            continue
        data = read_json(filepath)
        data["masked"] = False
        write_json(filepath, data)
        print(f"Unmasked: {filepath}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Print a performance summary table of all iterations."""
    files = list_iteration_files(args.workspace)
    if not files:
        print("No iteration files found.", file=sys.stderr)
        sys.exit(1)

    header = f"{'Version':<10} {'Masked':<8} {'TFLOPS':>10} {'BW(GB/s)':>12} {'TFLOPS%':>10} {'BW%':>10} {'Status':<14} {'Action'}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    for filepath in files:
        data = read_json(filepath)
        version = data.get("version", filepath.stem)
        masked = "YES" if data.get("masked", False) else ""
        perf = data.get("performance", {})
        tflops = perf.get("tflops")
        bandwidth = perf.get("bandwidth_gbps")
        tflops_pct = perf.get("tflops_peak_utilization_pct")
        bw_pct = perf.get("bandwidth_peak_utilization_pct")
        correctness = data.get("correctness", {})
        status = correctness.get("status", "")
        gate = data.get("quality_gate", {})
        gate_result = gate.get("result", "")
        combined_status = f"{status or ''}/{gate_result or ''}"
        optimization = data.get("optimization", {})
        action = optimization.get("action_category", "") or ""

        tflops_str = f"{tflops:.1f}" if tflops is not None else "-"
        bw_str = f"{bandwidth:.1f}" if bandwidth is not None else "-"
        tflops_pct_str = f"{tflops_pct:.1f}%" if tflops_pct is not None else "-"
        bw_pct_str = f"{bw_pct:.1f}%" if bw_pct is not None else "-"

        print(f"{version:<10} {masked:<8} {tflops_str:>10} {bw_str:>12} {tflops_pct_str:>10} {bw_pct_str:>10} {combined_status:<14} {action}")


def cmd_latest(args: argparse.Namespace) -> None:
    """Print the latest unmasked iteration version."""
    files = list_iteration_files(args.workspace)
    if not files:
        print("No iteration files found.", file=sys.stderr)
        sys.exit(1)

    latest: Optional[str] = None
    for filepath in reversed(files):
        data = read_json(filepath)
        if not data.get("masked", False):
            latest = data.get("version", filepath.stem)
            break

    if latest:
        print(latest)
    else:
        print("No unmasked iterations found.", file=sys.stderr)
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """List all iteration files with their masked status."""
    files = list_iteration_files(args.workspace)
    if not files:
        print("No iteration files found.", file=sys.stderr)
        sys.exit(1)

    for filepath in files:
        data = read_json(filepath)
        version = data.get("version", filepath.stem)
        masked_flag = " [MASKED]" if data.get("masked", False) else ""
        timestamp = data.get("timestamp", "")
        print(f"{version}{masked_flag}  {timestamp}  {filepath}")


# ── Argument Parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage structured memory JSON files for gpu-kernel-optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    sub_init = subparsers.add_parser("init", help="Create memory/ directory.")
    sub_init.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")

    # create
    sub_create = subparsers.add_parser("create", help="Create a new iteration JSON file.")
    sub_create.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")
    sub_create.add_argument("--version", required=True, help="Version string, e.g. v0, v1.")
    sub_create.add_argument("--force", action="store_true", help="Overwrite if file exists.")

    # read
    sub_read = subparsers.add_parser("read", help="Read iteration JSON file(s).")
    sub_read.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")
    sub_read.add_argument("--version", help="Specific version to read, e.g. v1.")
    sub_read.add_argument("--field", help="Extract a specific field using dot notation, e.g. 'performance.tflops'.")
    sub_read.add_argument("--unmasked-only", action="store_true", help="Only read unmasked iterations.")

    # update
    sub_update = subparsers.add_parser("update", help="Update fields in an iteration JSON file.")
    sub_update.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")
    sub_update.add_argument("--version", required=True, help="Version to update, e.g. v1.")
    sub_update.add_argument("--set", action="append", required=True,
                            help="Field=value pair using dot notation, e.g. 'performance.tflops=150.2'.")

    # mask
    sub_mask = subparsers.add_parser("mask", help="Set masked=true on iteration(s).")
    sub_mask.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")
    sub_mask.add_argument("--version", nargs="+", required=True, help="Version(s) to mask, e.g. v2 v3.")

    # unmask
    sub_unmask = subparsers.add_parser("unmask", help="Set masked=false on iteration(s).")
    sub_unmask.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")
    sub_unmask.add_argument("--version", nargs="+", required=True, help="Version(s) to unmask, e.g. v2.")

    # summary
    sub_summary = subparsers.add_parser("summary", help="Print performance summary table.")
    sub_summary.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")

    # latest
    sub_latest = subparsers.add_parser("latest", help="Print latest unmasked iteration version.")
    sub_latest.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")

    # list
    sub_list = subparsers.add_parser("list", help="List all iteration files with masked status.")
    sub_list.add_argument("--workspace", required=True, help="Path to kernel optimization workspace.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "init": cmd_init,
        "create": cmd_create,
        "read": cmd_read,
        "update": cmd_update,
        "mask": cmd_mask,
        "unmask": cmd_unmask,
        "summary": cmd_summary,
        "latest": cmd_latest,
        "list": cmd_list,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
