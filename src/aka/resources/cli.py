"""Standalone consumer for a previously frozen AKA Agent tool plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .host import ResourcesError, restore_bound


def _read_request(source: str):
    text = (
        sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    )

    def bad_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(text, parse_constant=bad_constant)


def _bound(args):
    plan_path = os.environ.get("AKA_AGENT_TOOL_PLAN")
    plan_digest = os.environ.get("AKA_AGENT_TOOL_PLAN_DIGEST")
    workspace = os.environ.get("AKA_AGENT_WORKSPACE")
    campaign_id = os.environ.get("AKA_AGENT_CAMPAIGN_ID")
    if not all((plan_path, plan_digest, workspace, campaign_id)):
        raise ResourcesError(
            "selection_changed", "tool plan binding environment is incomplete"
        )
    if args.workspace and Path(args.workspace).resolve() != Path(workspace).resolve():
        raise ResourcesError(
            "selection_changed",
            "CLI workspace differs from the canonical resource binding",
        )
    binding = {
        "schema": "aka.bound-agent-resources.v1",
        "workspace": str(Path(workspace).resolve()),
        "campaign_id": campaign_id,
        "plan": str(Path(plan_path).resolve()),
        "digest": plan_digest,
    }
    base = None
    selection_path, selection_digest = (
        os.environ.get("AKA_LAUNCH_SELECTION"),
        os.environ.get("AKA_LAUNCH_DIGEST"),
    )
    if selection_path or selection_digest:
        if not selection_path or not selection_digest:
            raise ResourcesError(
                "selection_changed",
                "launch selection path and digest must travel together",
            )
        from aka.bootstrap.continuation import base_payload, digest

        raw = Path(selection_path).read_text(encoding="utf-8")
        if digest(raw) != selection_digest:
            raise ResourcesError(
                "selection_changed", "launch selection digest mismatch"
            )
        selection = json.loads(raw)
        if selection.get("version") != 3 or selection.get("agent_resources") != binding:
            raise ResourcesError(
                "selection_changed", "launch selection and tool plan binding differ"
            )
        base = base_payload(raw)
    return restore_bound(binding, base_selection=base)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aka tools", description=__doc__)
    parser.add_argument("--workspace", help="assert the canonical workspace binding")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    call = commands.add_parser("call")
    call.add_argument("tool")
    call.add_argument(
        "--input", required=True, help="JSON request file, or - for stdin"
    )
    args = parser.parse_args(argv)
    try:
        bound = _bound(args)
        if args.command == "list":
            installation = bound.resource_result()
            result = {
                "tools": installation.catalog(),
                "skills": installation.skill_catalog(),
            }
        else:
            result = bound.call_result(args.tool, _read_request(args.input))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        if args.command == "call" and result["receipt"]["status"] != "ok":
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI errors must always stay machine-readable.
        print(
            json.dumps(
                {
                    "error": {
                        "code": getattr(exc, "code", "invalid_input"),
                        "message": str(exc),
                    },
                    "schema": "aka.tool-result.v1",
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
