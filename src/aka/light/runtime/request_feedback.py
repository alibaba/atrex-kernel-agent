"""Effective Light request contracts for rejected calls, not a new validation policy.

Mirror Runtime's issues/request_schema/recovery feedback while keeping Light CLI
names, IDs and the existing error envelope. Never describe controller authority
flags, resolved defaults, private evidence or a disabled research workflow.
"""
from __future__ import annotations

import argparse
import re
from copy import deepcopy


JOURNAL_TOOLS = {
    "direction_update": "update-direction", "experiment_record": "record-experiment",
    "episode_report": "episode-report", "directions_list": "list-directions",
    "experiments_list": "list-experiments", "direction_load": "load-direction",
    "experiment_load": "load-experiment",
}


def _text():
    return {"type": "string", "minLength": 1, "maxLength": 16000}


def _ids(prefix, *, minimum=0, maximum=64):
    return {"type": "array", "minItems": minimum, "maxItems": maximum, "uniqueItems": True,
            "items": {"type": "string", "pattern": f"^{prefix}[0-9a-f]{{32}}$"}}


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": sorted(required),
            "additionalProperties": False}


def journal_schema(operation, body, *, research_records=False, supervisor_git=True):
    """The schema describes the JSON request file, not its HTTP envelope."""
    from aka.light.workflow.journal_service import (
        DIRECTION_PROPOSAL_FIELDS,
        DIRECTION_UPDATE_FIELDS,
        DIRECTION_CLOSURE_FIELDS,
        EXPERIMENT_FIELDS,
        EXPERIMENT_ACTIONS,
    )
    from aka.light.workflow.direction_lifecycle import DIRECTION_STATUS, CLOSURES
    from aka.light.workflow.direction_genealogy import RELATIONSHIPS

    if operation in {"direction_update", "experiment_record"} and not research_records:
        return {"available": False, "reason": "research_records_disabled"}
    if operation == "episode_report":
        properties = {"status": {"enum": ["candidate_ready", "pivot", "blocked"]}, "summary": _text(),
                      "accepted_ppu_diagnostics": {"type": "array", "items": {"type": "object"}}}
        required = {"status", "summary"}
        status = body.get("status")
        status = status if isinstance(status, str) else None
        if status in {None, "candidate_ready"}:
            field = "selected_experiment_id" if research_records else "gateway_record_ids"
            properties[field] = ({"type": "string", "pattern": "^experiment_[0-9a-f]{32}$"}
                                 if research_records else _ids("gateway-", minimum=1))
            if status == "candidate_ready":
                required.add(field)
            if not supervisor_git:
                properties["candidate_commit"] = {"type": "string", "pattern": "^[0-9a-f]{40}([0-9a-f]{24})?$"}
                if status == "candidate_ready":
                    required.add("candidate_commit")
        if status in {None, "blocked"}:
            properties["blocker"] = _text()
            if status == "blocked":
                required.add("blocker")
        return _object(properties, required)
    if operation == "experiment_record":
        from aka.legacy.evaluation.evidence_fields import (
            QUERY_ID_RE, WIKI_ID_RE, WIKI_CITE_RE, WIKI_USAGE_DISPOSITIONS, WIKI_USAGE_STATUSES,
        )
        query_id = {"type": "string", "minLength": 1, "maxLength": 256,
                    "pattern": "^" + QUERY_ID_RE.pattern + "$"}
        usage = _object({
            "query_id": query_id,
            "wiki_id": {"type": "string", "minLength": 1, "maxLength": 1090,
                        "pattern": "^" + WIKI_ID_RE.pattern + "$"},
            "cite": {"type": "string", "minLength": 1, "maxLength": 4096,
                     "pattern": "^" + WIKI_CITE_RE.pattern.replace("(?P<path>", "(")
                     .replace("(?P<start>", "(").replace("(?P<end>", "(") + "$",
                     "description": "Copy wiki@version relative-path:start-end verbatim; no parent paths; end >= start."},
            "disposition": {"type": "string", "enum": sorted(WIKI_USAGE_DISPOSITIONS)},
            "use": _text(), "evidence": _text(),
        }, {"query_id", "disposition", "use", "evidence"})
        usage["anyOf"] = [{"required": ["wiki_id"]}, {"required": ["cite"]}]
        properties = {key: _text() for key in EXPERIMENT_FIELDS}
        properties.update(direction_id={"type": "string", "pattern": "^direction_[0-9a-f]{32}$"},
                          gateway_record_ids=_ids("gateway-", minimum=1), action={"enum": sorted(EXPERIMENT_ACTIONS)},
                          wiki_usage={"type": "array", "maxItems": 64, "items": usage},
                          wiki_query_ids={"type": "array", "maxItems": 64, "items": query_id},
                          wiki_usage_status={"type": "string", "enum": sorted(WIKI_USAGE_STATUSES)})
        return _object(properties, EXPERIMENT_FIELDS)
    if operation == "direction_update":
        action = body.get("action")
        if not isinstance(action, str) or action not in DIRECTION_STATUS:
            return {"oneOf": [journal_schema(operation, {"action": item}, research_records=True)
                              for item in DIRECTION_STATUS]}
        fields = (DIRECTION_PROPOSAL_FIELDS if action == "propose" else
                  DIRECTION_CLOSURE_FIELDS if action in CLOSURES else DIRECTION_UPDATE_FIELDS)
        properties = {key: _text() for key in fields}
        properties["action"] = {"const": action}
        if action == "propose":
            for key in ("plan", "success_criteria", "stop_conditions"):
                properties[key] = {"type": "array", "maxItems": 64,
                                   "items": {"type": "string", "minLength": 1, "maxLength": 4000}}
            properties.update(relationship={"enum": list(RELATIONSHIPS)},
                              derived_from_direction_ids=_ids("direction_", maximum=32),
                              derived_from_experiment_ids=_ids("experiment_", maximum=32),
                              supersedes_direction_id={"type": "string", "pattern": "^direction_[0-9a-f]{32}$"})
        else:
            properties["direction_id"] = {"type": "string", "pattern": "^direction_[0-9a-f]{32}$"}
            if action in CLOSURES:
                properties["hypothesis_status"] = {"enum": ["unresolved", "supported", "refuted"]}
                properties["supporting_experiment_ids"] = _ids("experiment_", minimum=1, maximum=32)
        return _object(properties, fields)
    if operation in {"directions_list", "experiments_list"}:
        return {"cli_options": {"--output-path": {"type": "string", "pattern": "^scratch/.+"}},
                "required_options": ["--output-path"]}
    if operation in {"direction_load", "experiment_load"}:
        prefix = "direction_" if operation == "direction_load" else "experiment_"
        return {"cli_options": {"--record-id": {"type": "string", "pattern": f"^{prefix}[0-9a-f]{{32}}$"}},
                "required_options": ["--record-id"]}
    return None


def gateway_schema(kind, parser):
    """Read option types/choices from the real parser; never expose its defaults."""
    groups = {
        "run": {"--mode", "--version", "--multi-seed", "--shape-id", "--timed-runs", "--baseline-path",
                "--comparison-repeats", "--comparison-run-timeout", "--shape-batch-size", "--input-path", "--shapes-path"},
        "profile": {"--kernel-name", "--kernel-regex", "--profile-source", "--launch-skip", "--launch-count",
                    "--profile-shape-id", "--requirement", "--deps-mode", "--profile-level", "--profiler",
                    "--profile-counter", "--top-kernels", "--include-raw-profile"},
        "check": {"--arch", "--sanitize", "--requirement", "--deps-mode"},
        "disassemble": {"--format", "--requirement", "--deps-mode"},
        "dev": set(), "env": {"--env-gpu", "--env-capabilities", "--env-force"},
        "record-read": {"--record-id"}, "kernel-read": {"--kernel-id", "--output-path"},
        "kernel-records": {"--kernel-id"}, "kernel-pareto-frontier": set(),
    }
    if kind not in groups:
        return {"supported_kinds": list(groups), "instruction": "Use an explicit --kind and full option names."}
    allowed = groups[kind] | {"--kind"}
    if kind in {"run", "profile", "dev", "check", "disassemble"}:
        allowed |= {"--input", "--sync", "--no-sync", "--env", "--inline-output", "--keep-pod", "--dry-run"}
    options = {}
    for action in parser._actions:
        for flag in action.option_strings:
            if flag not in allowed:
                continue
            item = {"type": "boolean" if action.nargs == 0 else
                    "integer" if action.type is int else "number" if action.type is float else "string"}
            if action.choices is not None:
                item["enum"] = list(action.choices)
            if isinstance(action, argparse._AppendAction):
                item["repeatable"] = True
            options[flag] = item
    options["--kind"] = {"const": kind}
    required = ["--kind", *({"record-read": ["--record-id"], "kernel-read": ["--kernel-id", "--output-path"],
                             "kernel-records": ["--kernel-id"]}.get(kind, []))]
    schema = {"cli_options": options, "required_options": required}
    if kind in {"run", "profile", "dev"}:
        schema["command"] = "Optional argv after --; run defaults to python3 test_kernel.py."
    return schema


def field_issues(body, schema):
    """Bounded field diagnostics after rejection; never echo submitted values."""
    issues = []
    def add(path, code, message):
        if len(issues) < 32:
            issues.append({"path": path, "code": code, "message": message})
    properties = schema.get("properties", {})
    for field in schema.get("required", []):
        if field not in body:
            add(field, "missing", "Field required")
    for field, value in body.items():
        if len(issues) >= 32:
            break
        if field not in properties:
            if schema.get("additionalProperties") is False:
                add(field, "extra_forbidden", "Unexpected field for this request")
            continue
        spec = properties[field]
        expected = spec.get("type")
        if expected == "string" and (not isinstance(value, str) or not value.strip()):
            add(field, "string_type", "Expected non-empty text")
        elif expected == "array" and not isinstance(value, list):
            add(field, "list_type", "Expected an array")
        elif "enum" in spec and value not in spec["enum"]:
            add(field, "enum", "Use one of the values in request_schema")
        elif "const" in spec and value != spec["const"]:
            add(field, "literal_error", "Use the value in request_schema")
        elif isinstance(value, str):
            if len(value) > spec.get("maxLength", len(value)):
                add(field, "string_too_long", "Text exceeds maxLength in request_schema")
            elif "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
                add(field, "string_pattern_mismatch", "Use the complete ID format in request_schema")
        elif isinstance(value, list):
            if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", len(value)):
                add(field, "list_length", "Array size must satisfy request_schema")
            else:
                item_spec = spec.get("items", {})
                for index, item in enumerate(value[:64]):
                    if item_spec.get("type") == "string" and (
                        not isinstance(item, str) or not item.strip()
                        or len(item) > item_spec.get("maxLength", len(item))
                        or ("pattern" in item_spec and re.fullmatch(item_spec["pattern"], item) is None)
                    ):
                        add(f"{field}.{index}", "invalid_string", "Expected text/ID matching request_schema")
    return issues


def augment_error(response, *, route, request, research_records=False, supervisor_git=True, parser=None,
                  catalog=()):
    """Add only public format guidance; preserve authoritative recovery/classification."""
    result = deepcopy(response)
    request = request if isinstance(request, dict) else {}
    error = result.get("error")
    if not isinstance(error, dict):
        return result
    if not result.get("repairable", False):
        return result  # Never suggest repairing/replaying an unknown-outcome mutation.
    if error.get("code") == "request_not_started":
        return result
    if route == "/v1/journal/execute":
        operation = request.get("operation")
        if not isinstance(operation, str):
            operation = ""
        body = request.get("request")
        body = body if isinstance(body, dict) else {}
        schema = journal_schema(operation, body, research_records=research_records, supervisor_git=supervisor_git)
        if schema is not None:
            result["request_schema"] = schema
            issues = field_issues(body, schema)
            if issues:
                # Schema hints supplement, never hide, the validator's semantic
                # finding (e.g. missing evidence plus a missing Commit field).
                specific = [item for item in result.get("issues", []) if item.get("path") != "$"]
                result["issues"] = (specific + [item for item in issues if item not in specific])[:32]
        else:
            result["supported_operations"] = [op for op in JOURNAL_TOOLS if research_records or op not in
                                               {"direction_update", "experiment_record"}]
        if operation in JOURNAL_TOOLS:
            result["operation"] = JOURNAL_TOOLS[operation]
        if operation == "episode_report":
            instruction = (
                "Correct the report fields or cited evidence and resubmit episode-report. "
                "A validation rejection does not complete the Episode. " +
                ("Select the current Episode's Experiment and close active Directions." if research_records else
                 "Do not create Direction/Experiment records. candidate_ready cites gateway_record_ids, "
                 "including a passing standard full Evaluate for a byte-identical complete candidate."))
            from aka.light.runtime.request_errors import REPAIR_REQUEST
            if error.get("next_action") == REPAIR_REQUEST:
                error["next_action"] = instruction
                if result.get("recovery") == [{"instruction": REPAIR_REQUEST}]:
                    result["recovery"] = []
            result["recovery"].append({"instruction": instruction})
    elif route == "/v1/gateway/execute" and parser is not None:
        argv = request.get("argv")
        argv = argv if isinstance(argv, list) else []
        kind = None
        for index, word in enumerate(argv):
            if word == "--":
                break
            if not isinstance(word, str):
                continue
            if word == "--kind" and index + 1 < len(argv):
                kind = argv[index + 1]
            elif word.startswith("--kind="):
                kind = word.split("=", 1)[1]
        result["request_schema"] = gateway_schema(kind if isinstance(kind, str) else None, parser)
        if isinstance(kind, str) and kind in {"record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier"}:
            result["recovery"].append({"instruction":
                "Use an ID returned by this Campaign. record-read needs --record-id; kernel-read needs "
                "--kernel-id and --output-path scratch/FILE; kernel-records needs --kernel-id. "
                "kernel-pareto-frontier takes no IDs. Queries never require a GPU resubmission."})
    elif route == "/v1/plugins/execute":
        name = request.get("tool")
        tool = next((item for item in catalog if item["name"] == name), None)
        if tool is not None:
            result["request_schema"] = deepcopy(tool["input_schema"])
        result["recovery"].append({"command": "python3 tools/plugin.py list",
                                    "instruction": "Use the enabled tool's declared input schema; do not bypass the Runtime."})
    elif route == "/v1/wiki/query":
        result["recovery"].append({"instruction":
            "Use python3 tools/sandbox.py --kind wiki-query --help (or wiki-search/wiki-hardware). "
            "Pass only workspace-relative input files and bounded max-bytes/max-records; "
            "do not specify private Wiki stores."})
    return result
