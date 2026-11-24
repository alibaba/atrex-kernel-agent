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
    "kernel_submit": "submit-kernel",
    "direction_prioritize": "prioritize-direction",
}
DIRECTION_TOOLS = {
    "direction_update", "direction_prioritize",
}
EXPERIMENT_TOOLS = {"experiment_record"}


def _text():
    return {"type": "string", "minLength": 1, "maxLength": 16000}


def _ids(prefix, *, minimum=0, maximum=64):
    return {"type": "array", "minItems": minimum, "maxItems": maximum, "uniqueItems": True,
            "items": {"type": "string", "pattern": f"^{prefix}[0-9a-f]{{32}}$"}}


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": sorted(required),
            "additionalProperties": False}


def journal_schema(operation, body, *, research_records=False, directions_enabled=None,
                   experiments_enabled=None):
    """The schema describes the JSON request file, not its HTTP envelope."""
    from aka.workflow.light.host.journal import (
        DIRECTION_PROPOSAL_FIELDS,
        DIRECTION_PROPOSAL_OPTIONAL_FIELDS,
        DIRECTION_REVISION_FIELDS,
        DIRECTION_CHANGE_SCOPES,
        DIRECTION_UPDATE_FIELDS,
        DIRECTION_CLOSURE_FIELDS,
        EXPERIMENT_FIELDS,
        EXPERIMENT_OPTIONAL_FIELDS,
        EXPERIMENT_EXPECTATION_FIELDS,
        EXPERIMENT_EXPECTATION_ASSESSMENTS,
    )
    from aka.workflow.light.host.direction_lifecycle import (
        DIRECTION_ACTIONS,
        DIRECTION_REVISION_ACTION,
        CLOSURES,
    )
    from aka.workflow.light.host.direction_genealogy import RELATIONSHIPS

    directions_enabled = research_records if directions_enabled is None else directions_enabled
    experiments_enabled = research_records if experiments_enabled is None else experiments_enabled
    if operation in DIRECTION_TOOLS and not directions_enabled:
        return {"available": False, "reason": "directions_disabled"}
    if operation in EXPERIMENT_TOOLS and not experiments_enabled:
        return {"available": False, "reason": "experiments_disabled"}
    if operation == "kernel_submit":
        return _object(
            {
                "kernel_commit_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{40}([0-9a-f]{24})?$",
                },
                "submission_id": {
                    "type": "string",
                    "pattern": "^submission-[0-9a-f]{32}$",
                    "description": (
                        "Omit to create a new Submission. Supply a Runtime-issued local "
                        "Submission ID to review the next immutable revision."
                    ),
                },
            },
            {"kernel_commit_id"},
        )
    if operation == "experiment_record":
        from aka.evaluation.evidence_fields import (
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
        }, {"query_id", "disposition"})
        usage["anyOf"] = [{"required": ["wiki_id"]}, {"required": ["cite"]}]
        fields = set(EXPERIMENT_FIELDS)
        properties = {key: _text() for key in fields | EXPERIMENT_OPTIONAL_FIELDS}
        properties["implementation"]["description"] = (
            "Concrete implementation, probe, or test method and its difference from prior attempts. "
            "Omission on revision preserves the previous value."
        )
        properties["next_step"] = {
            "anyOf": [_object({"action": _text(), "reason": _text()}, {"action", "reason"}), {"type": "null"}],
            "description": "Optional next action and rationale. Omit on revision to retain; null clears it. A plan, not an execution or lifecycle gate.",
        }
        expectation_properties = {key: _text() for key in EXPERIMENT_EXPECTATION_FIELDS}
        expectation_properties["assessment"]["enum"] = sorted(EXPERIMENT_EXPECTATION_ASSESSMENTS)
        properties["expectations"] = {
            "type": "array",
            "items": _object(expectation_properties, EXPERIMENT_EXPECTATION_FIELDS),
            "description": (
                "Scoped prediction/observation comparisons. Use unresolved when evidence is missing. "
                "Omission on revision preserves earlier rows; an explicit array replaces the rows "
                "in the new version. All frozen predictions are automatically imported from gateway_record_ids and retained on revision. Expectations are optional; no prediction index or repeated forecast is required. Unmatched Agent-authored expectations are retrospective. Earlier immutable versions remain archived."
            ),
        }
        properties["experiment_id"] = {
            "type": "string",
            "pattern": "^experiment_[0-9a-f]{32}$",
            "description": (
                "Omit to create a new Experiment. Supply an editable local Experiment ID "
                "to submit its next reviewed revision."
            ),
        }
        properties["references"] = {
            "type": "array",
            "minItems": 0,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 1024},
            "description": (
                "Workspace-relative files or directories to archive with this Experiment"
            ),
        }
        if directions_enabled:
            properties["direction_id"] = {
                "type": "string",
                "pattern": "^direction_[0-9a-f]{32}$",
                "description": (
                    "Intended local Direction. An explicitly named proposed Direction is valid "
                    "without requiring it to start first; omission is inferred only "
                    "when exactly one local Direction is in progress; otherwise it remains unassociated. "
                    "A revision may attach it to a local Direction; an existing binding is immutable."
                ),
            }
        properties.update(gateway_record_ids=_ids("gateway-", minimum=0),
                          wiki_usage={"type": "array", "maxItems": 64, "items": usage},
                          wiki_query_ids={"type": "array", "maxItems": 64, "items": query_id},
                          wiki_usage_status={"type": "string", "enum": sorted(WIKI_USAGE_STATUSES)})
        return _object(properties, fields)
    if operation == "direction_prioritize":
        schema = _object({
            "direction_id": {"type": "string", "pattern": "^direction_[0-9a-f]{32}$"},
            "before_direction_id": {"type": "string", "pattern": "^direction_[0-9a-f]{32}$"},
            "after_direction_id": {"type": "string", "pattern": "^direction_[0-9a-f]{32}$"},
            "reason": _text(),
        }, {"direction_id", "reason"})
        schema["not"] = {"required": ["before_direction_id", "after_direction_id"]}
        return schema
    if operation == "direction_update":
        action = body.get("action")
        if not isinstance(action, str) or action not in DIRECTION_ACTIONS:
            return {"oneOf": [journal_schema(operation, {"action": item}, research_records=True,
                                              directions_enabled=True, experiments_enabled=experiments_enabled)
                              for item in sorted(DIRECTION_ACTIONS)]}
        fields = (DIRECTION_PROPOSAL_FIELDS if action == "propose" else
                  DIRECTION_REVISION_FIELDS if action == DIRECTION_REVISION_ACTION else
                  DIRECTION_CLOSURE_FIELDS if action in CLOSURES else DIRECTION_UPDATE_FIELDS)
        properties = {key: _text() for key in fields}
        properties["action"] = {"const": action}
        if action in {"complete", "abandon"}:
            properties["action"]["description"] = (
                "Requires in_progress and a supported Review of this terminal decision. "
                "A partial or unavailable Review leaves the Direction in_progress."
            )
        elif action == "start":
            properties["action"]["description"] = (
                "Start proposed, deferred or blocked work without another Review; "
                "partially_supported is sufficient. analysis is optional; omitting it preserves the current assessment."
            )
            properties["analysis"] = {
                **_text(), "description": "Optional analysis; omit to preserve the existing assessment."
            }
        elif action in {"defer", "block"}:
            properties["action"]["description"] = (
                "Pause in_progress work with analysis and reopen_condition; "
                "partially_supported Review is sufficient."
            )
        if action in {"propose", DIRECTION_REVISION_ACTION}:
            properties["name"]["description"] = (
                "Short name for the optimization bottleneck or architecture path."
            )
            properties["change_scope"] = {
                "enum": sorted(DIRECTION_CHANGE_SCOPES),
                "description": (
                    "Scope of the Kernel path affected or unlocked by the objective; for a "
                    "probe-led Direction this is not the number of files touched by the probe."
                ),
            }
            properties["objective"]["description"] = (
                "Optimization bottleneck or architecture goal this Direction is responsible for "
                "resolving across its planned variants."
            )
            properties["applicability"]["description"] = (
                "Workload, Shape regime, semantic case, or hardware/compiler condition where "
                "the Direction applies."
            )
            properties["expected_benefit"]["description"] = (
                "Expected Kernel improvement or consequential implementation decision that this "
                "Direction can unlock."
            )
            for key in ("plan", "stop_conditions"):
                properties[key] = {"type": "array", "maxItems": 64,
                                   "items": {"type": "string", "minLength": 1, "maxLength": 4000}}
            properties["plan"]["description"] = (
                "Architecture families, implementation variants, and measurements to explore."
            )
            properties["stop_conditions"]["description"] = (
                "Optional array of non-empty conditions that make further work on this "
                "Direction no longer worthwhile."
            )
            if action == "propose":
                properties.update(relationship={"enum": list(RELATIONSHIPS)},
                                  derived_from_direction_ids=_ids("direction_", maximum=32),
                                  **({"derived_from_experiment_ids": _ids("experiment_", maximum=32)}
                                     if experiments_enabled else {}))
                if not experiments_enabled:
                    properties["gateway_record_ids"] = _ids(
                        "gateway-", minimum=0, maximum=64
                    )
            else:
                properties["analysis"] = {
                    **_text(),
                    "description": "Optional current assessment or interim findings. Omit to preserve the previous analysis; lifecycle status stays unchanged.",
                }
                properties["direction_id"] = {
                    "type": "string",
                    "pattern": "^direction_[0-9a-f]{32}$",
                    "description": (
                        "Editable local Direction identity. Proposed and active revisions receive "
                        "independent Review without changing lifecycle status."
                    ),
                }
                if not experiments_enabled:
                    properties["gateway_record_ids"] = _ids(
                        "gateway-", minimum=0, maximum=64
                    )
            for key in DIRECTION_PROPOSAL_OPTIONAL_FIELDS:
                properties.setdefault(key, _text())
        else:
            properties["direction_id"] = {"type": "string", "pattern": "^direction_[0-9a-f]{32}$"}
            if action in CLOSURES:
                properties["reopen_condition"] = _text()
                if experiments_enabled:
                    properties["supporting_experiment_ids"] = _ids(
                        "experiment_", minimum=0, maximum=32
                    )
                    properties["supporting_experiment_ids"]["description"] = (
                        "Optional measured Experiments that support this lifecycle decision."
                    )
                else:
                    properties["gateway_record_ids"] = _ids(
                        "gateway-", minimum=0, maximum=64
                    )
                    properties["gateway_record_ids"]["description"] = (
                        "Optional Gateway Record evidence that informed this Direction-only "
                        "lifecycle decision."
                    )
        return _object(properties, fields)
    return None


def gateway_schema(kind, parser, *, enabled_kinds=None):
    """Read option types/choices from the real parser; never expose its defaults."""
    groups = {
        "evaluate": {"--mode", "--version", "--shape-id", "--timed-runs", "--baseline-commit",
                "--comparison-repeats", "--comparison-run-timeout", "--input-path", "--shapes-path",
                "--warmup", "--candidate-timeout-s", "--perf-timeout-s", "--multi-seed",
                "--correctness-max-rel-l2"},
        "profile": {"--kernel-name", "--kernel-regex", "--profile-source", "--launch-skip", "--launch-count",
                    "--profile-shape-id", "--requirement", "--deps-mode", "--profile-level", "--profiler",
                    "--profile-counter", "--top-kernels", "--include-raw-profile"},
        "check": {"--arch", "--sanitize", "--requirement", "--deps-mode"},
        "disassemble": {"--format", "--requirement", "--deps-mode"},
        "dev": set(), "command": set(),
        "env": {"--env-gpu", "--env-capabilities", "--env-force", "--force",
                "--frameworks", "--limits", "--counters"},
    }
    if enabled_kinds is not None:
        groups = {key: value for key, value in groups.items() if key in enabled_kinds}
    if kind not in groups:
        return {"supported_kinds": list(groups), "instruction": "Use an explicit --kind and full option names."}
    allowed = groups[kind] | {"--kind"}
    if kind in {"evaluate", "profile", "dev", "command", "check", "disassemble"}:
        allowed |= {"--input", "--sync", "--no-sync", "--env", "--inline-output", "--keep-pod", "--dry-run"}
    options = {}
    from aka.execution.gateway.native_options import COMMAND_BY_KIND
    native_command = COMMAND_BY_KIND.get(kind)
    for action in parser._actions:
        native_for_kind = native_command in getattr(action, "agate_commands", ())
        for flag in action.option_strings:
            if flag not in allowed and not native_for_kind:
                continue
            value_type = getattr(action, "agate_value_type", action.type)
            schema_type = getattr(action, "agate_schema_type", None)
            item = {"type": schema_type or (
                "boolean" if action.nargs == 0 else
                "integer" if value_type is int else
                "number" if value_type is float else "string"
            )}
            if action.choices is not None:
                item["enum"] = list(action.choices)
            if isinstance(action, argparse._AppendAction):
                item["repeatable"] = True
            if native_for_kind and flag.startswith("--"):
                item["source"] = "agate"
            if isinstance(action.help, str) and action.help:
                item["description"] = action.help
            options[flag] = item
    options["--kind"] = {"const": kind}
    required = ["--kind"]
    schema = {"cli_options": options, "required_options": required}
    command_contract = {
        "evaluate": (
            "Optional argv after --; without it Runtime executes the trusted "
            "private evaluator."
        ),
        "profile": (
            "Optional argv after --; without it Runtime profiles the current Kernel "
            "using the standard campaign input."
        ),
        "dev": (
            "Required custom probe argv after --. Stateless Dev is not official Evaluate "
            "evidence; declare every extra workspace input with --input and request returned "
            "files with --sync."
        ),
        "command": (
            "Required argv after --; uploads a bounded Agent-visible workspace engineering "
            "projection and synchronizes scratch/ back."
        ),
    }
    if kind in command_contract:
        schema["command"] = command_contract[kind]
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


def augment_error(response, *, route, request, research_records=False, directions_enabled=None,
                  experiments_enabled=None, parser=None, catalog=(), enabled_kinds=None):
    """Add only public format guidance; preserve authoritative recovery/classification."""
    directions_enabled = research_records if directions_enabled is None else directions_enabled
    experiments_enabled = research_records if experiments_enabled is None else experiments_enabled
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
        schema = journal_schema(operation, body, research_records=research_records,
                                directions_enabled=directions_enabled,
                                experiments_enabled=experiments_enabled)
        if schema is not None:
            result["request_schema"] = schema
            issues = field_issues(body, schema)
            if issues:
                # Schema hints supplement, never hide, the validator's semantic
                # finding (e.g. missing evidence plus a missing Commit field).
                specific = [item for item in result.get("issues", []) if item.get("path") != "$"]
                result["issues"] = (specific + [item for item in issues if item not in specific])[:32]
        else:
            result["supported_operations"] = [
                op for op in JOURNAL_TOOLS
                if (op not in DIRECTION_TOOLS or directions_enabled)
                and (op not in EXPERIMENT_TOOLS or experiments_enabled)
            ]
        if operation in JOURNAL_TOOLS:
            result["operation"] = JOURNAL_TOOLS[operation]
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
        result["request_schema"] = gateway_schema(kind if isinstance(kind, str) else None, parser,
                                                  enabled_kinds=enabled_kinds)
    elif route == "/v1/plugins/execute":
        name = request.get("tool")
        tool = next((item for item in catalog if item["name"] == name), None)
        if tool is not None:
            result["request_schema"] = deepcopy(tool["input_schema"])
        result.setdefault("recovery", []).append({
            "command": "python3 tools/plugin.py list",
            "instruction": "Use the enabled tool's declared input schema; do not bypass the Runtime.",
        })
    elif route == "/v1/wiki/query":
        result.setdefault("recovery", []).append({"instruction":
            "Use python3 tools/sandbox.py --kind wiki-query --help (or wiki-search/wiki-hardware). "
            "Pass only workspace-relative input files and bounded max-bytes/max-records; "
            "do not specify private Wiki stores."})
    return result
