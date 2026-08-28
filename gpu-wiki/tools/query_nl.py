#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Fast natural-language front door to the GPU knowledge stores.

``query_bridge_agent`` performs one small task: turn prose into a typed intent.
Supported CLIs must expose an explicit no-tools, no-MCP JSON protocol; Claude
and Qoder currently meet that contract. The bridge is forbidden to inspect or
query any store. This script owns every
deterministic operation after that point: vocabulary normalization, staged
widening, execution, deduplication, projection and context limits.

The output has a per-invocation ``query_id``, plus records and notes. Records
from kernel experience and hardware facts share one backward-compatible id-keyed
mapping, while every value emits its canonical ``store::record`` ``wiki_id``.
The public store is always queried; an installed sibling ``internal_gpu_wiki``
is queried as a second isolated store. Internal mapping keys are namespaced so a
private record can never overwrite a public record with the same stable id.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import agent_launch
import hardware_identity
import operator_scope
import query_wiki

HERE = Path(__file__).resolve().parent
OWN_STORE_ROOT = HERE.parent
SIBLING_INTERNAL = OWN_STORE_ROOT.parent / "internal_gpu_wiki"
PUBLIC_STORE = "gpu_wiki"
INTERNAL_STORE = "internal_gpu_wiki"
STORE_ENV = "ATREX_WIKI_STORE_ROOT"
BRIDGE_CLI_ENV = "ATREX_WIKI_BRIDGE_CLI"
METRICS_LOG_ENV = "ATREX_WIKI_METRICS_LOG"
TASK_ID_ENV = "ATREX_WIKI_TASK_ID"

INTENT_FILE = "query_intent.json"
# Compatibility for callers that only need the handoff filename.
PLAN_FILE = INTENT_FILE
DEFAULT_MAX_RECORDS = 20
ENOUGH_RECORDS = 8
MAX_TERMS = 6
MAX_COMPONENT_TERMS = 6
MAX_OPERATOR_SCOPES = 8

INTENTS = {"technique", "pitfall", "documentation", "diagnosis", "correctness"}
HARDWARE_KINDS = {"product", "instruction", "feature"}
INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "architecture", "vendor", "dsl", "operator_terms", "component_terms",
        "measured_symptoms", "free_text_terms", "intents", "hardware_requests",
    ],
    "properties": {
        "architecture": {"type": ["string", "null"]},
        "vendor": {"type": ["string", "null"]},
        "dsl": {"type": ["string", "null"]},
        "operator_terms": {
            "type": "array", "maxItems": MAX_TERMS, "items": {"type": "string"},
        },
        "component_terms": {
            "type": "array", "maxItems": MAX_COMPONENT_TERMS,
            "items": {"type": "string"},
        },
        "measured_symptoms": {
            "type": "array", "maxItems": 2, "items": {"type": "string"},
        },
        "free_text_terms": {
            "type": "array", "maxItems": MAX_TERMS, "items": {"type": "string"},
        },
        "intents": {
            "type": "array", "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(INTENTS)},
        },
        "hardware_requests": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["kind", "value", "field", "vs"],
                "properties": {
                    "kind": {"type": "string", "enum": sorted(HARDWARE_KINDS)},
                    "value": {"type": "string"},
                    "field": {"type": ["string", "null"]},
                    "vs": {"type": ["string", "null"]},
                },
            },
        },
    },
}

PLAIN_JSON_BRIDGE_PROMPT = """You are query_bridge_agent, a fast store-blind intent extractor.

Return exactly one JSON object, with no markdown or prose, in this shape:
{{"architecture":string|null,"vendor":string|null,"dsl":string|null,
"operator_terms":[string],"component_terms":[string],
"measured_symptoms":[string],
"free_text_terms":[string],
"intents":["technique"|"pitfall"|"documentation"|"diagnosis"|"correctness"],
"hardware_requests":[{{"kind":"product"|"instruction"|"feature",
"value":string,"field":string|null,"vs":string|null}}]}}

Include every key exactly once. Do not research, invoke tools,
inspect a store, invent missing scope values, explain the result, or generate
query flags. Copy architecture/vendor/DSL/product spellings from the request.
The architecture slot is only for a GPU architecture or public GPU product
(for example sm_100, Blackwell, or B200), never a model/operator acronym such
as GDN. A target product is query scope, not a hardware_request unless the
caller also asks for hardware specifications. Use caller-language
operator/API/mechanism phrases. Put the requested operator
or fused/composite operation in operator_terms. Put independently queryable
sub-operations in component_terms, preserving the caller's words. Decompose a
clearly composite name (for example QK norm + RoPE + KV-cache write). Do not
research or infer hidden model structure; a deterministic resolver handles
established cross-operator relationships. Do not invent implementation-specific
components when the decomposition is uncertain.
A measured symptom requires
an explicit profile counter or number; keep hypotheses in free_text_terms. Emit
at most 6 operator_terms, at most 6 component_terms, exactly 0-2 measured_symptoms, at most 6
free_text_terms, and at most 4 hardware_requests. Raw counters are evidence for
classification, not separate symptoms: when many counters are present, select
no more than two short diagnosis labels such as register-pressure or tail-effect.
Hardware requests are only for specifications, peak/roofline values, ISA,
architecture features, or product comparisons. Never invent a hardware `field`
path: set field to null unless the caller supplied the exact literal dot-path.
For several facts about one product, emit one product request with field null,
not one request per fact. Do not duplicate a hardware address. When the request
states a measured diagnosis label such as memory-bound, register-pressure, or
tail-effect, copy that exact label rather than a raw counter sentence. Use null
and empty arrays for missing information. Finish in this single response.

<<<REQUEST
{request}
REQUEST>>>
"""

FAST_BRIDGE_RE = re.compile(
    r"^\s*Target\s+hardware\s+(?P<hardware>[A-Za-z0-9_-]+)\s*,\s*"
    r"DSL\s+(?P<dsl>[A-Za-z0-9_+.-]+)\s*\.\s*"
    r"Optimize\s+operator\s+(?P<operator>[A-Za-z0-9_+.-]+)"
    r"(?:\s+and\s+retrieve\s+(?P<retrieve>techniques(?:\s+and\s+pitfalls)?|pitfalls))?"
    r"\s*\.\s*$",
    re.IGNORECASE,
)


def deterministic_bridge_intent(request: str) -> dict | None:
    """Handle the optimizer's narrow standard request without launching an LLM."""
    match = FAST_BRIDGE_RE.fullmatch(request)
    if match is None:
        return None
    requested = (match.group("retrieve") or "techniques").casefold()
    intents = ["technique"] if "techniques" in requested else []
    if "pitfalls" in requested:
        intents.append("pitfall")
    doc = {
        "architecture": match.group("hardware"),
        "vendor": None,
        "dsl": match.group("dsl"),
        "operator_terms": [match.group("operator")],
        "component_terms": [],
        "measured_symptoms": [],
        "free_text_terms": [],
        "intents": intents,
        "hardware_requests": [],
    }
    strictly_validate_intent(doc)
    return validate_intent(doc)

REPAIR_SUFFIX = """

Your prior response failed strict local validation: {error}
Return a corrected JSON object now. Output only JSON; include every required key
and no additional keys.
"""

def _append_metric(path: str | None, metric: dict) -> None:
    """Append one compact JSONL event without changing the query result."""
    if not path:
        return
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(metric, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError as exc:
        print("WARNING wiki metric could not be written: %s" % exc, file=sys.stderr)


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    print("ERROR %s" % msg, file=sys.stderr)
    raise SystemExit(code)


def resolve_store_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif os.environ.get(STORE_ENV):
        root = Path(os.environ[STORE_ENV]).expanduser().resolve()
    else:
        root = OWN_STORE_ROOT
    for name in ("query_wiki.py", "query_hardware.py"):
        if not (root / "tools" / name).is_file():
            die("bad-store-root %s has no tools/%s" % (root, name))
    return root


def _selected_store_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get(STORE_ENV):
        return Path(os.environ[STORE_ENV]).expanduser().resolve()
    return OWN_STORE_ROOT


def _queryable_store(root: Path) -> bool:
    """A placeholder directory is not an installed Wiki store."""
    required = (
        root / "tools" / "query_wiki.py",
        root / "tools" / "query_hardware.py",
        root / "kernel_wiki" / "records" / "index.json",
        root / "hardware_wiki" / "records" / "index.json",
    )
    return root.is_dir() and all(path.is_file() for path in required)


def resolve_store_roots(explicit: str | None) -> list[tuple[str, Path]]:
    """Return the two fixed Wiki module slots without pre-classifying availability."""
    primary = _selected_store_root(explicit)
    internal = (
        primary.parent / INTERNAL_STORE
        if explicit or os.environ.get(STORE_ENV)
        else SIBLING_INTERNAL
    ).expanduser().resolve()
    return [(PUBLIC_STORE, primary), (INTERNAL_STORE, internal)]


def _string(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _strings(value, limit: int = MAX_TERMS) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = _string(item)
        if text and text not in out:
            out.append(text)
    return out[:limit]


def validate_intent(doc: object) -> dict:
    """Accept only semantic slots; executable tools and flags are impossible."""
    if not isinstance(doc, dict):
        die("bad-intent top level must be an object")
    forbidden = {"queries", "flags", "tool", "reading_guide"} & set(doc)
    if forbidden:
        die("bad-intent executable/research fields are forbidden: %s"
            % ", ".join(sorted(forbidden)))
    intents = [x for x in _strings(doc.get("intents")) if x in INTENTS]
    hardware = doc.get("hardware_requests") or []
    if not isinstance(hardware, list):
        die("bad-intent hardware_requests must be a list")
    clean_hw = []
    for i, request in enumerate(hardware, 1):
        if not isinstance(request, dict):
            die("bad-intent hardware_requests[%d] must be an object" % i)
        kind, value = _string(request.get("kind")), _string(request.get("value"))
        if kind not in HARDWARE_KINDS or not value:
            die("bad-intent hardware request needs kind product/instruction/feature and value")
        clean_hw.append({
            "kind": kind, "value": value,
            "field": _string(request.get("field")),
            "vs": _string(request.get("vs")),
        })
    deduped_hw = []
    positions = {}
    for request in clean_hw:
        address = (
            request["kind"], request["value"].casefold(),
            (request["vs"] or "").casefold(),
        )
        if address not in positions:
            positions[address] = len(deduped_hw)
            deduped_hw.append(request)
            continue
        previous = deduped_hw[positions[address]]
        if previous != request and request["kind"] == "product":
            # Several facts about one product are served more safely by one
            # complete spec sheet than by model-invented field paths.
            previous["field"] = None
    return {
        "architecture": _string(doc.get("architecture")),
        "vendor": _string(doc.get("vendor")),
        "dsl": _string(doc.get("dsl")),
        "operator_terms": _strings(doc.get("operator_terms")),
        "component_terms": _strings(doc.get("component_terms"), MAX_COMPONENT_TERMS),
        "measured_symptoms": _strings(doc.get("measured_symptoms"), 2),
        "free_text_terms": _strings(doc.get("free_text_terms")),
        "intents": intents,
        "hardware_requests": deduped_hw[:4],
    }


def strictly_validate_intent(doc: object) -> dict:
    """Validate the plain-JSON bridge contract without coercion or data loss."""
    if not isinstance(doc, dict):
        raise ValueError("top level must be an object")
    expected = set(INTENT_SCHEMA["required"]) - {"component_terms"}
    allowed = set(INTENT_SCHEMA["properties"])
    actual = set(doc)
    missing = sorted(expected - actual)
    extra = sorted(actual - allowed)
    if missing:
        raise ValueError("missing keys: %s" % ", ".join(missing))
    if extra:
        raise ValueError("unexpected keys: %s" % ", ".join(extra))
    # Legacy bridges predate decomposition-aware retrieval. Missing component
    # terms mean "no explicit decomposition", not an invalid request.
    doc.setdefault("component_terms", [])

    for key in ("architecture", "vendor", "dsl"):
        if doc[key] is not None and not isinstance(doc[key], str):
            raise ValueError("%s must be a string or null" % key)
        if isinstance(doc[key], str) and not doc[key].strip():
            raise ValueError("%s must be non-empty or null" % key)

    list_limits = {
        "operator_terms": MAX_TERMS,
        "component_terms": MAX_COMPONENT_TERMS,
        "measured_symptoms": 2,
        "free_text_terms": MAX_TERMS,
    }
    for key, limit in list_limits.items():
        value = doc[key]
        if not isinstance(value, list):
            raise ValueError("%s must be an array" % key)
        if len(value) > limit:
            raise ValueError("%s has more than %d items" % (key, limit))
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("%s must contain only non-empty strings" % key)

    intents = doc["intents"]
    if not isinstance(intents, list):
        raise ValueError("intents must be an array")
    if any(not isinstance(item, str) or item not in INTENTS for item in intents):
        raise ValueError("intents contains an unsupported value")
    if len(intents) != len(set(intents)):
        raise ValueError("intents must not contain duplicates")

    hardware = doc["hardware_requests"]
    if not isinstance(hardware, list):
        raise ValueError("hardware_requests must be an array")
    if len(hardware) > 4:
        raise ValueError("hardware_requests has more than 4 items")
    hardware_keys = {"kind", "value", "field", "vs"}
    for index, request in enumerate(hardware):
        label = "hardware_requests[%d]" % index
        if not isinstance(request, dict):
            raise ValueError("%s must be an object" % label)
        if set(request) != hardware_keys:
            raise ValueError("%s must contain exactly kind, value, field, vs" % label)
        if request["kind"] not in HARDWARE_KINDS:
            raise ValueError("%s.kind is unsupported" % label)
        if not isinstance(request["value"], str) or not request["value"].strip():
            raise ValueError("%s.value must be a non-empty string" % label)
        for key in ("field", "vs"):
            if request[key] is not None and not isinstance(request[key], str):
                raise ValueError("%s.%s must be a string or null" % (label, key))
            if isinstance(request[key], str) and not request[key].strip():
                raise ValueError("%s.%s must be non-empty or null" % (label, key))
    return doc


def bridge_prompt(request: str, store_root: Path, workspace: Path,
                  skill_path: Path | None, exclude: str | None = None,
                  plain_json: bool = False) -> str:
    del store_root, workspace, skill_path, exclude, plain_json
    return PLAIN_JSON_BRIDGE_PROMPT.format(request=request)


def parse_bridge_json_output(stdout: str) -> tuple[dict, dict]:
    """Extract plain JSON plus telemetry from a supported CLI result envelope."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("bridge stdout is not a JSON result envelope: %s" % exc) from exc
    if not isinstance(envelope, dict):
        raise ValueError("bridge stdout result envelope must be an object")
    payload = envelope.get("result")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("bridge result is not plain JSON: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise ValueError("bridge result JSON must be an object")
    return payload, _bridge_telemetry(envelope)


def parse_claude_json_output(stdout: str) -> tuple[dict, dict]:
    """Compatibility alias for callers predating the multi-CLI JSON bridge."""
    return parse_bridge_json_output(stdout)


def _bridge_telemetry(envelope: dict) -> dict:
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
    return {
        "bridge_duration_api_ms": envelope.get("duration_api_ms"),
        "bridge_ttft_ms": envelope.get("ttft_ms"),
        "bridge_num_turns": envelope.get("num_turns"),
        "bridge_input_tokens": usage.get("input_tokens"),
        "bridge_output_tokens": usage.get("output_tokens"),
        "bridge_cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "bridge_cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
    }


def _merge_bridge_telemetry(total: dict, current: dict) -> None:
    """Accumulate retry costs while retaining first-attempt TTFT semantics."""
    if "bridge_ttft_ms" not in total and current.get("bridge_ttft_ms") is not None:
        total["bridge_ttft_ms"] = current["bridge_ttft_ms"]
    for key, value in current.items():
        if key == "bridge_ttft_ms" or not isinstance(value, (int, float)):
            continue
        total[key] = total.get(key, 0) + value


def read_request(args) -> str:
    if args.request:
        text = " ".join(args.request)
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        die("no request: pass it as an argument, with --file, or on stdin")
    text = text.strip()
    if not text:
        die("empty request")
    return text


def _resolve_vocab_all(value: str | None, allowed: set[str]) -> list[str]:
    if not value:
        return []
    raw = value.strip().casefold()
    exact = sorted(token for token in allowed if token.casefold() == raw)
    equivalent = sorted(
        token for token in allowed
        if token not in exact and query_wiki.fold(token) == query_wiki.fold(value)
    )
    return exact + equivalent


def _resolve_vocab(value: str | None, allowed: set[str]) -> str | None:
    matches = _resolve_vocab_all(value, allowed)
    return matches[0] if matches else None


def _resolve_arch(value: str | None, allowed: set[str]) -> str | None:
    """Resolve runtime spelling without printing the unified query CLI error channel."""
    if not value:
        return None
    token = value.strip().lower()
    if token in allowed:
        return token
    family = (query_wiki.ARCH_ALIASES.get(token)
              or query_wiki.ARCH_ALIASES.get(token.replace("sm_", "sm"))
              or hardware_identity.PRODUCT_ARCH.get(
                  hardware_identity.normalize_product_name(value)))
    return family if family in allowed else None


def normalize_intent(intent: dict, store_root: Path,
                     request_text: str | None = None) -> tuple[dict, list[str]]:
    index = query_wiki.load_index(store_root / "kernel_wiki")
    vv = query_wiki.vocab(index["records"])
    notes: list[str] = []
    arch = _resolve_arch(intent["architecture"], vv["arch"])
    if intent["architecture"]:
        if not arch:
            notes.append("architecture %r is not represented by this store; kernel lookup remains unscoped"
                         % intent["architecture"])
    else:
        notes.append("no runtime architecture was supplied; kernel matches are not architecture-pinned")
    vendor = _resolve_vocab(intent["vendor"], vv["vendor"])
    # Only the Bridge's typed runtime-hardware slot may pin a kernel query to a
    # product. Product names in hardware fact requests or surrounding prose are
    # independent fact lookups and must never override an explicit runtime arch.
    # ``request_text`` remains in the signature for callers on the earlier API;
    # it is deliberately not inspected for scope inference.
    runtime_product = hardware_identity.normalize_product_name(
        intent["architecture"] or "")
    if runtime_product not in hardware_identity.HARDWARE_IDENTITIES:
        runtime_product = None
    if runtime_product:
        product_arch = hardware_identity.PRODUCT_ARCH[runtime_product]
        product_vendor = hardware_identity.HARDWARE_IDENTITIES[runtime_product]["vendor"]
        if product_arch in vv["arch"] and arch != product_arch:
            arch = product_arch
            notes.append(
                "runtime scope was pinned from explicit product %s to architecture %s"
                % (runtime_product, product_arch)
            )
        if product_vendor in vv["vendor"] and vendor != product_vendor:
            vendor = product_vendor
    product_queryable = bool(
        runtime_product and runtime_product in vv.get("product", set())
    )
    arch_query = (
        runtime_product
        if runtime_product in query_wiki.ARCH_ALIASES else arch
    )
    if runtime_product and not product_queryable:
        notes.append(
            "product %s has no product-scoped kernel records; lookup starts at architecture level"
            % runtime_product
        )
    dsl_scopes = _resolve_vocab_all(intent["dsl"], vv["dsl"])
    dsl = dsl_scopes[0] if dsl_scopes else None
    if intent["dsl"] and not dsl:
        notes.append("DSL %r is not a store scope token and was kept as free text"
                     % intent["dsl"])
    elif len(dsl_scopes) > 1:
        notes.append(
            "DSL spelling maps to equivalent isolated scopes: %s"
            % ", ".join(dsl_scopes)
        )

    resolver = operator_scope.OperatorScopeResolver.from_store(store_root)
    operator_scopes, unresolved_operator_terms = resolver.resolve_many(
        intent["operator_terms"], intent["component_terms"],
        limit=MAX_OPERATOR_SCOPES,
    )
    preferred = next(
        (scope for scope in operator_scopes if scope.role == "primary"),
        operator_scopes[0] if operator_scopes else None,
    )
    operator = (preferred.value
                if preferred is not None and preferred.axis == "operator" else None)
    family = (preferred.value
              if preferred is not None and preferred.axis == "family" else None)
    spill = list(unresolved_operator_terms)
    voted = [scope for scope in operator_scopes if scope.confidence == "voted"]
    related = [scope for scope in operator_scopes if scope.role == "related"]
    if voted:
        notes.append(
            "operator naming was mapped from the Store corpus: %s"
            % ", ".join("%s=%s" % (scope.axis, scope.value) for scope in voted)
        )
    if related:
        notes.append(
            "operator decomposition added %d isolated related scope(s)" % len(related)
        )
    resolved_symptoms = [
        (raw, _resolve_vocab(raw, vv["symptom"]))
        for raw in intent["measured_symptoms"]
    ]
    symptom = next((resolved for _, resolved in resolved_symptoms if resolved), None)
    for raw, resolved in resolved_symptoms:
        if not resolved or resolved != symptom:
            spill.append(raw)
    if intent["measured_symptoms"] and not symptom:
        notes.append("measured symptom did not match an indexed token and was kept as free text")
    terms = []
    for term in spill + intent["free_text_terms"] + ([intent["dsl"]] if intent["dsl"] and not dsl else []):
        if term and term not in terms:
            terms.append(term)
    # Never allow an empty text query to flood the caller when semantic operator
    # axes were unavailable. The structured axes alone remain a valid query.
    hardware_requests = []
    for request in intent["hardware_requests"]:
        normalized_request = dict(request)
        if request["kind"] == "feature":
            feature_product = hardware_identity.normalize_product_name(request["value"])
            if feature_product in hardware_identity.HARDWARE_IDENTITIES:
                normalized_request.update(
                    kind="product", value=feature_product, field=None,
                )
                notes.append(
                    "hardware identity %r was classified as a feature; treated as product %s"
                    % (request["value"], feature_product)
                )
            elif _resolve_arch(request["value"], vv["arch"]):
                notes.append(
                    "runtime architecture %r was classified as a feature and was ignored"
                    % request["value"]
                )
                continue
        if request["kind"] == "product":
            normalized_request["value"] = hardware_identity.normalize_product_name(
                request["value"])
            if request["vs"]:
                normalized_request["vs"] = hardware_identity.normalize_product_name(
                    request["vs"])
        address = (
            normalized_request["kind"],
            normalized_request["value"].casefold(),
            (normalized_request.get("vs") or "").casefold(),
        )
        duplicate = next((row for row in hardware_requests if (
            row["kind"], row["value"].casefold(),
            (row.get("vs") or "").casefold(),
        ) == address), None)
        if duplicate is None:
            hardware_requests.append(normalized_request)
        elif duplicate["kind"] == "product" and duplicate.get("field") != normalized_request.get("field"):
            duplicate["field"] = None
    return dict(intent, arch=arch, arch_query=arch_query, vendor=vendor,
                product=runtime_product, product_queryable=product_queryable,
                dsl=dsl, dsl_scopes=dsl_scopes, operator=operator,
                family=family, operator_scopes=operator_scopes,
                symptom=symptom, terms=terms[:MAX_TERMS],
                hardware_requests=hardware_requests), notes


def _kernel_flags(intent: dict, *, drop_operator=False, drop_symptom=False,
                  any_terms=False, cross_arch=False, limit=8,
                  scope: operator_scope.OperatorScope | None = None,
                  drop_terms: bool = False,
                  drop_product: bool = False,
                  dsl_scope: str | None = None,
                  exclude: str | None = None, max_bytes: int | None = None) -> list[str]:
    flags: list[str] = []
    arch_value = (intent["arch"] if drop_product
                  else intent.get("arch_query") or intent["arch"])
    for flag, value in (("--arch", arch_value),
                        ("--vendor", intent["vendor"]),
                        ("--dsl", dsl_scope if dsl_scope is not None else intent["dsl"])):
        if value:
            flags += [flag, value]
    if not drop_operator:
        if scope is not None:
            flags += ["--" + scope.axis, scope.value]
        elif intent["operator"]:
            flags += ["--operator", intent["operator"]]
        elif intent["family"]:
            flags += ["--family", intent["family"]]
    if intent["symptom"] and not drop_symptom:
        flags += ["--symptom", intent["symptom"]]
    terms = [] if drop_terms else intent["terms"]
    flags += terms
    if any_terms and terms:
        flags.append("--any")
    if cross_arch and intent["arch"]:
        flags.append("--cross-arch")
    flags += ["--emit-json", "--no-fallback", "--limit", str(max(1, limit))]
    if exclude:
        flags += ["--exclude", exclude]
    if max_bytes:
        flags += ["--max-bytes", str(max_bytes)]
    return flags


def _run_json(argv: list[str]) -> tuple[int, object | None, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    payload = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return proc.returncode, payload, (proc.stderr or "").strip()


def _project_kernel_record(rid: str, entry: object) -> tuple[str, dict] | None:
    """Normalize public and legacy-internal results to one bias-free projection."""
    if not isinstance(entry, dict):
        return None
    if isinstance(entry.get("payload"), dict):
        return rid, {
            "source": "kernel_wiki",
            "type": entry.get("type"),
            "applies_to": entry.get("applies_to") or {},
            "match": entry.get("match") or {},
            "payload": entry["payload"],
        }
    nested = entry.get("records")
    if not isinstance(nested, dict) or not isinstance(nested.get("payload"), dict):
        return None
    stable_id = str(nested.get("id") or rid)
    return stable_id, {
        "source": "kernel_wiki",
        "type": nested.get("type"),
        "applies_to": entry.get("applies_to") or {},
        "match": entry.get("match") or {},
        "payload": nested["payload"],
    }


def _query_kernel_single(
    store_root: Path,
    intent: dict,
    records: dict,
    notes: list[str],
    max_records: int,
    exclude: str | None,
    max_bytes: int | None,
    *,
    scope: operator_scope.OperatorScope | None = None,
    report_empty: bool = True,
) -> None:
    if scope is not None:
        # A decomposition lane may relax its text and architecture, but never
        # drops its operator/family scope.  That keeps components isolated and
        # prevents an empty lane from filling its quota with unrelated records.
        stages = [
            ("exact scope", {"scope": scope}),
        ]
        if intent.get("product_queryable"):
            stages.append((
                "same operator scope at architecture level",
                {"scope": scope, "drop_product": True},
            ))
        stages.append((
            "same operator scope with text removed",
            {"scope": scope, "drop_terms": True},
        ))
        if intent.get("product_queryable"):
            stages.append((
                "same operator scope with text removed at architecture level",
                {"scope": scope, "drop_terms": True, "drop_product": True},
            ))
        if intent["arch"]:
            stages.append(("same operator scope on sibling architectures",
                           {"scope": scope, "drop_terms": True,
                            "drop_product": True, "cross_arch": True}))
    else:
        stages = [
            ("exact scope", {}),
        ]
        if intent.get("product_queryable"):
            stages.append(("architecture-level scope",
                           {"drop_product": True}))
        stages.extend([
            ("symptom scope removed", {"drop_symptom": True}),
            ("text terms widened to OR", {"drop_symptom": True,
                                           "any_terms": True}),
        ])
        if intent["arch"]:
            stages.append(("same-vendor sibling architectures included",
                           {"drop_symptom": True, "drop_product": True,
                            "any_terms": True, "cross_arch": True}))
    seen_flags = set()
    dsl_scopes = list(intent.get("dsl_scopes") or [])
    if not dsl_scopes:
        dsl_scopes = [intent.get("dsl")]
    for label, options in stages:
        for current_dsl in dsl_scopes:
            remaining = max_records - len(records) if max_records else DEFAULT_MAX_RECORDS
            if max_records and remaining <= 0:
                notes.append("kernel results were truncated at the %d-record cap" % max_records)
                break
            flags = _kernel_flags(
                intent, limit=min(8, max(1, remaining)), exclude=exclude,
                max_bytes=max_bytes, dsl_scope=current_dsl, **options,
            )
            signature = tuple(flags)
            if signature in seen_flags:
                continue
            seen_flags.add(signature)
            code, payload, error = _run_json(
                [sys.executable, str(store_root / "tools" / "query_wiki.py")] + flags)
            if code != 0 or not isinstance(payload, dict):
                notes.append("kernel lookup failed at %s: %s" %
                             (label, (error.splitlines() or ["invalid output"])[0][:240]))
                continue
            found = payload.get("records") or {}
            for raw_rid, raw_entry in found.items():
                projected = _project_kernel_record(raw_rid, raw_entry)
                if projected is None:
                    continue
                rid, entry = projected
                if rid not in records:
                    entry["match"] = dict(entry.get("match") or {})
                    if scope is not None:
                        entry["match"].update({
                            "operator_role": scope.role,
                            "operator_scope": "%s:%s" % (scope.axis, scope.value),
                            "operator_confidence": scope.confidence,
                        })
                    if current_dsl:
                        entry["match"]["dsl_scope"] = current_dsl
                    if intent.get("product"):
                        entry["match"].update({
                            "requested_product": intent["product"],
                            "product_scope": (
                                "architecture-fallback"
                                if (options.get("drop_product")
                                    or not intent.get("product_queryable"))
                                else "exact"
                            ),
                        })
                    records[rid] = entry
                    if max_records and len(records) >= max_records:
                        break
            if found and label != "exact scope":
                note = "kernel lookup widened: %s" % label
                if note not in notes:
                    notes.append(note)
            if found and current_dsl != intent.get("dsl"):
                note = "kernel lookup used equivalent DSL spelling %s" % current_dsl
                if note not in notes:
                    notes.append(note)
            if scope is not None and found:
                # One successful scoped query is enough for this lane.
                return
            kernel_count = sum(
                r.get("source") != "hardware_wiki" for r in records.values()
            )
            target = min(ENOUGH_RECORDS, max_records or ENOUGH_RECORDS)
            if kernel_count >= target:
                return
    if report_empty and not any(
        r.get("source") != "hardware_wiki" for r in records.values()
    ):
        notes.append("the kernel Wiki returned no matching records")


def _lane_budgets(scopes: list[operator_scope.OperatorScope], total: int) -> list[int]:
    """Give primary lanes twice the quota without starving related components."""
    if not scopes:
        return []
    weights = [2 if scope.role == "primary" else 1 for scope in scopes]
    total_weight = sum(weights)
    budgets = [max(1, total * weight // total_weight) for weight in weights]
    while sum(budgets) > total and any(value > 1 for value in budgets):
        index = max(range(len(budgets)), key=lambda i: budgets[i])
        budgets[index] -= 1
    index = 0
    while sum(budgets) < total:
        budgets[index % len(budgets)] += 1
        index += 1
    return budgets


def query_kernel(store_root: Path, intent: dict, records: dict, notes: list[str],
                 max_records: int, exclude: str | None,
                 max_bytes: int | None) -> None:
    scopes = list(intent.get("operator_scopes") or [])
    available = max(0, max_records - len(records)) if max_records else DEFAULT_MAX_RECORDS
    cap = len(records) + available
    requested_operator_scope = bool(
        intent.get("operator_terms") or intent.get("component_terms")
    )
    if not scopes and requested_operator_scope:
        notes.append(
            "operator name could not be mapped unambiguously; unscoped text widening was suppressed"
        )
        return
    if not scopes or available <= 0:
        _query_kernel_single(
            store_root, intent, records, notes, max_records, exclude, max_bytes,
        )
        return

    scopes = scopes[:available]
    groups: list[tuple[operator_scope.OperatorScope, dict[str, dict]]] = []
    lane_notes: list[str] = []
    for scope, budget in zip(scopes, _lane_budgets(scopes, available)):
        lane_records: dict[str, dict] = {}
        _query_kernel_single(
            store_root, intent, lane_records, lane_notes, budget, exclude,
            max_bytes, scope=scope, report_empty=False,
        )
        groups.append((scope, lane_records))

    # Round-robin merge preserves payload isolation and prevents the first
    # primary/component from consuming the global cap.
    pending = [(scope, iter(group.items())) for scope, group in groups]
    exhausted: set[int] = set()
    while len(exhausted) < len(pending) and len(records) < cap:
        progressed = False
        for index, (_scope, iterator) in enumerate(pending):
            if index in exhausted:
                continue
            try:
                rid, entry = next(iterator)
            except StopIteration:
                exhausted.add(index)
                continue
            progressed = True
            records.setdefault(rid, entry)
            if len(records) >= cap:
                break
        if not progressed:
            break

    matched_lanes = sum(bool(group) for _scope, group in groups)
    if matched_lanes:
        notes.append(
            "operator retrieval used %d isolated scope lane(s); %d returned records"
            % (len(groups), matched_lanes)
        )
        notes.extend(lane_notes)
        return

    notes.append(
        "mapped operator scopes returned no records; unscoped text widening was suppressed"
    )


def _hardware_record(request: dict, payload: dict) -> tuple[str, dict]:
    rid = str(payload.get("id") or "hardware.%s.%s" %
              (request["kind"], query_wiki.fold(request["value"])))
    ptype = payload.get("type") or {
        "product": "spec-sheet", "instruction": "instruction", "feature": "arch-feature"
    }[request["kind"]]
    applies = {}
    for key in ("vendor", "arch", "product", "sm_arch", "mnemonic", "feature"):
        value = payload.get(key)
        if value is not None:
            applies[key] = value
    if request["kind"] == "product":
        applies.setdefault("product", request["value"].lower())
    body = {k: v for k, v in payload.items()
            if k not in {"id", "type", "vendor", "arch", "product", "sm_arch",
                         "mnemonic", "feature", "provenance", "evidence"}}
    # Field answers carry provenance as a factual source class, not an experience
    # confidence verdict. Preserve it inside the isolated hardware payload.
    if "provenance" in payload:
        body["provenance"] = payload["provenance"]
    return rid, {"source": "hardware_wiki", "type": ptype,
                 "applies_to": applies, "match": {"kind": "exact"}, "payload": body}


def query_hardware(store_root: Path, intent: dict, records: dict,
                   notes: list[str], max_records: int) -> None:
    for request in intent["hardware_requests"]:
        if max_records and len(records) >= max_records:
            notes.append("hardware results were not all served because the record cap was reached")
            return
        flags = ["--" + request["kind"], request["value"]]
        if request["field"] and request["kind"] == "product":
            flags += ["--field", request["field"]]
        if request["vs"] and request["kind"] == "product":
            flags += ["--vs", request["vs"]]
        code, payload, error = _run_json(
            [sys.executable, str(store_root / "tools" / "query_hardware.py")] + flags)
        if code != 0 and request["kind"] == "product" and request["field"]:
            # The bridge is store-blind and cannot validate a field vocabulary.
            # Fail safely to the exact product record instead of dropping the
            # hardware store from the answer.
            code, payload, fallback_error = _run_json([
                sys.executable,
                str(store_root / "tools" / "query_hardware.py"),
                "--product", request["value"],
            ])
            if code == 0 and isinstance(payload, dict):
                notes.append(
                    "hardware field %r was not addressable; served the complete %s product record"
                    % (request["field"], request["value"])
                )
                error = ""
            else:
                error = fallback_error or error
        if code == 4 and isinstance(payload, dict):
            notes.append("hardware %s %s is not recorded; use obtain_instead from the store"
                         % (request["kind"], request["value"]))
            continue
        if code != 0 or not isinstance(payload, dict):
            notes.append("hardware lookup failed for %s %s: %s" %
                         (request["kind"], request["value"],
                          (error.splitlines() or ["invalid output"])[0][:240]))
            continue
        rid, entry = _hardware_record(request, payload)
        records[rid] = entry


def _exclude_for_store(exclude: str | None, store: str) -> str | None:
    """Translate served namespaced ids back to the selected store's raw ids."""
    if not exclude:
        return None
    selected = []
    for raw in exclude.split(","):
        rid = raw.strip()
        if not rid:
            continue
        if "::" not in rid:
            if store == PUBLIC_STORE:
                selected.append(rid)
            continue
        namespace, inner = rid.split("::", 1)
        if namespace == store and inner:
            selected.append(inner)
    return ",".join(selected) or None


def merge_store_records(
    groups: list[tuple[str, dict[str, dict]]], max_records: int,
) -> tuple[dict[str, dict], int]:
    """Round-robin isolated stores under one global record cap.

    Public ids stay unchanged for compatibility. Internal ids are always
    namespaced, so equal raw ids from the two repositories cannot collide.
    """
    pending = [(store, iter(records.items())) for store, records in groups]
    merged: dict[str, dict] = {}
    exhausted: set[str] = set()
    while len(exhausted) < len(pending):
        progressed = False
        for store, iterator in pending:
            if store in exhausted:
                continue
            try:
                rid, raw_entry = next(iterator)
            except StopIteration:
                exhausted.add(store)
                continue
            progressed = True
            served_id = rid if store == PUBLIC_STORE else "%s::%s" % (store, rid)
            entry = dict(raw_entry)
            entry["store"] = store
            entry["wiki_id"] = "%s::%s" % (store, rid)
            merged[served_id] = entry
            if max_records and len(merged) >= max_records:
                remaining = sum(1 for _, records in groups for _ in records) - len(merged)
                return merged, max(0, remaining)
        if not progressed:
            break
    return merged, 0


def main(argv=None) -> int:
    query_id = "wiki-query-%s" % uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    bridge_ms = None
    retrieval_ms = None
    bridge_telemetry: dict[str, object] = {}
    bridge_attempts = 0
    bridge_protocol = "not_started"
    record_count = 0
    records_by_source: dict[str, int] = {}
    records_by_store: dict[str, int] = {
        PUBLIC_STORE: 0,
        INTERNAL_STORE: 0,
    }
    status = "error"
    selected_cli = os.environ.get(BRIDGE_CLI_ENV, agent_launch.DEFAULT_CLI)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("request", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--store-root", default=None)
    ap.add_argument("--agent-cli", choices=agent_launch.SUPPORTED,
                    default=selected_cli)
    ap.add_argument("--timeout", type=int, default=agent_launch.DEFAULT_TIMEOUT)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--max-bytes", type=int, default=None)
    ap.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-workspace", action="store_true")
    # Kept for command compatibility; full isolated payloads are always served.
    ap.add_argument("--brief", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    # The environment value is an operator policy, not merely a default. This
    # lets a campaign require one audited bridge runtime even if an episode
    # copies an old command line that names another CLI explicitly.
    locked_cli = os.environ.get(BRIDGE_CLI_ENV)
    if locked_cli:
        if locked_cli not in agent_launch.SUPPORTED:
            die("bad %s=%r (supported: %s)" %
                (BRIDGE_CLI_ENV, locked_cli, ", ".join(agent_launch.SUPPORTED)))
        if args.agent_cli != locked_cli:
            print("query_bridge_agent: overriding --agent-cli %s with policy %s" %
                  (args.agent_cli, locked_cli), file=sys.stderr)
        args.agent_cli = locked_cli

    request = read_request(args)
    store_roots = resolve_store_roots(args.store_root)
    store_root = store_roots[0][1]
    workspace = Path(tempfile.mkdtemp(prefix="query-bridge-"))
    try:
        # Security boundary: user text may reach only a verified no-tools JSON
        # protocol. Codex remains excluded because read-only mode still permits
        # shell execution and file reads; Qoder uses tools="" plus empty MCP.
        plain_json_bridge = True
        bridge_protocol = "plain_json_stdout_v3_no_tools_%s" % args.agent_cli
        skill_path = None
        prompt = bridge_prompt(
            request, store_root, workspace, skill_path, args.exclude,
            plain_json=plain_json_bridge,
        )
        if args.dry_run:
            print("query_bridge_agent/SKILL.md\nMUST NOT run query_wiki.py\n"
                  "legacy handoff: query_intent.json\n\n" + prompt)
            status = "dry_run"
            return 0
        bridge_started = time.perf_counter()
        intent = deterministic_bridge_intent(request)
        failures = []
        if intent is not None:
            bridge_protocol = "deterministic_standard_request_v1"
            bridge_telemetry["bridge_num_turns"] = 0
            print(
                "query_bridge_agent: deterministic standard request",
                file=sys.stderr,
            )
        else:
            print("query_bridge_agent: extracting intent", file=sys.stderr)
            for attempt in range(2):
                bridge_attempts += 1
                current_prompt = prompt
                if failures:
                    current_prompt += REPAIR_SUFFIX.format(error=failures[-1][:300])
                try:
                    out, err, code, timed_out = agent_launch.run_json(
                        args.agent_cli, current_prompt, workspace, args.timeout
                    )
                except agent_launch.LaunchError as exc:
                    # A missing/forbidden local executable cannot be repaired by
                    # prompting the same executable again.  Fail with the same
                    # stable bridge error channel instead of a Python traceback.
                    failures.append("launch failed: %s" % exc)
                    print(
                        "query_bridge_agent: bridge attempt %d/2 failed: %s"
                        % (attempt + 1, failures[-1][:300]),
                        file=sys.stderr,
                    )
                    break
                if code != 0 or timed_out:
                    tail = (err or out or "").strip()[-800:]
                    failures.append("call failed (exit=%s timed_out=%s)%s" % (
                        code, timed_out, ": " + tail if tail else ""))
                    if attempt == 0:
                        print(
                            "query_bridge_agent: bridge attempt 2/2 retrying after: %s"
                            % failures[-1][:300],
                            file=sys.stderr,
                        )
                    continue
                # Compatibility with older embedded launch adapters that write
                # the former handoff file. Verified bridge CLIs return stdout,
                # so this path is normally unreachable in current deployments.
                legacy_handoff = workspace / INTENT_FILE
                if not out.strip() and legacy_handoff.is_file():
                    out = json.dumps({"result": legacy_handoff.read_text(
                        encoding="utf-8", errors="replace")})
                try:
                    raw_envelope = json.loads(out)
                    if isinstance(raw_envelope, dict):
                        _merge_bridge_telemetry(
                            bridge_telemetry, _bridge_telemetry(raw_envelope))
                except json.JSONDecodeError:
                    pass
                try:
                    intent_doc, _ = parse_bridge_json_output(out)
                    strictly_validate_intent(intent_doc)
                    intent = validate_intent(intent_doc)
                    break
                except ValueError as exc:
                    failures.append(str(exc))
                    if attempt == 0:
                        print(
                            "query_bridge_agent: bridge attempt 2/2 retrying after: %s"
                            % failures[-1][:300],
                            file=sys.stderr,
                        )
        bridge_ms = round((time.perf_counter() - bridge_started) * 1000, 3)
        if intent is None:
            detail = failures[-1] if failures else "no-intent"
            if ("result JSON must be an object" in detail
                    or "result is not plain JSON" in detail):
                die("bad-intent query_bridge_agent failed after %d attempts: %s" %
                    (bridge_attempts, detail), 2)
            die("bad-intent no-intent: query_bridge_agent failed after %d attempts: %s" %
                (bridge_attempts, detail), 4)

        retrieval_started = time.perf_counter()
        notes: list[str] = []
        groups: list[tuple[str, dict[str, dict]]] = []
        for store, current_root in store_roots:
            if not _queryable_store(current_root):
                groups.append((store, {}))
                notes.append("[%s] store unavailable; module returned empty" % store)
                continue
            try:
                normalized, current_notes = normalize_intent(
                    intent, current_root, request_text=request)
                current_records: dict[str, dict] = {}
                query_hardware(
                    current_root, normalized, current_records, current_notes,
                    args.max_records,
                )
                query_kernel(
                    current_root, normalized, current_records, current_notes,
                    args.max_records, _exclude_for_store(args.exclude, store),
                    args.max_bytes,
                )
            except (OSError, ValueError, KeyError) as exc:
                groups.append((store, {}))
                notes.append(
                    "[%s] retrieval failed; module returned empty: %s"
                    % (store, str(exc).splitlines()[0][:160])
                )
                continue
            groups.append((store, current_records))
            notes.extend("[%s] %s" % (store, note) for note in current_notes)
        records, dropped = merge_store_records(groups, args.max_records)
        if dropped:
            notes.append(
                "records from isolated stores were truncated at the %d-record global cap; %d omitted"
                % (args.max_records, dropped)
            )
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
        record_count = len(records)
        for entry in records.values():
            source = str(entry.get("source") or "unknown")
            records_by_source[source] = records_by_source.get(source, 0) + 1
            store = str(entry.get("store") or "unknown")
            records_by_store[store] = records_by_store.get(store, 0) + 1
        print(json.dumps({
            "query_id": query_id,
            "records": records,
            "notes": notes,
        }, ensure_ascii=False))
        status = "ok"
        return 0
    finally:
        if args.keep_workspace:
            print("workspace kept at %s" % workspace, file=sys.stderr)
        else:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        finished_at = datetime.now(timezone.utc)
        _append_metric(os.environ.get(METRICS_LOG_ENV), {
            "query_id": query_id,
            "task_id": os.environ.get(TASK_ID_ENV),
            "pid": os.getpid(),
            "status": status,
            "agent_cli": getattr(args, "agent_cli", selected_cli),
            "bridge_protocol": bridge_protocol,
            "bridge_attempts": bridge_attempts,
            "bridge_retry_count": max(0, bridge_attempts - 1),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "bridge_latency_ms": bridge_ms,
            "retrieval_latency_ms": retrieval_ms,
            "record_count": record_count,
            "records_by_source": records_by_source,
            "records_by_store": records_by_store,
            **bridge_telemetry,
        })


if __name__ == "__main__":
    raise SystemExit(main())
