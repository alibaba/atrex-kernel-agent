#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Fast natural-language front door to the GPU knowledge stores.

``query_bridge_agent`` performs one small task: turn prose into a typed intent.
The Claude path is a tool-free plain-JSON response with strict local validation;
other CLIs retain the legacy file handoff. The bridge is forbidden to inspect or
query either store. This script owns every
deterministic operation after that point: vocabulary normalization, staged
widening, execution, deduplication, projection and context limits.

The output intentionally has only two top-level fields: records and notes.
Records from kernel_wiki and hardware_wiki share one id-keyed mapping, while
each payload remains an independent JSON value under its own stable id. The
public store is always queried; an installed sibling ``internal_gpu_wiki`` is
queried as a second isolated store. Internal ids are namespaced so a private
record can never overwrite a public record with the same stable id.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import agent_launch
import hardware_identity
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
SKILL_NAME = "query_bridge_agent"
DEFAULT_MAX_RECORDS = 20
ENOUGH_RECORDS = 8
MAX_TERMS = 6

INTENTS = {"technique", "pitfall", "documentation", "diagnosis", "correctness"}
HARDWARE_KINDS = {"product", "instruction", "feature"}

INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "architecture", "vendor", "dsl", "operator_terms",
        "measured_symptoms", "free_text_terms", "intents", "hardware_requests",
    ],
    "properties": {
        "architecture": {"type": ["string", "null"]},
        "vendor": {"type": ["string", "null"]},
        "dsl": {"type": ["string", "null"]},
        "operator_terms": {
            "type": "array", "maxItems": MAX_TERMS, "items": {"type": "string"},
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
"operator_terms":[string],"measured_symptoms":[string],
"free_text_terms":[string],
"intents":["technique"|"pitfall"|"documentation"|"diagnosis"|"correctness"],
"hardware_requests":[{{"kind":"product"|"instruction"|"feature",
"value":string,"field":string|null,"vs":string|null}}]}}

Include every key exactly once. Do not research, invoke tools,
inspect a store, invent missing scope values, explain the result, or generate
query flags. Copy architecture/vendor/DSL/product spellings from the request.
Use caller-language operator/API/mechanism phrases. A measured symptom requires
an explicit profile counter or number; keep hypotheses in free_text_terms. Emit
at most 6 operator_terms, exactly 0-2 measured_symptoms, at most 6
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

REPAIR_SUFFIX = """

Your prior response failed strict local validation: {error}
Return a corrected JSON object now. Output only JSON; include every required key
and no additional keys.
"""

FILE_BRIDGE_PROMPT = """You are query_bridge_agent, a fast intent extractor.

Read {skill_path}. You MUST NOT run query_wiki.py, query_hardware.py, grep the
stores, enumerate vocabularies, inspect records, or perform research. The caller
script does all retrieval deterministically after you finish.

Write exactly one file, {intent_path}, containing one JSON object matching the
skill. Write no other file. Keep terms in the caller's own words; do not guess a
store token. Distinguish measured symptoms from hypotheses.

<<<REQUEST
{request}
REQUEST>>>
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
        "measured_symptoms": _strings(doc.get("measured_symptoms"), 2),
        "free_text_terms": _strings(doc.get("free_text_terms")),
        "intents": intents,
        "hardware_requests": deduped_hw[:4],
    }


def strictly_validate_intent(doc: object) -> dict:
    """Validate the plain-JSON bridge contract without coercion or data loss."""
    if not isinstance(doc, dict):
        raise ValueError("top level must be an object")
    expected = set(INTENT_SCHEMA["required"])
    actual = set(doc)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError("missing keys: %s" % ", ".join(missing))
    if extra:
        raise ValueError("unexpected keys: %s" % ", ".join(extra))

    for key in ("architecture", "vendor", "dsl"):
        if doc[key] is not None and not isinstance(doc[key], str):
            raise ValueError("%s must be a string or null" % key)
        if isinstance(doc[key], str) and not doc[key].strip():
            raise ValueError("%s must be non-empty or null" % key)

    list_limits = {
        "operator_terms": MAX_TERMS,
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


def skill_source(store_root: Path) -> Path | None:
    for root in (OWN_STORE_ROOT, store_root):
        candidate = root / "skills" / SKILL_NAME / "SKILL.md"
        if candidate.is_file():
            return candidate
    return None


def link_skill(workspace: Path, store_root: Path) -> Path | None:
    source = skill_source(store_root)
    if source is None:
        return None
    for host in (".claude", ".qoder", ".agents"):
        skills = workspace / host / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        link = skills / SKILL_NAME
        if not link.exists():
            os.symlink(source.parent, link, target_is_directory=True)
    return source


def bridge_prompt(request: str, store_root: Path, workspace: Path,
                  skill_path: Path | None, exclude: str | None = None,
                  plain_json: bool = False) -> str:
    del store_root, exclude  # the extractor is intentionally store-blind
    if plain_json:
        return PLAIN_JSON_BRIDGE_PROMPT.format(request=request)
    return FILE_BRIDGE_PROMPT.format(
        skill_path=skill_path or "(skill missing; follow this prompt)",
        intent_path=workspace / INTENT_FILE,
        request=request,
    )


def parse_claude_json_output(stdout: str) -> tuple[dict, dict]:
    """Extract plain JSON plus timing/token telemetry from Claude's JSON envelope."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Claude stdout is not a JSON result envelope: %s" % exc) from exc
    if not isinstance(envelope, dict):
        raise ValueError("Claude stdout result envelope must be an object")
    payload = envelope.get("result")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Claude result is not plain JSON: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise ValueError("Claude result JSON must be an object")
    return payload, _claude_telemetry(envelope)


def _claude_telemetry(envelope: dict) -> dict:
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


def _resolve_vocab(value: str | None, allowed: set[str]) -> str | None:
    if not value:
        return None
    folded = {query_wiki.fold(x): x for x in allowed}
    return folded.get(query_wiki.fold(value))


def _resolve_arch(value: str | None, allowed: set[str]) -> str | None:
    """Resolve runtime spelling without printing query_wiki's CLI error channel."""
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


def normalize_intent(intent: dict, store_root: Path) -> tuple[dict, list[str]]:
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
    dsl = _resolve_vocab(intent["dsl"], vv["dsl"])
    if intent["dsl"] and not dsl:
        notes.append("DSL %r is not a store scope token and was kept as free text"
                     % intent["dsl"])

    operator = family = None
    spill = []
    for term in intent["operator_terms"]:
        operator = operator or _resolve_vocab(term, vv["operator"])
        family = family or _resolve_vocab(term, vv["family"])
        if term not in (operator, family):
            spill.append(term)
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
            product = hardware_identity.normalize_product_name(request["value"])
            if product in hardware_identity.HARDWARE_IDENTITIES:
                normalized_request.update(
                    kind="product", value=product, field=None,
                )
                notes.append(
                    "hardware identity %r was classified as a feature; treated as product %s"
                    % (request["value"], product)
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
    return dict(intent, arch=arch, vendor=vendor, dsl=dsl, operator=operator,
                family=family, symptom=symptom, terms=terms[:MAX_TERMS],
                hardware_requests=hardware_requests), notes


def _kernel_flags(intent: dict, *, drop_operator=False, drop_symptom=False,
                  any_terms=False, cross_arch=False, limit=8,
                  exclude: str | None = None, max_bytes: int | None = None) -> list[str]:
    flags: list[str] = []
    for flag, value in (("--arch", intent["arch"]), ("--vendor", intent["vendor"]),
                        ("--dsl", intent["dsl"])):
        if value:
            flags += [flag, value]
    if not drop_operator:
        if intent["operator"]:
            flags += ["--operator", intent["operator"]]
        elif intent["family"]:
            flags += ["--family", intent["family"]]
    if intent["symptom"] and not drop_symptom:
        flags += ["--symptom", intent["symptom"]]
    flags += intent["terms"]
    if any_terms and intent["terms"]:
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


def query_kernel(store_root: Path, intent: dict, records: dict, notes: list[str],
                 max_records: int, exclude: str | None,
                 max_bytes: int | None) -> None:
    stages = [
        ("exact scope", {}),
        ("operator scope removed", {"drop_operator": True}),
        ("symptom scope removed", {"drop_operator": True, "drop_symptom": True}),
        ("text terms widened to OR", {"drop_operator": True, "drop_symptom": True,
                                      "any_terms": True}),
    ]
    if intent["arch"]:
        stages.append(("same-vendor sibling architectures included",
                       {"drop_operator": True, "drop_symptom": True,
                        "any_terms": True, "cross_arch": True}))
    seen_flags = set()
    for label, options in stages:
        remaining = max_records - len(records) if max_records else DEFAULT_MAX_RECORDS
        if max_records and remaining <= 0:
            notes.append("kernel results were truncated at the %d-record cap" % max_records)
            break
        flags = _kernel_flags(intent, limit=min(8, max(1, remaining)), exclude=exclude,
                              max_bytes=max_bytes, **options)
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
                records[rid] = entry
                if max_records and len(records) >= max_records:
                    break
        if found and label != "exact scope":
            notes.append("kernel lookup widened: %s" % label)
        kernel_count = sum(r.get("source") == "kernel_wiki" for r in records.values())
        target = min(ENOUGH_RECORDS, max_records or ENOUGH_RECORDS)
        if kernel_count >= target:
            break
    if not any(r.get("source") == "kernel_wiki" for r in records.values()):
        notes.append("kernel_wiki returned no matching records")


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
            merged[served_id] = entry
            if max_records and len(merged) >= max_records:
                remaining = sum(1 for _, records in groups for _ in records) - len(merged)
                return merged, max(0, remaining)
        if not progressed:
            break
    return merged, 0


def main(argv=None) -> int:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    bridge_ms = None
    retrieval_ms = None
    bridge_telemetry: dict[str, object] = {}
    bridge_attempts = 0
    bridge_protocol = "file_handoff"
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
        plain_json_bridge = args.agent_cli == "claude"
        bridge_protocol = (
            "plain_json_stdout_v2" if plain_json_bridge else "file_handoff"
        )
        skill_path = None if plain_json_bridge else link_skill(workspace, store_root)
        prompt = bridge_prompt(
            request, store_root, workspace, skill_path, args.exclude,
            plain_json=plain_json_bridge,
        )
        if args.dry_run:
            print(prompt)
            status = "dry_run"
            return 0
        print("query_bridge_agent: extracting intent", file=sys.stderr)
        bridge_started = time.perf_counter()
        if plain_json_bridge:
            intent = None
            failures = []
            for attempt in range(2):
                bridge_attempts += 1
                current_prompt = prompt
                if failures:
                    current_prompt += REPAIR_SUFFIX.format(error=failures[-1][:300])
                out, err, code, timed_out = agent_launch.run_claude_json(
                    current_prompt, workspace, args.timeout
                )
                if code != 0 or timed_out:
                    tail = (err or out or "").strip()[-800:]
                    failures.append("call failed (exit=%s timed_out=%s)%s" % (
                        code, timed_out, ": " + tail if tail else ""))
                    continue
                try:
                    raw_envelope = json.loads(out)
                    if isinstance(raw_envelope, dict):
                        _merge_bridge_telemetry(
                            bridge_telemetry, _claude_telemetry(raw_envelope))
                except json.JSONDecodeError:
                    pass
                try:
                    intent_doc, _ = parse_claude_json_output(out)
                    strictly_validate_intent(intent_doc)
                    intent = validate_intent(intent_doc)
                    break
                except ValueError as exc:
                    failures.append(str(exc))
            bridge_ms = round((time.perf_counter() - bridge_started) * 1000, 3)
            if intent is None:
                die("bad-intent query_bridge_agent failed after %d attempts: %s" %
                    (bridge_attempts, failures[-1]), 4)
        else:
            out, err, code, timed_out = agent_launch.run(
                args.agent_cli, prompt, workspace, args.timeout
            )
            bridge_attempts = 1
            bridge_ms = round((time.perf_counter() - bridge_started) * 1000, 3)
            intent_path = workspace / INTENT_FILE
            if not intent_path.is_file():
                tail = (err or out or "").strip()[-800:]
                die("no-intent query_bridge_agent wrote no %s (exit=%s timed_out=%s)%s"
                    % (INTENT_FILE, code, timed_out,
                       "\n--- agent output tail ---\n" + tail if tail else ""), 4)
            try:
                intent = validate_intent(json.loads(intent_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                die("bad-intent %s is not valid json: %s" % (INTENT_FILE, exc))

        retrieval_started = time.perf_counter()
        notes: list[str] = []
        groups: list[tuple[str, dict[str, dict]]] = []
        for store, current_root in store_roots:
            if not _queryable_store(current_root):
                groups.append((store, {}))
                notes.append("[%s] store unavailable; module returned empty" % store)
                continue
            try:
                normalized, current_notes = normalize_intent(intent, current_root)
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
        print(json.dumps({"records": records, "notes": notes}, ensure_ascii=False))
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
