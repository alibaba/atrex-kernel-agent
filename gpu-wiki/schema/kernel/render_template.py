#!/usr/bin/env python3
"""Render schema/TEMPLATE.md from the JSON Schema.

The schema is the single source of truth; a hand-written field reference would
drift from it silently the first time someone adds a field. So this reads
schema.json and emits the human-readable template, and --check fails
when the committed document no longer matches, which is what keeps the two
honest.

    python3 render_template.py            # regenerate schema/TEMPLATE.md
    python3 render_template.py --check    # exit 1 if it is stale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(os.path.dirname(HERE))
OUT_ROOT = GPU_WIKI
SCHEMA_PATH = os.path.join(HERE, "schema.json")
DOC_PATH = os.path.join(HERE, "TEMPLATE.md")
RECORDS_ROOT = os.path.join(GPU_WIKI, "kernel_wiki", "records")

TYPE_ORDER = ["strategy", "anti-strategy", "technique-card", "symptom-card",
              "reference-kernel", "doc", "numerics-rule", "dispatch-rule"]

SHARED_ORDER = ["goal", "problem", "trace", "implementation", "cost",
                "metric_delta"]
# The gain defs document worth.gain, not a payload field, so they are rendered
# with the layer they belong to.
WORTH_ORDER = ["gain", "gain_metric", "metric_name"]
SHARED_TITLES = {
    "gain_metric": "Elements of worth.gain.metrics[] / regressions[]",
    "metric_name": "metric vocabulary (closed)",
    "metric_delta": "metric_delta — elements of evidence.summary.mechanism_metrics",
    "goal": "goal — one sentence: why the agent is reading this record",
    "links": "links — engine-side id graph (stripped when served)",
    "problem": "problem — what problem it solves (self-contained, no retrieval needed)",
    "trace": "trace — which approach it improves on",
    "implementation": "implementation — code (never a repository path)",
    "gain": "worth.gain — expected gain (percentages only)",
    "cost": "cost — cost of adopting it",
    "relations": "relations — internal ids, a convenience",
}
# payload field name -> $defs name, for the per-type tables
DEF_OF_FIELD = {"goal": "goal", "problem": "problem", "trace": "trace",
                "implementation": "implementation", "cost": "cost"}


def typename(node: dict, defs: dict) -> str:
    if "$ref" in node:
        return "→ " + node["$ref"].split("/")[-1]
    if "const" in node:
        return "const %r" % node["const"]
    if "enum" in node:
        values = [v for v in node["enum"] if v is not None]
        joined = " \\| ".join(str(v) for v in values)
        return joined if len(joined) <= 90 else joined[:88] + "…"
    kind = node.get("type")
    if isinstance(kind, list):
        base = "/".join(k for k in kind if k != "null")
        return base + ("?" if "null" in kind else "")
    if kind == "array":
        item = node.get("items") or {}
        if "$ref" in item:
            return "[→ %s]" % item["$ref"].split("/")[-1]
        if item.get("type") == "object":
            return "object[]"
        return "%s[]" % (item.get("type") or "any")
    return kind or "any"


def plural(count: int, noun: str) -> str:
    """`3 records` / `1 record`, for the per-type section headings."""
    return "%d %s%s" % (count, noun, "" if count == 1 else "s")


def one_line(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def field_rows(props: dict, required: set, defs: dict) -> list[str]:
    rows = ["| Field | Required | Type | Description |", "|---|:--:|---|---|"]
    for name, spec in props.items():
        rows.append("| `%s` | %s | %s | %s |" % (
            name, "●" if name in required else "",
            typename(spec, defs), one_line(spec.get("description", ""))))
    return rows


def nested_object_rows(name: str, spec: dict, defs: dict) -> list[str]:
    """One extra level, for the objects that carry real structure.

    A $ref is resolved first, otherwise a referenced object such as
    retrieval.links would show as a bare arrow with its fields never listed.
    """
    if "$ref" in spec:
        spec = defs[spec["$ref"].split("/")[-1]]
    if spec.get("type") != "object" or not spec.get("properties"):
        return []
    out = ["", "Inner structure of `%s`:" % name, ""]
    out += field_rows(spec["properties"], set(spec.get("required") or []), defs)
    return out


def record_counts() -> Counter:
    counts = Counter()
    root = RECORDS_ROOT
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name == "index.json":
                continue
            if name.endswith(".json"):
                with open(os.path.join(dirpath, name)) as handle:
                    counts[json.load(handle)["type"]] += 1
    return counts


def example_for(rtype: str) -> dict | None:
    root = RECORDS_ROOT
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name == "index.json" or not name.endswith(".json"):
                continue
            with open(os.path.join(dirpath, name)) as handle:
                record = json.load(handle)
            if record["type"] == rtype:
                return record
    return None


def render() -> str:
    schema = json.load(open(SCHEMA_PATH))
    defs = schema["$defs"]
    counts = record_counts()

    branches: dict[str, dict] = {}
    for branch in schema["allOf"]:
        condition = branch["if"]["properties"]["type"]
        for rtype in ([condition["const"]] if "const" in condition
                      else condition["enum"]):
            branches[rtype] = branch["then"]["properties"]["payload"]

    L = ["# Record template (`clean-1.3`)", "",
         "> **This file is generated from [`schema.json`](schema.json) by "
         "`schema/render_template.py`. Do not hand-edit it.**",
         "> The schema is the single source of truth and the schema gate in "
         "`tools/check_kernel_wiki.py` enforces it; this file is only its "
         "human-readable projection. To change a field, change the schema and "
         "re-run the renderer.", ""]

    # ------------------------------------------------------------- top level
    L += ["## Record top level (identical across all 8 types)", "",
          "```jsonc", "{"]
    for name, spec in schema["properties"].items():
        if name in ("retrieval", "payload", "evidence", "worth"):
            L.append('  "%s": { ... },' % name)
        else:
            L.append('  "%s": %s,' % (name, typename(spec, defs)))
    L += ["}", "```", ""]
    L += field_rows(schema["properties"], set(schema["required"]), defs) + [""]

    # ----------------------------------------------------------- four layers
    L += ["## What the four layers are for", "",
          "| Layer | For whom | Description |", "|---|---|---|",
          "| `retrieval` | the retrieval engine | One shape across all 8 types, hard-filterable. "
          "Includes `locator` (the engine-side locator, stripped when served, "
          "never visible to the agent) |",
          "| `payload` | agent | **Self-contained**: this layer alone is enough to act on. "
          "Polymorphic by type, detailed below |",
          "| `evidence.summary` | validation/maintenance only | Bottleneck evidence, mechanism metrics, "
          "measurement environment; never returned to a consuming agent |",
          "| `evidence.raw` | **humans only** | De-anonymized provenance and the absolute geomean, "
          "never part of the served projection |",
          "| `worth` | agent + ranking | The expected gain and the ranking derived from it, combined "
          "into one field. `rank.score` / `rank.tier` and `gain` are served; `track` "
          "(counters + corpus prior) stays engine-side |", ""]

    # ------------------------------------------------- retrieval / evidence
    for layer, title, blurb in (
        ("retrieval", "retrieval — for the retrieval engine (one shape across all 8 types)",
         "Hard-filter by scope first, then rank by text. `locator` is the engine-side "
         "locator and is stripped when served."),
        ("evidence", "evidence — summary for the agent, raw for humans only", ""),
        ("worth", "worth — gain + ranking (combined into one field)",
         "The agent gets only `rank.score`, `rank.tier` and `gain`; `track` and the score "
         "breakdown are engine-side and stripped when served — handing the agent the raw "
         "counters amounts to asking it to recompute a ranking that is already computed."),
    ):
        spec = schema["properties"][layer]
        L += ["## " + title, ""]
        if blurb:
            L += [blurb, ""]
        if spec.get("description"):
            L += ["> " + one_line(spec["description"], 400), ""]
        props = spec.get("properties") or {}
        L += field_rows(props, set(spec.get("required") or []), defs)
        for name, sub in props.items():
            if layer == "worth" and name in WORTH_ORDER:
                continue          # rendered as its own section just below
            L += nested_object_rows(name, sub, defs)
        L.append("")
        if layer == "worth":
            for key in WORTH_ORDER:
                sub = defs[key]
                L += ["### " + SHARED_TITLES[key], ""]
                if sub.get("description"):
                    L += ["> " + one_line(sub["description"], 400), ""]
                sub_props = sub.get("properties") or {}
                if sub_props:
                    L += field_rows(sub_props, set(sub.get("required") or []), defs)
                    for name, inner in sub_props.items():
                        L += nested_object_rows(name, inner, defs)
                else:
                    L.append("Type: `%s`" % typename(sub, defs))
                L.append("")

    # -------------------------------------------------------- shared blocks
    L += ["## payload shared blocks", "",
          "Every type requires `goal` / `problem` (the gain does not live here, see "
          "`worth.gain` above). The types that describe a concrete change "
          "(`strategy`, `reference-kernel`) additionally require `trace` and "
          "`implementation`.", ""]
    for key in SHARED_ORDER:
        spec = defs[key]
        L += ["### " + SHARED_TITLES[key], ""]
        if spec.get("description"):
            L += ["> " + one_line(spec["description"], 400), ""]
        props = spec.get("properties") or {}
        if props:
            L += field_rows(props, set(spec.get("required") or []), defs)
            for name, sub in props.items():
                L += nested_object_rows(name, sub, defs)
        else:
            # A scalar def has no fields to tabulate; state its type instead.
            L.append("Type: `%s`" % typename(spec, defs))
        L.append("")

    # ------------------------------------------------------------- per type
    L += ["## The payload of each type", ""]
    L += ["| type | Records | Required shared blocks | Own fields |",
          "|---|---:|---|---|"]
    for rtype in TYPE_ORDER:
        payload = branches[rtype]
        props = payload.get("properties") or {}
        required = set(payload.get("required") or [])
        shared = [n for n in props if n in DEF_OF_FIELD]  # noqa: F841
        own = [n for n in props if n not in DEF_OF_FIELD]
        count = counts.get(rtype, 0)
        L.append("| **%s** | %s | %s | %s |" % (
            rtype, count if count else "0 (reserved)",
            ", ".join("`%s`%s" % (n, "●" if n in required else "") for n in shared),
            ", ".join("`%s`%s" % (n, "●" if n in required else "") for n in own) or "—"))
    L += ["", "`●` = required.", ""]

    for rtype in TYPE_ORDER:
        payload = branches[rtype]
        props = payload.get("properties") or {}
        required = set(payload.get("required") or [])
        own = {n: s for n, s in props.items() if n not in DEF_OF_FIELD}
        count = counts.get(rtype, 0)
        L += ["### %s (%s)" % (rtype, plural(count, "record") if count
                                else "0 records, reserved"), ""]
        if own:
            L += field_rows(own, required, defs)
            for name, sub in own.items():
                if sub.get("type") == "array" and (sub.get("items") or {}).get("properties"):
                    L += ["", "Element structure of `%s[]`:" % name, ""]
                    L += field_rows(sub["items"]["properties"],
                                    set(sub["items"].get("required") or []), defs)
        else:
            L.append("No own fields: the shared blocks cover everything it "
                     "needs to express.")
        example = example_for(rtype)
        if example:
            L += ["", "Taken from `%s`:" % example["id"], "", "```jsonc",
                  json.dumps({k: v for k, v in example["payload"].items()
                              if k in own} or example["payload"],
                             ensure_ascii=False, indent=1)[:1400], "```"]
        L.append("")

    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Fail if the committed document is stale.")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = open(DOC_PATH).read() if os.path.exists(DOC_PATH) else ""
        if current != rendered:
            print("STALE %s -- rerun: python3 render_template.py" % DOC_PATH,
                  file=sys.stderr)
            return 1
        print("OK %s is current" % os.path.relpath(DOC_PATH, OUT_ROOT))
        return 0

    with open(DOC_PATH, "w") as handle:
        handle.write(rendered)
    print("wrote %s (%d lines)" % (os.path.relpath(DOC_PATH, OUT_ROOT),
                                   rendered.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
