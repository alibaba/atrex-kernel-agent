#!/usr/bin/env python3
"""Derive the `opt-trace-1.0` validation profile from this repository's clean-1.3.

The records this skill produces are ordinary `clean-1.3` records: they enter the
shared store through `skills/wiki-gate`, so they must validate against
`schema/kernel/schema.json` exactly as it is. This file therefore does NOT invent
a dialect. It generates a **profile**: the same schema with this pipeline's own
rules turned into constraints, and every constraint NARROWS.

Narrowing-only is the invariant that makes the profile safe. A patch may forbid a
field, require an optional one, or shrink an enum; it may never add a property or
widen a type, because anything the profile accepts has to remain acceptable to
the store. `build()` asserts the two structural halves of that invariant it can
check mechanically (the `schema` const still says `clean-1.3`, and no patch adds
a property), and each patch states the corpus fact behind it.

Why generate instead of hand-writing a checklist: PATCHES is the complete,
auditable list of how this pipeline is stricter than the store, and re-running
picks up base changes. `--check` fails when the committed profile no longer
matches either the patch list or the base, which is what catches a schema change
upstream that this pipeline has not been reviewed against.

Usage:
  python3 make_schema.py                    # (re)generate the profile
  python3 make_schema.py --check            # exit 1 if it is stale
  python3 make_schema.py --show             # the patch list, with reasons
  python3 make_schema.py --validate REC...  # validate records against it
"""
import argparse
import hashlib
import json
import sys

import config as c

# Paths inside the base schema, spelled once.
RAW = ["properties", "evidence", "properties", "raw"]
SUMMARY = ["properties", "evidence", "properties", "summary"]
GAIN = ["$defs", "gain"]
ANTI_PAYLOAD = ["allOf", 1, "then", "properties", "payload"]

# The three record types a single-kernel trace can evidence. A technique-card is a
# cross-corpus aggregate and a doc has no measurement, so neither can be produced
# from one trace; `numerics-rule` / `dispatch-rule` need a rule that holds beyond
# the operator at hand.
TRACE_TYPES = ["strategy", "anti-strategy", "reference-kernel"]

PATCHES = [
    {"why": "A trace is one kernel. Records from it are operator-level, or at "
            "most cross-operator when the same lever is confirmed elsewhere; "
            "`generic` would promote one operator's measurement into the L2 tier "
            "and make it recallable for every architecture.",
     "path": ["properties", "level", "enum"], "op": "set",
     "value": ["operator", "cross-operator"]},

    {"why": "partition.py emits exactly these three shapes. Allowing the others "
            "here would let a distilling agent invent an aggregate "
            "(technique-card) whose numbers no single trace can support.",
     "path": ["properties", "type", "enum"], "op": "set",
     "value": TRACE_TYPES},

    {"why": "Provenance is the trace repository plus the commit, and both are "
            "mandatory: the verbatim and provenance gates re-resolve the record "
            "against that commit, and a record with neither cannot be audited "
            "again once the packets are gone.",
     "path": RAW, "op": "require",
     "value": ["source_repo", "version", "git_commit"]},

    {"why": "evidence.raw is closed to the fields a trace can actually supply. "
            "The base leaves it open, which is how it accumulated fields "
            "pointing at a markdown page and at an absolute file inside "
            "someone's checkout -- both dead references in a published store, "
            "and this repository has no markdown tree at all. Closing the "
            "object removes them without having to name them, and stops a "
            "distiller from inventing further raw keys that no consumer reads.",
     "path": RAW, "op": "restrict",
     "value": ["source_repo", "version", "git_commit", "effect", "file_paths",
               "hunk_index", "evidence_extra"]},

    {"why": "Every number here comes from the trace's own harness, and "
            "conclusions expire when the environment changes, so the agent must "
            "be told which harness and which GPU produced them. The base leaves "
            "measured_on optional because a documentation-derived record has no "
            "harness; a trace-derived one always does.",
     "path": SUMMARY, "op": "require",
     "value": ["confidence", "measured_on"]},

    {"why": "A trace always knows what kind of benefit it measured: performance "
            "for a milestone, none for a rejected attempt. Leaving `kind` unset "
            "makes an anti-strategy indistinguishable from a strategy whose "
            "number is missing.",
     "path": GAIN, "op": "require", "value": ["basis", "kind"]},

    {"why": "The store's established-fact gate rejects an anti-strategy without "
            "`established_fact`, but the base schema still accepts one, so an "
            "agent can produce a record that validates and is then refused at "
            "the gate. Requiring it here moves that failure to the earliest "
            "point: the distiller's own validate_store run.",
     "path": ANTI_PAYLOAD, "op": "require",
     "value": ["goal", "problem", "attempted", "verdict", "lesson",
               "established_fact"]},
]


def descend(obj, path):
    for key in path[:-1]:
        obj = obj[key]
    return obj, path[-1]


def node_at(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def apply_patch(schema, patch):
    op = patch["op"]
    if op == "set":
        parent, key = descend(schema, patch["path"])
        if isinstance(parent, dict) and key not in parent:
            raise KeyError("%r is not present in the base" % key)
        parent[key] = patch["value"]
    elif op == "require":
        # Turning an optional field into a mandatory one is the commonest kind of
        # narrowing here, and the base does not always carry a `required` list to
        # extend -- `evidence.raw` has none at all.
        node = node_at(schema, patch["path"])
        declared = set(node.get("properties") or {})
        missing = [name for name in patch["value"] if name not in declared]
        if missing:
            raise KeyError("%s declares no propert%s %s"
                           % (".".join(str(x) for x in patch["path"]),
                              "y" if len(missing) == 1 else "ies", missing))
        node["required"] = list(patch["value"])
    elif op == "restrict":
        # Deleting a property from an object whose additionalProperties is true
        # forbids nothing, so the allowlist and the close have to happen
        # together. This is the strongest narrowing available on an open object.
        node = node_at(schema, patch["path"])
        declared = node.get("properties") or {}
        missing = [name for name in patch["value"] if name not in declared]
        if missing:
            raise KeyError("%s declares no propert%s %s"
                           % (".".join(str(x) for x in patch["path"]),
                              "y" if len(missing) == 1 else "ies", missing))
        node["properties"] = {name: declared[name] for name in patch["value"]}
        node["additionalProperties"] = False
    else:
        raise ValueError("unknown op %r" % op)


def build():
    base_text = c.SCHEMA_PATH.read_text()
    base = json.loads(base_text)
    schema = json.loads(base_text)

    for i, patch in enumerate(PATCHES):
        try:
            apply_patch(schema, patch)
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit(
                "patch %d on %s no longer applies (%s); %s changed, so this "
                "patch has to be re-reviewed against it"
                % (i, ".".join(str(x) for x in patch["path"]), exc,
                   c.SCHEMA_PATH.name))

    # The narrowing invariant, in the two forms that can be checked here.
    const = schema["properties"]["schema"].get("const")
    if const != c.SCHEMA_NAME:
        raise SystemExit(
            "a patch changed properties.schema.const to %r; records must keep "
            "declaring %r or the store's schema gate will reject every one of "
            "them" % (const, c.SCHEMA_NAME))
    _assert_no_new_properties(base, schema)

    schema["$id"] = "%s.schema.json" % c.PROFILE_NAME
    schema["title"] = "%s -- %s narrowed for optimisation traces" % (
        c.PROFILE_NAME, c.SCHEMA_NAME)
    schema["description"] = (
        "Validation profile used by skills/opt-trace-mining. Records still "
        "declare schema=%s and still validate against %s: every patch in "
        "make_schema.py PATCHES only narrows, so passing this profile implies "
        "passing the store's schema. PATCHES is the complete list of the ways "
        "this pipeline is stricter, with the reason for each."
        % (c.SCHEMA_NAME, c.SCHEMA_PATH.name))
    schema["x-profile-of"] = {
        "base": c.SCHEMA_NAME,
        "base_file": str(c.SCHEMA_PATH.relative_to(c.GPU_WIKI)),
        "base_sha256": hashlib.sha256(base_text.encode()).hexdigest(),
        "n_patches": len(PATCHES),
        "generated_by": "skills/opt-trace-mining/scripts/make_schema.py",
    }
    return schema


def _assert_no_new_properties(base, derived, path=()):
    """No patch may introduce a property the base does not have.

    A new property would make the profile accept records the store rejects, which
    inverts the whole point of the profile.
    """
    if isinstance(base, dict) and isinstance(derived, dict):
        if path[-1:] == ("properties",):
            added = sorted(set(derived) - set(base))
            if added:
                raise SystemExit(
                    "a patch added propert%s %s at %s; the profile may only "
                    "narrow the base"
                    % ("y" if len(added) == 1 else "ies", added,
                       ".".join(path) or "<root>"))
        for key in base:
            if key in derived:
                _assert_no_new_properties(base[key], derived[key],
                                          path + (str(key),))


def text_of(schema):
    return json.dumps(schema, ensure_ascii=False, indent=1) + "\n"


def validate(paths):
    try:
        import jsonschema
    except ImportError:
        print("jsonschema is not installed; cannot validate "
              "(pip install jsonschema)", file=sys.stderr)
        return 2
    if not c.PROFILE_SCHEMA.is_file():
        raise SystemExit("no profile at %s; run make_schema.py first"
                         % c.PROFILE_SCHEMA)
    validator = jsonschema.Draft202012Validator(
        json.loads(c.PROFILE_SCHEMA.read_text()))
    bad = 0
    for path in paths:
        record = json.loads(open(path).read())
        errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
        if errors:
            bad += 1
            print("FAIL %s" % path)
            for err in errors[:8]:
                where = "/".join(str(p) for p in err.path) or "<root>"
                print("       %s: %s" % (where, err.message[:160]))
        else:
            print("ok   %s" % path)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed profile is stale")
    ap.add_argument("--show", action="store_true",
                    help="print the patch list with its reasons")
    ap.add_argument("--validate", nargs="+", metavar="RECORD",
                    help="validate records against the profile")
    args = ap.parse_args()

    if args.validate:
        return validate(args.validate)

    if args.show:
        print("%s: %s + %d narrowing patches"
              % (c.PROFILE_NAME, c.SCHEMA_NAME, len(PATCHES)))
        for i, patch in enumerate(PATCHES):
            print("\n%2d. %s  [%s]" % (i, ".".join(str(x) for x in patch["path"]),
                                       patch["op"]))
            for line in patch["why"].split(". "):
                print("      %s" % line.strip().rstrip(".") + ".")
        return 0

    if not c.SCHEMA_PATH.is_file():
        raise SystemExit("base schema not found: %s" % c.SCHEMA_PATH)
    text = text_of(build())

    if args.check:
        if not c.PROFILE_SCHEMA.is_file():
            print("MISSING: %s; run `python3 make_schema.py`"
                  % c.PROFILE_SCHEMA, file=sys.stderr)
            return 1
        if c.PROFILE_SCHEMA.read_text() != text:
            print("STALE: %s no longer matches make_schema.py applied to %s.\n"
                  "Either the patch list or the base schema changed; review the "
                  "difference, then re-run `python3 make_schema.py`."
                  % (c.PROFILE_SCHEMA, c.SCHEMA_PATH), file=sys.stderr)
            return 1
        print("%s is up to date (%s + %d patches, base sha256 %s)"
              % (c.PROFILE_SCHEMA.name, c.SCHEMA_NAME, len(PATCHES),
                 hashlib.sha256(c.SCHEMA_PATH.read_bytes()).hexdigest()[:12]))
        return 0

    c.PROFILE_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    c.PROFILE_SCHEMA.write_text(text)
    print("wrote %s from %d patches" % (c.PROFILE_SCHEMA, len(PATCHES)))
    for patch in PATCHES:
        print("  %-28s %s" % (".".join(str(x) for x in patch["path"][-2:]),
                              patch["why"].split(".")[0][:66]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
