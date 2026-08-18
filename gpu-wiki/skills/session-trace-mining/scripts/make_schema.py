#!/usr/bin/env python3
"""Derive session-trace-1.0 from clean-1.3 by applying a declared patch list.

Generated rather than hand-copied: PATCHES is the complete, auditable record of
how this schema diverges from the base, and re-running picks up base changes.
Every patch carries the corpus fact that forced it.

The base is this repository's kernel schema, and it already fits kernel-optimization
sessions well: the scope vocabularies cover the hardware and DSLs these runs target,
`workload_family` already has the families they touch, and `evidence.raw` is already
`additionalProperties: true`. So this patch list is short on purpose. What the base
does not cover is that the corpus is a *transcript*: there is no repository to
resolve a citation against, and a large share of the numbers are the agent reading
back its own notes.

Usage:
  python3 make_schema.py                # derive from the pinned base
  python3 make_schema.py --check        # exit 1 if the output is stale
  python3 make_schema.py --sync-base    # re-pin the base, reporting what moved
  python3 make_schema.py --check-base   # report whether the base drifted
"""
import argparse
import hashlib
import json
import sys

from config import (BASE_NAME, BASE_SCHEMA as BASE, DERIVED_NAME,
                    DERIVED_SCHEMA as OUT, UPSTREAM_SCHEMA as UPSTREAM)

# SCOPE and GENERALITY carry no patch today: the base DSL vocabulary already
# covers these corpora. They are kept named because they are the two paths a new
# corpus has to extend *together* -- see the note at the head of PATCHES.
SCOPE = ["properties", "retrieval", "properties", "scope", "properties"]
GENERALITY = ["properties", "retrieval", "properties", "generality", "properties"]
GAIN = ["$defs", "gain", "properties"]
SUMMARY = ["properties", "evidence", "properties", "summary", "properties"]
RAW = ["properties", "evidence", "properties", "raw", "properties"]

# The provenance block. Every field exists to make one specific hand-edit
# detectable, which is why it is declared here instead of relying on
# evidence.raw's additionalProperties.
SESSION_PROVENANCE = {
    "type": "object",
    "additionalProperties": False,
    "description": "HUMAN ONLY. Where in the transcript archive this record came "
                   "from. There is no repository to resolve a citation against, "
                   "so the transcript line itself is the provenance and is "
                   "pinned by digest.",
    "required": ["set", "format", "rel_path", "session_id", "line_nos",
                 "line_digests", "candidate_id", "unit"],
    "properties": {
        "set": {"type": "string",
                "description": "Key in config.SETS. A name, never a path, so the "
                               "archive can be relocated without invalidating "
                               "records."},
        "format": {"enum": ["codex", "claude-code"]},
        "unit": {"enum": ["version-ladder", "ab-comparison"],
                 "description": "How the candidate was cut. Sets with a numbered "
                                "ladder get one candidate per version; sets "
                                "without one get one per measured A/B."},
        "rel_path": {"type": "string",
                     "description": "Transcript path relative to the set root."},
        "session_id": {"type": "string"},
        "line_nos": {"type": "array", "items": {"type": "integer"},
                     "description": "1-based, so `sed -n '<n>p'` reproduces the "
                                    "citation."},
        "line_digests": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "line_no -> sha256(raw line bytes)[:12]. Retargeting a "
                           "citation to another line fails the provenance gate."},
        "sibling_paths": {"type": "array", "items": {"type": "string"},
                          "description": "Other transcripts in the same set that "
                                         "contributed evidence, set-relative."},
        "anchor_uuid": {"type": ["string", "null"]},
        "anchor_turn_id": {"type": ["string", "null"]},
        "timestamp": {"type": ["string", "null"]},
        "candidate_id": {"type": "string"},
        "evidence_sha256": {"type": ["string", "null"],
                            "description": "sha256 of the packet evidence bundle "
                                           "this record was distilled from."},
        "dedup_key": {"type": ["string", "null"],
                      "description": "<repo basename>|<version>. Used to skip "
                                     "versions the main store already covers."},
        "diff_coverage": {"enum": ["full", "partial", "blind"],
                          "description": "blind means the edit was made by a shell "
                                         "write and left no diff in the "
                                         "transcript; such a candidate may not "
                                         "become a strategy record."},
        "number_tiers": {
            "type": "object",
            "additionalProperties": {"enum": ["T1", "T2", "T3"]},
            "description": "Per-metric trust tier: T1 tool output, T2 the agent "
                           "reading back its own notes, T3 an agent-authored "
                           "structured field."},
        "verdict": {"type": ["string", "null"],
                    "description": "committed | reverted | failed | unknown, as "
                                   "the run itself recorded it."},
    },
}

PATCHES = [
    # ---- identity enums ----
    # 基线的 scope.dsl / generality.language 词表已经覆盖了这些语料里出现的 DSL
    # （triton / cuda / cutedsl / gluon / …），所以这里没有补丁。新语料要加一个
    # DSL 时，**两处必须一起加**：只改一边会让同一条记录的两层自相矛盾，算子级
    # 经验会被错误提升到 generic 层而跨架构误召回。

    # ---- 收益的来源类别 ----
    {"why": "这批知识来自 agent 会话，不是我们自己跑的 harness，也不是公开仓库。"
            "服务层要能硬过滤掉「只有 agent 自己说」的证据，所以 source_kind 必填。",
     "path": GAIN + ["source_kind"], "op": "set",
     "value": {"enum": ["agent-session"],
               "description": "Provenance class of the number. Fixed for this "
                              "store: every record here comes from a coding "
                              "session transcript."}},
    {"why": "把 source_kind 加进 required，否则它会像 payload.config 一样长期为空。",
     "path": ["$defs", "gain", "required"], "op": "set",
     "value": ["basis", "source_kind"]},

    # ---- 置信度：会话里大量数字是 agent 回读自己的笔记 ----
    {"why": "基线的 confidence 枚举是给文档派生记录用的（measured/inferred/"
            "documented）。会话语料需要 reported（agent 回读自己的笔记）与 "
            "qualitative：存在只有机制描述、没有任何数字的候选（reverted 版本的"
            "根因分析），它们既不是 measured 也不是 documented，硬塞任一档都是谎报。",
     "path": SUMMARY + ["confidence", "enum"], "op": "set",
     "value": ["measured", "reported", "qualitative"]},

    # ---- 溯源：没有仓库可查，转录本身就是语料 ----
    {"why": "十六道门里的 provenance 门要靠它重新解析回转录：set 名 + 相对路径 + "
            "行号 + 行摘要。用绝对路径记溯源（基线 evidence.raw.detail_file 那种）"
            "在语料搬家或归档换机器之后就是死路径，这里不重犯：路径一律相对 set 根，"
            "set 只用名字引用。",
     "path": RAW + ["session"], "op": "set", "value": SESSION_PROVENANCE},
]


def descend(obj, path):
    for k in path[:-1]:
        obj = obj[k]
    return obj, path[-1]


def apply_patch(schema, p):
    parent, key = descend(schema, p["path"])
    op = p["op"]
    if op == "set":
        parent[key] = p["value"]
    elif op == "append":
        parent[key].append(p["value"])
    elif op == "remove":
        parent[key] = [x for x in parent[key] if x != p["value"]]
    else:
        raise ValueError("unknown op %r" % op)


def build():
    base_text = BASE.read_text()
    schema = json.loads(base_text)
    for i, p in enumerate(PATCHES):
        try:
            apply_patch(schema, p)
        except (KeyError, IndexError) as e:
            raise SystemExit(
                "patch %d on %s no longer applies (%s); the base changed, "
                "re-check this patch"
                % (i, ".".join(str(x) for x in p["path"]), e))
    schema["$id"] = "%s.schema.json" % DERIVED_NAME
    schema["title"] = "%s (derived from %s)" % (DERIVED_NAME, BASE_NAME)
    schema["description"] = (
        "Records distilled from AI coding-agent session transcripts. Derived "
        "from " + BASE_NAME + " by scripts/make_schema.py; PATCHES there is the "
        "full list of deviations with the corpus fact behind each.")
    schema["properties"]["schema"] = {"const": DERIVED_NAME}
    schema["x-derived-from"] = {
        "base": BASE_NAME,
        "base_sha256": hashlib.sha256(base_text.encode()).hexdigest(),
        "n_patches": len(PATCHES),
    }
    return schema


def sync_base():
    """Pull upstream into the pinned copy, reporting what moved so the patches
    can be re-reviewed against the change."""
    if not UPSTREAM.exists():
        raise SystemExit("upstream schema not found: %s" % UPSTREAM)
    new_text = UPSTREAM.read_text()
    new = json.loads(new_text)
    if BASE.exists():
        old_text = BASE.read_text()
        old = json.loads(old_text)
        for label, a, b in (
                ("top-level required", old.get("required"), new.get("required")),
                ("payload required",
                 old["properties"].get("payload", {}).get("required"),
                 new["properties"].get("payload", {}).get("required")),
                ("$defs keys", sorted(old.get("$defs", {})),
                 sorted(new.get("$defs", {})))):
            if a != b:
                print("  %s: %s -> %s" % (label, a, b), file=sys.stderr)
        print("  sha256 %s -> %s"
              % (hashlib.sha256(old_text.encode()).hexdigest()[:12],
                 hashlib.sha256(new_text.encode()).hexdigest()[:12]),
              file=sys.stderr)
    BASE.parent.mkdir(parents=True, exist_ok=True)
    BASE.write_text(new_text)
    print("pinned %s -> %s" % (UPSTREAM, BASE), file=sys.stderr)


def check_base():
    if not BASE.exists():
        print("no pinned base at %s; run --sync-base" % BASE, file=sys.stderr)
        return 1
    if not UPSTREAM.exists():
        print("upstream base not found: %s" % UPSTREAM, file=sys.stderr)
        return 1
    if BASE.read_text() != UPSTREAM.read_text():
        print("upstream %s has drifted from the pinned copy; review the diff, "
              "then run --sync-base and re-check every patch" % BASE_NAME,
              file=sys.stderr)
        return 1
    print("pinned base matches upstream", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk schema is stale")
    ap.add_argument("--sync-base", action="store_true",
                    help="refresh the pinned base copy from the wiki")
    ap.add_argument("--check-base", action="store_true",
                    help="report whether the upstream base has drifted")
    args = ap.parse_args()

    if args.sync_base:
        sync_base()
        return 0
    if args.check_base:
        return check_base()

    if not BASE.exists():
        raise SystemExit("no pinned base at %s; run --sync-base first" % BASE)
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"

    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("STALE: %s does not match make_schema.py / the pinned base"
                  % OUT, file=sys.stderr)
            return 1
        print("schema up to date")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print("wrote %s from %d patches" % (OUT, len(PATCHES)), file=sys.stderr)
    for p in PATCHES:
        print("  %s: %s" % (".".join(str(x) for x in p["path"][-2:]),
                            p["why"].splitlines()[0][:60]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
