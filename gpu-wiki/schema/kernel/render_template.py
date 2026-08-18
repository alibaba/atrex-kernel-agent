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
    "gain_metric": "worth.gain.metrics[] / regressions[] 的元素",
    "metric_name": "metric 词表（封闭）",
    "metric_delta": "metric_delta — evidence.summary.mechanism_metrics 的元素",
    "goal": "goal — 一句话：agent 为什么在读这条",
    "links": "links — 引擎侧 id 图（服务时剥离）",
    "problem": "problem — 解决什么问题（自包含，不依赖 retrieval）",
    "trace": "trace — 基于什么方案改进而来",
    "implementation": "implementation — 代码（不含任何仓库路径）",
    "gain": "worth.gain — 预期收益（只用百分比）",
    "cost": "cost — 采纳成本",
    "relations": "relations — 内部 id，便利项",
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


def one_line(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def field_rows(props: dict, required: set, defs: dict) -> list[str]:
    rows = ["| 字段 | 必填 | 类型 | 说明 |", "|---|:--:|---|---|"]
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
    out = ["", "`%s` 的内部结构：" % name, ""]
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

    L = ["# 记录模板（`clean-1.3`）", "",
         "> **本文件由 `schema/render_template.py` 从 "
         "[`schema.json`](schema.json) 生成，不要手改。**",
         "> schema 是唯一真相源并由 `tools/check_kernel_wiki.py` 的 schema 门强制校验；"
         "本文件只是它的人读投影。改字段请改 schema 再重跑生成。", ""]

    # ---------------------------------------------------------------- 顶层
    L += ["## 记录顶层（8 个 type 一致）", "", "```jsonc", "{"]
    for name, spec in schema["properties"].items():
        if name in ("retrieval", "payload", "evidence", "worth"):
            L.append('  "%s": { ... },' % name)
        else:
            L.append('  "%s": %s,' % (name, typename(spec, defs)))
    L += ["}", "```", ""]
    L += field_rows(schema["properties"], set(schema["required"]), defs) + [""]

    # ---------------------------------------------------------------- 四层
    L += ["## 四层职责", "",
          "| 层 | 给谁 | 说明 |", "|---|---|---|",
          "| `retrieval` | 检索引擎 | 8 个 type 形状统一，可硬过滤。含 `locator`（引擎侧定位符，"
          "服务时剥离，agent 看不到） |",
          "| `payload` | agent | **自包含**：只看这一层就能动手。按 type 多态，下面详列 |",
          "| `evidence.summary` | validation/maintenance only | 瓶颈证据、机制指标、测量环境；不向消费 Agent 返回 |",
          "| `evidence.raw` | **只给人** | 去匿名化溯源与绝对 geomean，永不进入服务投影 |",
          "| `worth` | agent + 排序 | 预期收益与由它导出的排序合成一个字段。"
          "`rank.score` / `rank.tier` 与 `gain` 服务，`track`（计数 + 语料先验）只在引擎侧 |", ""]

    # ------------------------------------------------- retrieval / evidence
    for layer, title, blurb in (
        ("retrieval", "retrieval — 检索引擎用（8 个 type 形状一致）",
         "先按作用域硬过滤，再做文本排序。`locator` 是引擎侧定位符，服务时剥离。"),
        ("evidence", "evidence — summary 给 agent，raw 只给人", ""),
        ("worth", "worth — 收益 + 排序（合成一个字段）",
         "agent 只拿 `rank.score`、`rank.tier` 和 `gain`；`track` 与打分分解是引擎侧，"
         "服务时剥离——把原始计数交给 agent，等于请它重算一个已经算好的排序。"),
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
                    L.append("类型：`%s`" % typename(sub, defs))
                L.append("")

    # ---------------------------------------------------------- 共享块
    L += ["## payload 共享块", "",
          "所有 type 必备 `goal` / `problem`（收益不在这里，见上面的 `worth.gain`）；"
          "描述具体改动的类型（`strategy`、`reference-kernel`）额外必备 `trace` 与 "
          "`implementation`。", ""]
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
            L.append("类型：`%s`" % typename(spec, defs))
        L.append("")

    # ------------------------------------------------------- 各 type
    L += ["## 各 type 的 payload", ""]
    L += ["| type | 记录数 | 必备共享块 | 独有字段 |", "|---|---:|---|---|"]
    for rtype in TYPE_ORDER:
        payload = branches[rtype]
        props = payload.get("properties") or {}
        required = set(payload.get("required") or [])
        shared = [n for n in props if n in DEF_OF_FIELD]  # noqa: F841
        own = [n for n in props if n not in DEF_OF_FIELD]
        count = counts.get(rtype, 0)
        L.append("| **%s** | %s | %s | %s |" % (
            rtype, count if count else "0（预留）",
            ", ".join("`%s`%s" % (n, "●" if n in required else "") for n in shared),
            ", ".join("`%s`%s" % (n, "●" if n in required else "") for n in own) or "—"))
    L += ["", "`●` = 必填。", ""]

    for rtype in TYPE_ORDER:
        payload = branches[rtype]
        props = payload.get("properties") or {}
        required = set(payload.get("required") or [])
        own = {n: s for n, s in props.items() if n not in DEF_OF_FIELD}
        count = counts.get(rtype, 0)
        L += ["### %s（%s）" % (rtype, ("%d 条" % count) if count
                                 else "0 条，预留"), ""]
        if own:
            L += field_rows(own, required, defs)
            for name, sub in own.items():
                if sub.get("type") == "array" and (sub.get("items") or {}).get("properties"):
                    L += ["", "`%s[]` 的元素结构：" % name, ""]
                    L += field_rows(sub["items"]["properties"],
                                    set(sub["items"].get("required") or []), defs)
        else:
            L.append("无独有字段：需要表达的内容全部由共享块覆盖。")
        example = example_for(rtype)
        if example:
            L += ["", "取自 `%s`：" % example["id"], "", "```jsonc",
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
