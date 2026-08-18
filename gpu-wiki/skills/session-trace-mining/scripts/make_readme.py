#!/usr/bin/env python3
"""Summarise a finished store into kernel_wiki/session_trace/<set>/README.md.

Generated from the records and the work files, so it stays accurate instead of
drifting as records accumulate. Re-run after every distillation batch.

Usage: STM_SET=<name> python3 make_readme.py
"""
import json
import sys
from collections import Counter

import config as c


def load(set_name):
    work = c.work(set_name)
    meta = json.loads((work / "meta.json").read_text())
    rows = [json.loads(l) for l in (work / "versions.jsonl").open()]
    seg_path = work / "segments.jsonl"
    segs = [json.loads(l) for l in seg_path.open()] if seg_path.is_file() else []
    recs = []
    for p in sorted(c.records(set_name).rglob("*.json")):
        if p.name == "index.json":
            continue
        try:
            recs.append((p, json.loads(p.read_text())))
        except json.JSONDecodeError:
            continue
    return meta, rows, segs, recs


def main():
    set_name, cfg, _root = c.require_set()
    meta, rows, segs, recs = load(set_name)

    w = []
    a = w.append
    a("# %s 会话优化经验库" % set_name)
    a("")
    a("由 `skills/session-trace-mining` 从 AI 编码 agent 的会话转录蒸馏而成。"
      "本文件由 `make_readme.py` 生成，请勿手改。")
    a("")
    a("## 语料")
    a("")
    a("- 会话文件：%d 个主转录 + %d 个子 agent 转录，解析失败 %d 个"
      % (meta["n_transcripts"], meta["n_subagent_transcripts"],
         meta["n_unparsed"]))
    a("- 转录格式：%s；候选单元：`%s`" % (meta["formats"], meta["unit"]))
    a("- 算子（取自会话的工作目录）：%s" % meta["operators"])
    a("- 检测到的硬件 / DSL：%s / %s"
      % (meta["products_detected"], meta["dsls_detected"]))
    a("- 候选 %d 条，切分出 segment %d 条，已蒸馏记录 **%d** 条，schema `%s`"
      % (len(rows), len(segs), len(recs), c.DERIVED_NAME))
    if meta.get("note"):
        a("")
        a("> %s" % meta["note"])
    a("")

    a("## 记录组织")
    a("")
    a("```")
    a("records/<type>/<vendor>/<arch>/<dsl>/<operator_family>/<id>.json")
    a("```")
    a("")
    a("四级目录与记录自身的 `retrieval.scope` 严格一致（由 layout 门强制），"
      "因此可以直接按类型、架构、DSL、算子族定位。")
    a("")

    if recs:
        cells = Counter()
        for _p, r in recs:
            sc = (r.get("retrieval") or {}).get("scope") or {}
            gen = (r.get("retrieval") or {}).get("generality") or {}
            cells[(r.get("type"), sc.get("arch"), sc.get("dsl"),
                   gen.get("workload_family"))] += 1
        a("| type | arch | dsl | family | 记录数 |")
        a("|---|---|---|---|---:|")
        for (t, arch, dsl, fam), n in sorted(cells.items(),
                                             key=lambda kv: -kv[1]):
            a("| %s | %s | %s | %s | %d |" % (t, arch, dsl, fam, n))
        a("")

        a("## 记录清单")
        a("")
        a("| id 尾段 | type | tier | 收益 | 基础 | shape 数 |")
        a("|---|---|---|---:|---|---:|")
        for _p, r in sorted(recs, key=lambda kv: kv[1].get("id") or ""):
            worth = r.get("worth") or {}
            gain = worth.get("gain") or {}
            rank = worth.get("rank") or {}
            reg = (gain.get("regressions") or [{}])[0]
            pct = gain.get("pct")
            shown = ("%.2f%%" % pct) if pct is not None else (
                ("%.2f%%" % reg["delta_pct"]) if reg.get("delta_pct") is not None
                else "—")
            n_shapes = ((r.get("payload") or {}).get("problem") or {}).get(
                "shape_contract", {}).get("n_benchmarked_shapes")
            a("| `%s` | %s | %s | %s | %s | %s |"
              % ((r.get("id") or "").rsplit(".", 1)[-1], r.get("type"),
                 rank.get("tier"), shown, gain.get("basis"), n_shapes or "—"))
        a("")

        tiers = Counter((r.get("worth") or {}).get("rank", {}).get("tier")
                        for _p, r in recs)
        bases = Counter((r.get("worth") or {}).get("gain", {}).get("basis")
                        for _p, r in recs)
        a("| tier | 条数 | | basis | 条数 |")
        a("|---|---:|---|---|---:|")
        keys = list(tiers) + [None] * max(0, len(bases) - len(tiers))
        bkeys = list(bases) + [None] * max(0, len(tiers) - len(bases))
        for tk, bk in zip(keys, bkeys):
            a("| %s | %s | | %s | %s |"
              % (tk or "", tiers.get(tk, "") if tk else "",
                 bk or "", bases.get(bk, "") if bk else ""))
        a("")

    a("## 怎么读一条记录")
    a("")
    a("| 层 | 给谁 | 内容 |")
    a("|---|---|---|")
    a("| `retrieval` | 检索引擎 | 作用域硬过滤、症状、技法标签、触发情境 |")
    a("| `payload` | agent | 自包含：目标、问题、基于什么改进而来、怎么改、机制 |")
    a("| `evidence.summary` | agent | 置信度、测量环境、瓶颈证据 |")
    a("| `evidence.raw` | **只给人** | 会话溯源：语料集名、相对路径、行号、行摘要 |")
    a("| `worth` | agent + 排序 | `rank`（分数 + 可信度分类）、`gain`（只用百分比） |")
    a("")

    a("## 可信度边界")
    a("")
    a("- 数字分三档：**T1** 是 benchmark / profiler 的工具输出，"
      "**T2** 是 agent 回读自己写的笔记，**T3** 是 agent 写入的结构化字段。"
      "`gain.basis=measured` 只在有 T1 段时才允许（由 `evidence-tier` 门强制）。")
    a("- agent 的散文与编排 prompt **不进证据池**：前者会让它自己编的数字成为自己的证据，"
      "后者写着目标百分比。")
    a("- `gain.comparable` 恒为 false：这是 sibling run，与主库主链不可比。")
    a("- `worth.gain` 里没有绝对时间；绝对值只在 `evidence.raw.effect` 里，服务时剥离。")
    a("- 与主库 `(算子, 版本)` 碰撞的版本**直接跳过**，本套跳过了 %d 个。"
      % len(meta.get("wiki_skipped") or []))
    a("")
    a("详见 `reports/recon.md`（这套语料能诚实产出什么）与 `reports/partition.md`"
      "（每条记录是怎么切出来的），schema 见 "
      "`skills/session-trace-mining/assets/schema/%s.schema.json`。" % c.DERIVED_NAME)
    a("")

    path = c.store(set_name) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(w) + "\n")
    print("records %d -> %s" % (len(recs), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
