#!/usr/bin/env python3
"""Evidence-density report: what this set can honestly yield.

Runs before any LLM is involved, because the shape of the corpus decides the
shape of the product. Three numbers have veto power:

  citable share    how many candidates have a number in a T1 tool output rather
                   than only in the agent's own notes. Below that line a record
                   may not claim `basis: measured`.
  diff coverage    how many have a verbatim diff. A candidate with none may not
                   become a `strategy` record at all, because such a record is
                   schema-required to carry an implementation.
  already covered  how many versions the committed stores already hold for this
                   operator. Those are skipped outright, and if the overlap is
                   most of the set there is little to add.

Usage: STM_SET=<name> python3 recon.py
"""
import json
import sys
from collections import Counter

import config as c


def load(set_name):
    work = c.work(set_name)
    rows = [json.loads(l) for l in (work / "versions.jsonl").open()]
    meta = json.loads((work / "meta.json").read_text())
    return rows, meta


def tier_of(row):
    tiers = set((row.get("number_tiers") or {}).values())
    for t in ("T1", "T2", "T3"):
        if t in tiers:
            return t
    return None


def classify(rows, unit):
    """What each candidate could become, before any agent sees it."""
    out = Counter()
    for r in rows:
        if r.get("implausible"):
            out["implausible"] += 1
            continue
        if unit == "version-ladder":
            if r.get("reverted"):
                out["anti-strategy (reverted)"] += 1
            elif not r.get("geomean_us"):
                out["no metric"] += 1
            elif not r.get("sha"):
                out["no commit"] += 1
            else:
                out["strategy candidate"] += 1
        else:
            if r["improve_pct_raw"] <= 0:
                out["anti-strategy (regression)"] += 1
            elif r.get("diff_coverage") == "blind":
                out["blind (no diff)"] += 1
            else:
                out["strategy candidate"] += 1
    return out


def existing_coverage(rows):
    """Which operators in this set the committed store already writes about."""
    ops = {r.get("operator_id") for r in rows if r.get("operator_id")}
    hits = {}
    for root in c.COMMITTED_STORES:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name == "index.json":
                continue
            try:
                rec = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            raw = (rec.get("evidence") or {}).get("raw") or {}
            repo = str(raw.get("source_repo") or "")
            for op in ops:
                if op and op in repo:
                    hits.setdefault(op, []).append(
                        (rec.get("id"), raw.get("version")))
    return hits


def main():
    set_name, cfg, _root = c.require_set()
    rows, meta = load(set_name)
    unit = meta["unit"]
    kinds = classify(rows, unit)
    tiers = Counter(tier_of(r) for r in rows)
    coverage = existing_coverage(rows)

    w = []
    a = w.append
    a("# 证据密度报告：%s" % set_name)
    a("")
    a("由 `recon.py` 生成。**先读这份报告再决定蒸馏什么** —— 语料的形状决定产品的形状。")
    a("")
    a("## 语料")
    a("")
    a("| | |")
    a("|---|---|")
    a("| 会话文件 | %d 主 + %d 子 agent |"
      % (meta["n_transcripts"], meta["n_subagent_transcripts"]))
    a("| 解析失败 | %d |" % meta["n_unparsed"])
    a("| 转录格式 | %s |" % meta["formats"])
    a("| 候选单元 | `%s` |" % unit)
    a("| 检测到的型号 | %s |" % meta["products_detected"])
    a("| 检测到的 DSL | %s |" % meta["dsls_detected"])
    a("| 算子（取自 cwd） | %s |" % meta["operators"])
    a("")
    a("声明的默认值：%s。检测优先于声明；`arch_basis` 记录了每条候选实际靠哪一个。"
      % meta["declared"])
    a("")
    if meta.get("note"):
        a("> %s" % meta["note"])
        a("")

    if unit == "version-ladder":
        a("## 版本阶梯")
        a("")
        a("| | |")
        a("|---|---:|")
        a("| `git log` 回显里的版本数 | %d |" % meta["ladder_versions"])
        a("| 能归属到某个会话的版本 | %d |" % meta["sessions_owning_a_version"])
        a("| 不产出版本的会话（规划/profiling/research） | %d |"
          % meta["sessions_without_version"])
        a("| 候选行 | %d |" % len(rows))
        a("| 其中有 geomean | %d |"
          % sum(1 for r in rows if r.get("geomean_us")))
        a("| 其中有 commit | %d |" % sum(1 for r in rows if r.get("sha")))
        a("| 其中 reverted | %d |" % sum(1 for r in rows if r.get("reverted")))
        a("")
        a("一个会话往往就是一个版本：编排 prompt 常常把每次会话限定成一个优化 cycle，"
          "所以阶梯只在**整套语料**的尺度上存在，单个文件里看不到。"
          "不产出版本的会话不是解析失败，而是规划与 profiling 会话，"
          "作为它们所讨论版本的 packet 证据。")
        a("")
    else:
        a("## A/B 候选")
        a("")
        by_basis = Counter(r["delta_basis"] for r in rows)
        a("| 来源 | 数量 |")
        a("|---|---:|")
        for k, n in by_basis.most_common():
            a("| `%s` | %d |" % (k, n))
        a("| **合计** | **%d** |" % len(rows))
        a("")
        a("这套语料没有版本阶梯，所以候选单元是**一次有度量的 A/B**。"
          "其中 `*-variant` 是转录里现成的对照（同一次输出里同时打印了两侧），"
          "`before-after` 是同一条 benchmark 命令在一次改动前后各跑一遍。"
          "codex 语料里后者几乎不存在：每次跑的都是新写的内联脚本，"
          "实测一个转录里 61 个不同的 benchmark 身份、0 个重复。")
        a("")

    a("## 可蒸馏池")
    a("")
    a("| 分类 | 数量 |")
    a("|---|---:|")
    for k, n in kinds.most_common():
        a("| %s | %d |" % (k, n))
    a("")

    a("## 数字的信任档位")
    a("")
    a("| 档 | 含义 | 候选数 | `gain.basis` 上限 |")
    a("|---|---|---:|---|")
    a("| T1 | benchmark / profiler 的工具输出 | %d | `measured` |"
      % tiers.get("T1", 0))
    a("| T2 | agent 回读自己写的笔记（`cat NOTES.md`、`git log`、`Read vN.json`） | %d | `reported` |"
      % tiers.get("T2", 0))
    a("| T3 | agent 直接写入的结构化字段 | %d | `reported` |" % tiers.get("T3", 0))
    a("| — | 没有可引用的数字 | %d | `qualitative` |" % tiers.get(None, 0))
    a("")
    a("T4（agent 散文与 thinking）与 T5（编排 prompt）**不进证据池**："
      "前者会让 agent 自己编的数字成为它自己的证据，"
      "后者写着目标百分比，会给任何接近阈值的说法发通行证。")
    a("")

    a("## diff 可见性")
    a("")
    cov = Counter(r.get("diff_coverage") for r in rows)
    a("| 覆盖 | 数量 | 后果 |")
    a("|---|---:|---|")
    a("| `full`（有逐字 diff） | %d | 可以做 `strategy` |" % cov.get("full", 0))
    a("| `partial` | %d | 可以做 `strategy`，snippet 需人工确认 |"
      % cov.get("partial", 0))
    a("| `blind`（改动未留下 diff） | %d | **不得**做 `strategy` |"
      % cov.get("blind", 0))
    a("")
    a("盲区来自 shell 写文件（`cat >`/`tee`）：这类改动不产生 patch 事件。"
      "`diff-coverage` 门把它记账而不是默许。")
    a("")

    a("## 与主库的重叠")
    a("")
    skipped = meta.get("wiki_skipped") or []
    a("按用户决定：**碰撞版本直接跳过**，连 packet 都不生成。")
    a("")
    a("| | |")
    a("|---|---:|")
    a("| 因已被主库覆盖而跳过的版本 | %d |" % len(skipped))
    a("")
    if skipped:
        a("| 版本 | 已有记录 |")
        a("|---|---|")
        for s in skipped:
            a("| `%s` | `%s` |" % (s["dedup_key"],
                                   ", ".join(str(x) for x in s["covered_by"][:2])))
        a("")
    if coverage:
        a("主库已就本套算子写过的记录（同算子、可能不同 run）：")
        a("")
        for op, recs in sorted(coverage.items()):
            vers = sorted({str(v) for _i, v in recs if v})
            a("- `%s`：%d 条，版本 %s" % (op, len(recs), ", ".join(vers[:14])))
        a("")

    a("## 被排除的归档条目")
    a("")
    a("| 条目 | 原因 |")
    a("|---|---|")
    for name, why in sorted(meta.get("excluded_from_archive", {}).items()):
        a("| `%s` | %s |" % (name, why))
    a("")

    path = c.reports(set_name) / "recon.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(w) + "\n")

    print("set        %s (%s)" % (set_name, unit))
    print("candidates %d" % len(rows))
    for k, n in kinds.most_common():
        print("  %-28s %d" % (k, n))
    print("tiers      %s" % {str(k): v for k, v in tiers.items()})
    print("diff       %s" % {str(k): v for k, v in cov.items()})
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
