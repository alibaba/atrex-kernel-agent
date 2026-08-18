# 蒸馏简报（下发给蒸馏 agent 时逐字复制，替换 `<>` 占位）

## 任务

把 `<N>` 个 packet 各写成一条 `session-trace-1.0` 记录，落到指定路径，并让
`validate_store.py` 十六道门全绿。

你**看不到**会话原文，只看到 packet。这是设计使然：packet 里填不出来的字段就必须留空，
不许从常识补。

## 背景

语料是 AI 编码 agent 的会话转录（GPU 算子优化过程）。这些记录会被另一个 agent 检索，
用来决定下一步怎么优化某个 kernel，所以每条记录必须能独立回答四个问题：
为什么读它、解决什么问题、基于什么改进而来、预期收益多少。

- Python：`python3`（`schema` 门需要 `jsonschema`）
- scripts：`<SKILL>/scripts`
- packet 目录：`/tmp/session-trace-mining/<SET>/packets/`
- 产物根：`<STORE>`（记录写到 `packet.target.output_dir` 下，文件名 `<packet.target.id>.json`）
- 参考样板：`<EXAMPLE_RECORD>` —— 本库里已过全部门禁的那一条，路径形如
  `<STORE>/records/<type>/<vendor>/<arch>/<dsl>/<family>/<id>.json`，字段填法照它。
  第一批蒸馏时还没有样板：先写一条、过完门禁，再把它作为后续批次的样板下发。

## packet 各字段怎么用

| 字段 | 用途 |
|---|---|
| `evidence_text` | **唯一可引用数字的地方**。每段带 `### [T1\|T2\|T3 tool-output line N]` 头。T1=benchmark/profiler 输出，T2=agent 回读自己写的笔记，T3=agent 写入的结构化字段。 |
| `<seg_id>.diff` | 逐字代码改动。`implementation.snippet` **必须**逐行出自这里。 |
| `measurement` | 抽取器算好的 before/after/改善百分比/shape 数。`worth.gain` 的数字用它。 |
| `narrative` | 运行自己的叙述（subject、action_description、profile_evidence、pitfall、open_directions）。写 `mechanism` 和 `next_steps` 的主要原料。 |
| `scope` / `target` | 直接抄进 `retrieval.scope` 与落盘路径。别自己推断架构。 |
| `session` | **整块原样抄进** `evidence.raw.session`。一个字都别改——provenance 门会逐行核对摘要。 |

## 铁律（都由门机械检查）

1. **数字只能来自 `evidence_text` 或 `measurement`。** 门会把 `worth.gain` 里每个数
   （含 `note`、`correctness`、`measured_over`）拿去和证据比对。允许的推导只有两种：
   由 before/after 算百分比，或由 `speedup=Nx` 算百分比。其它一律算编造。
   **`<seg_id>.diff` 只是代码，不是数字来源**：它不在审计池里。从 diff 里读到的
   迭代次数、常量、tile 大小都不能作为 `worth.gain` 里任何数字的依据；
   要说这些就写进 `mechanism`。
2. **`snippet` 必须逐字出自 `<seg_id>.diff`。** 改写过的片段比没有更糟：它看着权威，
   但编译不过。`format` 写 `"unified-diff (verbatim; '-' = before this step, '+' = after)"`。
3. **`worth.gain` 里不许出现绝对时间。** 微秒数脱离 shape 集合没有意义。
   绝对值放 `evidence.raw.effect.new_geomean_us` / `parent_geomean_us`（只给人看）。
4. **`gain.pct` 必须等于 primary 指标的 `delta_pct`。**
5. **`gain.source_kind` 恒为 `"agent-session"`**，`basis=measured` 只有在 packet
   有 T1 段时才允许，否则填 `reported`。
6. **agent 可见层不许出现路径与人名。** `/home/...`、`/root/...`、`rollout-*.jsonl`
   一律不能进 `payload` 或 `retrieval`。要提源文件就说「the kernel source」。
   `evidence.raw` 是唯一豁免层。
7. **`diff_coverage=blind` 不许做 `strategy`。** packet 已经写好了 `record_type`，别改。
8. **`retrieval.links` 留空对象 `{}`**，除非引用的 id 真的存在于本库。

## 字段填写要点

- `id` / `type` / `level`(`operator`) / `status`(`active`) 照 packet。
  `episode_key` = `<arch>|<dsl>|<operator_slug>|<technique_snake>|<level>`。
- `retrieval.signals.symptoms`：用小写下划线的归一化症状名，别抄整句日志。
- `retrieval.triggers`：写「什么情况下该读这条」，三条以内，用第二人称情境句。
- `payload.problem.statement`：**说清楚为什么慢**，不是重复标题。
- `payload.mechanism`：**机器层面的因果**——为什么这个改动会让它变快。
  这是整条记录最有价值的字段，也是最容易写空的。写不出机制就说明 packet 证据不足，
  在报告里说明，别编。
- `payload.trace.builds_on.approach`：**用文字描述前一版方案**（packet 的
  `narrative.builds_on`），不要写 id。
- `payload.next_steps`：只写 packet 里真有的方向（`narrative.open_directions`）。
- `worth.rank`：固定占位 `{"score": 0.0, "tier": "provisional", "builder_version": "session-trace-0.1"}`，
  之后由 `score_records.py` 统一算。
- `worth.gain.comparable`：**false**。这是 sibling run，与主链不可比。

## 交付与自检

```bash
cd <SKILL>/scripts
STM_SET=<SET> python3 validate_store.py --verbose
```

迭代到全绿。**不要修改 `scripts/` 下任何文件**，也不要为了过门放宽门。
门看起来不对时，用注入法验证：改坏一条记录，确认是预期那道门开火。

最后交一份 ≤400 词的报告，必须包含：

1. 写了哪几条，各是什么类型；
2. **哪些字段因为 packet 证据太薄而留空**（这是改进流水线的主要信号）；
3. **被哪道门卡过、卡了几轮**；
4. 你认为哪条记录的价值最低，为什么。

一个说「毫无困难」的批次要当成可疑：它更可能是没认真核对证据。
