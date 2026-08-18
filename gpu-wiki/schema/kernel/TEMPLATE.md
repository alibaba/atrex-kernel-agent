# 记录模板（`clean-1.3`）

> **本文件由 `schema/render_template.py` 从 [`schema.json`](schema.json) 生成，不要手改。**
> schema 是唯一真相源并由 `tools/check_kernel_wiki.py` 的 schema 门强制校验；本文件只是它的人读投影。改字段请改 schema 再重跑生成。

## 记录顶层（8 个 type 一致）

```jsonc
{
  "schema": const 'clean-1.3',
  "id": string,
  "type": strategy \| anti-strategy \| technique-card \| symptom-card \| reference-kernel \| doc \…,
  "level": operator \| cross-operator \| generic,
  "status": active \| superseded \| deprecated,
  "episode_key": string,
  "retrieval": { ... },
  "payload": { ... },
  "evidence": { ... },
  "worth": { ... },
}
```

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `schema` | ● | const 'clean-1.3' |  |
| `id` | ● | string | Deterministic primary key: <vendor>.<product>.<dsl>.<operator_family>.<technique>-<discriminator>. Rebuilding from the same corpus must produce the s… |
| `type` | ● | strategy \| anti-strategy \| technique-card \| symptom-card \| reference-kernel \| doc \… |  |
| `level` | ● | operator \| cross-operator \| generic | Which fallback tier this record lands on: L0 exact operator, L1 sibling transfer, L2 generic. |
| `status` | ● | active \| superseded \| deprecated |  |
| `episode_key` | ● | string | Merge handle '<arch>\|<dsl>\|<operator_family>\|<technique>\|<level>'. Several versions of the same technique on the same operator share it and colla… |
| `retrieval` | ● | object |  |
| `payload` | ● | object | Self-contained actionable knowledge. Shape depends on type; enforced by the allOf branches below. |
| `evidence` | ● | object |  |
| `worth` | ● | object | Everything about how much this record is worth, in one field. Merged on purpose: the agent asks a single question -- is this worth my next version? -… |

## 四层职责

| 层 | 给谁 | 说明 |
|---|---|---|
| `retrieval` | 检索引擎 | 8 个 type 形状统一，可硬过滤。含 `locator`（引擎侧定位符，服务时剥离，agent 看不到） |
| `payload` | agent | **自包含**：只看这一层就能动手。按 type 多态，下面详列 |
| `evidence.summary` | validation/maintenance only | 瓶颈证据、机制指标、测量环境；不向消费 Agent 返回 |
| `evidence.raw` | **只给人** | 去匿名化溯源与绝对 geomean，永不进入服务投影 |
| `worth` | agent + 排序 | 预期收益与由它导出的排序合成一个字段。`rank.score` / `rank.tier` 与 `gain` 服务，`track`（计数 + 语料先验）只在引擎侧 |

## retrieval — 检索引擎用（8 个 type 形状一致）

先按作用域硬过滤，再做文本排序。`locator` 是引擎侧定位符，服务时剥离。

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `scope` | ● | object | Hard identity. Measured evidence never crosses vendor/arch/product/dsl. |
| `generality` | ● | object | Transfer keys. Decide whether a coarser query can still reach this record. |
| `locator` |  | object | ENGINE ONLY. Where the retrieval tool can find the related source inside this repository. Stripped from the agent-facing projection: the agent cannot… |
| `signals` | ● | object | Normalized profiler signals. The agent profiles first, then retrieves with these. |
| `technique_tags` | ● | string[] | Technique labels. Drive recall and dedup ('do not recommend a lever already tried'). |
| `triggers` | ● | string[] | Natural-language conditions: when should this record be considered. |
| `links` |  | → links |  |

`scope` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `vendor` | ● | nvidia \| amd \| generic |  |
| `arch` | ● | ampere \| hopper \| blackwell \| blackwell-ultra \| blackwell-geforce \| cdna3 \| cdna4 … |  |
| `product` |  | string? | Product overlay slug when the record is product-specific (b200, b300, mi300x, mi308x, mi355x, sm120). 'any' or null for architecture- or vendor-gener… |
| `dsl` | ● | cuda \| cutedsl \| flydsl \| gluon \| triton \| aiter \| any |  |
| `operator_family` |  | string? | Precise operator identity (the wiki page slug), e.g. moe-expert-compute. Drives L0 exact hits and provenance isolation. null for records that are not… |
| `operators` |  | string[] | Benchmark operator names this record was measured on. Part of hard identity, and what --operator filters against. |
| `shape_signature` |  | object | Semantic constants only (hidden_size, head_dim, block_scale, num_experts...). Size axes that vary across the benchmark (M/N/K/num_tokens/batch) must … |
| `architectures` |  | string[] | Explicit multi-architecture scope for a record that applies to several architectures at once (an H100->B200 comparison). Empty when the single arch f… |

`generality` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `arch` | ● | ampere \| hopper \| blackwell \| blackwell-ultra \| blackwell-geforce \| cdna3 \| cdna4 … | 'any' promotes the record into the generic tier and makes it reachable across architectures. |
| `language` | ● | cuda \| cutedsl \| flydsl \| gluon \| triton \| aiter \| any |  |
| `workload_family` |  | attention \| conv-vision \| decoder-layer \| gemm-projection \| mask-index \| misc \| ml… | Coarse family. The key that actually carries L1 sibling transfer, because exact operator_family hits are frequently zero. |

`locator` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `kernel_source` |  | string? | Wiki-relative path to the operator's published implementation. |

`signals` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `bottleneck` |  | launch-overhead \| host-sync-stall \| store-bandwidth-bound \| alu-saturated \| tiny-m-g… | Optional on purpose. Absent means the engine classifies from metrics. Never use it as a hard filter: a misclassification would silently hide the righ… |
| `metrics` | ● | object | Normalized numbers used for ranking and for engine-side bottleneck fallback. |
| `shape_regime` | ● | object | Where the record applies along the SIZE axes. R1: size axes live only here, never in scope. |
| `symptoms` | ● | string[] |  |

`links` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `parent` |  | string? | The record this one builds on. |
| `depends_on` |  | string[] |  |
| `conflicts_with` |  | string[] |  |
| `see_also` |  | string[] |  |
| `supersedes` |  | string[] |  |
| `cited_records` |  | string[] | Records this one quotes, such as the measured wins on a technique card or the candidates on a symptom card. |

## evidence — summary 给 agent，raw 只给人

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `summary` | ● | object | Agent-visible. Lets the agent judge how much to trust the record and lets the engine rank it. Absolute latencies live here, not in payload.expected_g… |
| `raw` | ● | object | HUMAN ONLY. Never serialized into any agent-facing output. Holds the de-anonymized provenance: contributor repo, version label, commit, absolute corp… |

`summary` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `bottleneck_evidence` |  | object |  |
| `mechanism_metrics` |  | object | Machine-level numbers that explain WHY the gain happened, and which decide whether it will transfer. Not benefits, so they are deliberately outside p… |
| `confidence` | ● | measured \| inferred \| documented | measured = harness numbers; inferred = reasoned from partial evidence; documented = distilled from a curated documentation page, no harness run. |
| `measured_on` |  | object | Which toolchain / harness produced the numbers. The agent must see this: conclusions can expire when the environment changes. |

`raw` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `source_repo` |  | string? |  |
| `version` |  | string? |  |
| `git_commit` |  | string? |  |
| `detail_file` |  | string? |  |
| `hunk_index` |  | integer? |  |
| `file_paths` |  | string[] |  |
| `page_path` |  | string? | The markdown page this record was distilled from, relative to the wiki root. markdown stays the source of truth; this is the reconciliation link. |
| `page_anchor` |  | string? |  |
| `evidence_extra` |  | object | The long tail of per-step evidence keys that were not normalized into summary. Kept for fidelity, not served. |
| `effect` |  | object | HUMAN ONLY. Absolute geomeans for auditing the extraction. Lives in raw because a microsecond is meaningless without the shape set; the agent gets th… |

## worth — 收益 + 排序（合成一个字段）

agent 只拿 `rank.score`、`rank.tier` 和 `gain`；`track` 与打分分解是引擎侧，服务时剥离——把原始计数交给 agent，等于请它重算一个已经算好的排序。

> Everything about how much this record is worth, in one field. Merged on purpose: the agent asks a single question -- is this worth my next version? -- and the engine needs a single scalar to sort by, so the benefit and the standing derived from it must not live in two places. rank and track are GENERATED; do not hand-edit.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `rank` | ● | object | What the engine sorts by and what the agent reads to judge standing. Only score and tier are served. |
| `gain` | ● | → gain |  |
| `track` |  | object | ENGINE ONLY, never served. The bookkeeping that produced rank: online counters plus the corpus cold start. Withheld from the agent because raw counts… |

`rank` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `score` | ● | number | The single sort key, in [0, 1]. |
| `tier` | ● | proven \| promising \| provisional \| cautionary | Coarse standing, so the agent never has to guess whether 0.80 is high. Derived from the same base-rate-corrected signal as the score, never from raw … |
| `formula` |  | string |  |
| `components` |  | object | ENGINE ONLY. Score decomposition, so --explain can show why a record ranked where it did. |
| `computed_at` |  | string? | ENGINE ONLY. |
| `builder_version` |  | string | ENGINE ONLY. |

`track` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `counters` | ● | object | Cache of feedback/events.jsonl, which is the truth. Rebuildable at any time by rebuild_importance.py. |
| `prior` | ● | object | Cold-start belief derived from the corpus itself, so a never-queried record is not stuck at the bottom. Also absorbs what used to be evidence.summary… |
| `confirm_keys` |  | string[] | ENGINE ONLY. Identity of every incoming record already counted into counters.verified_effective, so re-running a trace scan cannot inflate the count … |

### worth.gain — 预期收益（只用百分比）

> Expected benefit as a list of normalized metrics, so a record can report more than one axis and say which one it was actually optimizing. Absolute latency never appears here: a millisecond means nothing without the shape set and it tempts an agent into comparing across operators, so a latency entry carries only delta_pct and the raw geomeans stay in evidence.raw.effect, which is not served. Non-t…

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `basis` | ● | measured \| reported \| qualitative | measured = we have the harness number; reported = the number comes from someone else's write-up; qualitative = no clean number. |
| `kind` |  | performance \| accuracy \| footprint \| none | Which kind of benefit this is. none for records that claim no benefit of their own, such as a symptom card or a rejected attempt. |
| `primary` |  | → metric_name | Which metric was the actual target of this change, when several moved. |
| `pct` |  | number? | Sign-normalized delta_pct of the primary metric, lifted out of the list so ranking and --min-gain never have to walk metrics[]. Positive is always an… |
| `metrics` |  | [→ gain_metric] | Improvements. Empty when the record claims no gain. |
| `regressions` |  | [→ gain_metric] | The price of the gain, which the corpus does record: a slower compile, or the same metric going the wrong way on one shape bucket. Reading a gain wit… |
| `correctness` |  | object | Proof the gain was not bought with lost precision. |
| `comparable` |  | boolean | false when the percentages are relative to a different baseline than the main chain, so they must not be compared with it. |
| `note` |  | string |  |

`correctness` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `gate` |  | string | Which gate was run. |
| `observed` |  | string | Outcome, e.g. pass, or 16/16 PASS. |
| `max_diff` |  | number? | Largest numerical deviation, when recorded. |

### worth.gain.metrics[] / regressions[] 的元素

> One normalized measurement. delta_pct is sign-normalized so a POSITIVE value always means improvement, whichever way the underlying metric points.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `metric` | ● | → metric_name |  |
| `direction` | ● | lower-better \| higher-better | Which way the raw metric improves. Stated so delta_pct can be read without knowing the metric. |
| `delta_pct` |  | number? | Sign-normalized change: positive = improvement, negative = regression. |
| `before` |  | number? | Only for non-time metrics. A latency entry must omit this; absolute runtimes belong in evidence.summary.effect. |
| `after` |  | number? | Only for non-time metrics. |
| `unit` |  | string | Never a time unit on a benefit metric; the no-absolute-latency gate rejects it. |
| `measured_over` |  | string | The measurement scope: which shapes, which configuration. |
| `note` |  | string |  |
| `source` | ● | harness \| step-record \| profiler \| corpus-aggregate \| pr-body \| inferred | Where the number came from. Required so a no-fabrication check can tell a measured number from an inferred one. |

### metric 词表（封闭）

> Closed vocabulary. latency plus the roofline coordinates (sol / mfu / dram_throughput / compute_throughput / arithmetic_intensity) are the benefit axes; compile_time and memory_footprint exist because the corpus records them as costs. Machine-level diagnostics such as occupancy, register pressure or launch count are NOT benefits and live in evidence.summary.mechanism_metrics.

类型：`latency \| sol \| mfu \| dram_throughput \| compute_throughput \| arithmetic_intensity \…`

## payload 共享块

所有 type 必备 `goal` / `problem`（收益不在这里，见上面的 `worth.gain`）；描述具体改动的类型（`strategy`、`reference-kernel`）额外必备 `trace` 与 `implementation`。

### goal — 一句话：agent 为什么在读这条

> One line telling the agent why this record is in front of it. First thing to read.

类型：`string`

### problem — 解决什么问题（自包含，不依赖 retrieval）

> What this record is about, restated so payload never depends on retrieval. Deliberately redundant with retrieval.scope: retrieval is for the engine, this is for the agent.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `statement` | ● | string | The problem in one sentence: which operator, and what was wrong with it. |
| `operator` |  | string? | Human-readable operator name. |
| `operator_id` |  | string? | Wiki slug, usable as --operator. |
| `dsl` |  | string |  |
| `target` | ● | string | Hardware the numbers were measured on, spelled out. |
| `workload` |  | string | The benchmark contract: what the kernel computes and over which shapes. |
| `shape_contract` |  | object | Semantic constants, the variable axes, and how many shapes were benchmarked. |
| `bottleneck` |  | string? | Normalized bottleneck, when the corpus identified one. |
| `observed_symptom` |  | string | What the profile or bench actually showed. |

### trace — 基于什么方案改进而来

> Lineage. Which solution this record improves on, and which levers were already spent on this chain, so the agent can tell an incremental tweak from a rewrite and does not re-recommend something already tried.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `builds_on` | ● | object |  |
| `techniques_applied_before` |  | string[] | Levers already used earlier in this chain. Do not recommend these again. |
| `technique_here` |  | string[] |  |
| `rediscovered` |  | integer? | anti-strategy: how many independent runs hit this same trap. |
| `n_operators` |  | integer? | anti-strategy: across how many operators. |

`builds_on` 的内部结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `approach` | ● | string | What the previous state of the code did, in words, so one query is enough and no relation id has to be resolved. |
| `is_baseline` |  | boolean | true when this is the first change on top of the untouched reference. |

### implementation — 代码（不含任何仓库路径）

> The code, always verbatim from the corpus; the distiller never writes code of its own. format carries the reading convention, so no separate explanation field is needed. No path ever appears here: the agent has no access to this repository, so a path would be something it cannot open.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `snippet` |  | string | The hunk or source excerpt. |
| `format` | ● | unified-diff (verbatim; '-' = before this step, '+' = after) \| source (verbatim) \| none | Self-describing on purpose: a diff is useless if the agent cannot tell which side is which, and a constant explanation repeated on every record would… |
| `core_kernel_name` |  | string |  |
| `dispatch_snippet` |  | string | Shape-dependent routing, where most tuning knowledge lives. |
| `source_text` |  | string | RESERVED, not produced by tools/query_wiki.py. A serve-time inliner may place a full implementation here; the agent-facing code is implementation.sni… |
| `source_truncated` |  | boolean | SERVE TIME ONLY. true when source_text was capped. |

### cost — 采纳成本

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `edit_scope` | ● | one-line \| epilogue \| kernel-rewrite \| architecture |  |
| `risk` | ● | low \| med \| high |  |
| `risk_basis` |  | string | Why that risk level, in corpus terms. |

### metric_delta — evidence.summary.mechanism_metrics 的元素

> One machine-level diagnostic reading. before/after when the corpus stated both, value when it stated one level, saved when it stated only the amount removed. quote keeps the source sentence because these are regex-extracted from prose and a human must be able to audit them.

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `before` |  | number? |  |
| `after` |  | number? |  |
| `value` |  | number? | A single stated level, when it is not clear whether it is before or after. |
| `saved` |  | number? | Amount removed, when that is all the corpus gave. |
| `delta_pct` |  | number? | Derived, only when before and after are both known. |
| `unit` | ● | string |  |
| `quote` |  | string | Verbatim source sentence, for auditing the extraction. |

## 各 type 的 payload

| type | 记录数 | 必备共享块 | 独有字段 |
|---|---:|---|---|
| **strategy** | 0（预留） | `goal`●, `problem`●, `trace`●, `implementation`●, `cost`● | `change`●, `mechanism`●, `config`, `applies_when`, `next_steps`● |
| **anti-strategy** | 191 | `goal`●, `problem`●, `trace`, `implementation`, `cost` | `attempted`●, `established_fact`, `hypothesis`, `observed`, `root_cause`, `would_retry_if`, `verdict`●, `lesson`● |
| **technique-card** | 452 | `goal`●, `problem`●, `cost`, `implementation` | `technique`●, `what`●, `when`●, `success_rate_pct`, `typical_gain_pct`, `best_gain_pct`, `attempts`, `kept`, `measured_wins`, `caveats`● |
| **symptom-card** | 13 | `goal`●, `problem`● | `likely_causes`, `candidate_techniques`●, `corpus_example`, `measured_on_operators` |
| **reference-kernel** | 0（预留） | `goal`●, `problem`●, `trace`●, `implementation`● | — |
| **doc** | 191 | `goal`●, `problem`●, `implementation` | `title`●, `summary`●, `sections`●, `key_facts` |
| **numerics-rule** | 0（预留） | `goal`●, `problem`● | `rule`●, `rationale`●, `applies_when`, `violation_symptom` |
| **dispatch-rule** | 0（预留） | `goal`●, `problem`● | `rule`●, `rationale`●, `applies_when`, `violation_symptom` |

`●` = 必填。

### strategy（0 条，预留）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `change` | ● | string | What this step changed, verbatim from the corpus. |
| `mechanism` | ● | string | Why it worked, at the machine level (launch cost, HBM traffic, occupancy). Absolute latency restatements are stripped: the number belongs in expected… |
| `config` |  | object | Key knobs plus the range that was actually explored. Empty when the corpus records none; do not infer. |
| `applies_when` |  | string[] | Mechanistic preconditions only. Size conditions live in retrieval.signals.shape_regime. |
| `next_steps` | ● | string[] | Known untried directions, verbatim from open_directions. |

### anti-strategy（191 条）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `attempted` | ● | string | What was tried. |
| `established_fact` |  | object | A negative record is only worth keeping if it states a FACT: under checkable condition C, doing X necessarily yields the bad result. Without this, a … |
| `hypothesis` |  | string | The assumption at the time. Marked when inferred. |
| `observed` |  | string | What actually happened. Empty when the page states no separate result, rather than echoing the lesson. |
| `root_cause` |  | string |  |
| `would_retry_if` |  | string | Under which changed condition this becomes worth retrying. |
| `verdict` | ● | accuracy-gate/ceiling \| api-limitation \| not-worth-it-here \| performance-ceiling | 'unknown' and 'unstable' are deliberately absent: a run that ended without a conclusion has nothing to record as a negative fact. |
| `lesson` | ● | string |  |

取自 `amd.cdna4.flydsl.pitfalls.chunk-gdn-pitfalls.1-silent-corruption-caused-by-a-single-large-smemptr-memref`：

```jsonc
{
 "attempted": "To implement k-LDS double buffering (4 buffers), a natural approach is to allocate a single large memref covering all buffersmq, and select them using a dynamic offset:",
 "verdict": "accuracy-gate/ceiling",
 "lesson": "In FlyDSL, never use a single large SmemPtr memref to simulate multiple logical buffers. Each independent LDS region should have its own SmemPtr instance.",
 "established_fact": {
  "condition": {
   "toolchain": "flydsl"
  },
  "mechanism": "This is a bug in FlyDSL SmemPtr / MLIR memref lowering. Possible causes: - The base pointer + offset calculation for large memrefs may produce integer overflow or incorrect address computation during LLVM lowering - memref alias analysis may assume different regions of a large memref can alias, leading to incorrect load/store reordering - LDS alignment constraints may not be satisfied with a large memref"
 },
 "root_cause": "This is a bug in FlyDSL SmemPtr / MLIR memref lowering. Possible causes: - The base pointer + offset calculation for large memrefs may produce integer overflow or incorrect address computation during LLVM lowering - memref alias analysis may assume different regions of a large memref can alias, leading to incorrect load/store reordering - LDS alignment constraints may not be satisfied with a large memref"
}
```

### technique-card（452 条）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `technique` | ● | string |  |
| `what` | ● | string | What the lever is. |
| `when` | ● | string | When it pays off and when it does not. |
| `success_rate_pct` |  | number? | Share of corpus attempts that were retained. NOT a speedup. |
| `typical_gain_pct` |  | number? | Median percentage gain among the retained wins. |
| `best_gain_pct` |  | number? |  |
| `attempts` |  | integer? |  |
| `kept` |  | integer? | How many of those attempts were retained. |
| `measured_wins` |  | object[] | Best measured applications. Each carries the gain as a percentage and the id of the record with the code. |
| `caveats` | ● | string[] |  |

`measured_wins[]` 的元素结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `operator` |  | string |  |
| `dsl` |  | string |  |
| `gain_pct` |  | number? |  |
| `change` |  | string |  |

取自 `amd.cdna4.gluon.kernel-opt.fused_attention.bottleneck-characteristics`：

```jsonc
{
 "technique": "Bottleneck Characteristics",
 "what": "Flash Attention is typically **Memory Bound** because: - QK matmul output needs to be written back to HBM (or kept in registers) - V load is the primary bandwidth consumer - Softmax exp/log computation is relatively lightweight",
 "when": "Flash Attention is typically **Memory Bound** because: - QK matmul output needs to be written back to HBM (or kept in registers) - V load is the primary bandwidth consumer - Softmax exp/log computation is relatively lightweight",
 "caveats": [],
 "success_rate_pct": null,
 "typical_gain_pct": null
}
```

### symptom-card（13 条）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `likely_causes` |  | string[] |  |
| `candidate_techniques` | ● | object[] | Ordered candidate levers, each with its corpus success rate and typical percentage gain so one query is enough to choose. |
| `corpus_example` |  | string |  |
| `measured_on_operators` |  | object[] | Operators where this really was the measured bottleneck. |

`candidate_techniques[]` 的元素结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `technique` |  | string |  |
| `success_rate_pct` |  | number? |  |
| `typical_gain_pct` |  | number? |  |
| `best_gain_pct` |  | number? |  |

`measured_on_operators[]` 的元素结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `operator` |  | string |  |
| `measured_as` |  | string |  |

取自 `amd.generic.any.symptom.memory-bound`：

```jsonc
{
 "likely_causes": [],
 "candidate_techniques": [
  {
   "technique": "common-causes-and-fixes",
   "success_rate_pct": null,
   "typical_gain_pct": null
  },
  {
   "technique": "core-principles",
   "success_rate_pct": null,
   "typical_gain_pct": null
  },
  {
   "technique": "diagnostic-methods",
   "success_rate_pct": null,
   "typical_gain_pct": null
  },
  {
   "technique": "relationship-with-other-optimizations",
   "success_rate_pct": null,
   "typical_gain_pct": null
  }
 ]
}
```

### reference-kernel（0 条，预留）

无独有字段：需要表达的内容全部由共享块覆盖。

### doc（191 条）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `title` | ● | string |  |
| `summary` | ● | string |  |
| `sections` | ● | object[] | Section-level index so an agent can jump instead of reading the whole page. |
| `key_facts` |  | string[] |  |

`sections[]` 的元素结构：

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `heading` |  | string |  |
| `anchor` |  | string |  |
| `gist` |  | string |  |

取自 `amd.cdna4.flydsl.ref-docs.cdna4-chunk-gdn`：

```jsonc
{
 "title": "FlyDSL Chunk-GDN Optimization (MI355X / gfx950)",
 "summary": "Applicability: backend: flydsl; hardware: amd; topic: reference",
 "sections": [
  {
   "heading": "1. Pipeline Overview",
   "anchor": "1-pipeline-overview",
   "gist": "5 kernels execute in order:"
  },
  {
   "heading": "2. Performance Summary",
   "anchor": "2-performance-summary",
   "gist": "table columns: Kernel, T=4K, Ratio, T=16K, Ratio, T=65K, Ratio, T=262K"
  },
  {
   "heading": "3. Optimization Techniques per Kernel",
   "anchor": "3-optimization-techniques-per-kernel",
   "gist": "fwd_h is the hottest kernel in the pipeline (~55% of total time), implementing the recurrence update for chunked linear attention."
  },
  {
   "heading": "4. General Optimization Takeaways",
   "anchor": "4-general-optimization-takeaways",
   "gist": "All 5 kernels should use the O=3 monkey-patch."
  }
 ]
}
```

### numerics-rule（0 条，预留）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `rule` | ● | string |  |
| `rationale` | ● | string |  |
| `applies_when` |  | string[] |  |
| `violation_symptom` |  | string |  |

### dispatch-rule（0 条，预留）

| 字段 | 必填 | 类型 | 说明 |
|---|:--:|---|---|
| `rule` | ● | string |  |
| `rationale` | ● | string |  |
| `applies_when` |  | string[] |  |
| `violation_symptom` |  | string |  |
