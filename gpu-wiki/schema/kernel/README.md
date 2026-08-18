# Schema — `kernel_wiki`(优化经验库)

The schema is the **single source of truth** for every record in this store.
`tools/check_kernel_wiki.py` 的 schema 门校验库内每一条记录;索引与检索工具都在它
下游。本库**没有 markdown**——记录本身就是唯一真相源。

| 文件 | 角色 |
|---|---|
| [`schema.json`](schema.json) | 唯一真相源(JSON Schema, draft 2020-12)。当前版本 `clean-1.3` |
| [`TEMPLATE.md`](TEMPLATE.md) | 逐字段人读参考,**由 schema 生成**,勿手改 |
| [`render_template.py`](render_template.py) | `schema.json` → `TEMPLATE.md` 渲染器 |

## 版本策略

只有**一个** `schema.json`,版本由 **git** 区分——不再用 `clean-1.N.schema.json`
这样的文件名后缀。`schema.json` 内部 `schema` 常量记录当前语义版本(现为
`clean-1.3`),记录的 `schema` 字段与之匹配,由 schema 门强制。改 schema 就直接
编辑 `schema.json` 并提交,git 历史即版本史。

## 记录的四层

每条记录分四层,各服务不同消费者,**不可混用**:

| 层 | 给谁 | 内容 | 服务时 |
|---|---|---|---|
| `retrieval` | 检索引擎 | 8 个 type 形状统一,可硬过滤:`scope`(vendor/arch/product/dsl/operator_family)、`generality`、`signals`、`technique_tags`、`triggers`;另有引擎侧 `locator`(源码位置)、`links`(内部 id 图) | `locator`/`links` **剥离** |
| `payload` | agent | **自包含**的可执行知识,按 type 多态;代码逐字取自语料 | 保留 |
| `evidence` | `summary` 给 agent、`raw` **只给人** | `summary`:瓶颈证据、机制指标、测量环境;`raw`:去匿名化溯源 + 绝对 geomean | `raw` **剥离** |
| `worth` | agent + 排序 | `gain`(预期收益,归一化百分比)、`rank`(`score` + `tier`)、`track`(计数 + 语料先验) | 只服务 `rank.score`/`rank.tier`/`gain`;`track` 与 `rank.components` **剥离** |

**服务投影**(`tools/query_wiki.py --emit-json`)构造性地剥离:`retrieval.locator`、
`retrieval.links`、`evidence.raw`、`worth.track`、`worth.rank.components`。这五样
要么是引擎账本、要么是只给人的溯源,agent 不该也不需要看到。

## payload 自包含契约

agent 一次查询、一条记录,不回看 `retrieval`、不解析任何 id、不发第二次查询,
就能回答:为什么读这条(`goal`)、解决什么问题(`problem`)、基于什么方案改进
(`trace.builds_on.approach`)、怎么改涨多少(`change`/`mechanism`/`implementation`
+ `worth.gain`)。由 self-contained 门强制。

## 八种记录类型

`strategy`(保留的优化步骤)、`anti-strategy`(被推翻的尝试,负面证据 —— 必须是**既定事实**:可检验条件 + 因果机制 + 结论性 verdict,三条全满足,见 `TEMPLATE.md` 该节)、
`reference-kernel`(可直接读的实现)、`technique-card`(手法全语料成败统计)、
`symptom-card`(症状→候选手法入口)、`doc`(硬件事实)、`numerics-rule` /
`dispatch-rule`(预留)。逐 type 字段见 `TEMPLATE.md`。

## 收益模型(`worth.gain`)

归一化指标 list,词表封闭:`latency` + roofline 坐标(`sol`/`mfu`/`dram_throughput`/
`compute_throughput`/`arithmetic_intensity`)+ `compile_time`/`memory_footprint`。
`delta_pct` 符号归一(正数恒为改善);`pct` 抬到顶层供排序/`--min-gain` 直接读;
**绝对延迟永不进入**(留在 `evidence.raw.effect`),每条指标必带 `source`。

## 重新生成

```bash
python3 schema/render_template.py            # schema.json -> TEMPLATE.md
python3 schema/render_template.py --check     # 检查 TEMPLATE.md 是否与 schema 同步
python3 tools/check_kernel_wiki.py --full    # 9 道门,含 schema 校验
```


## 本库与测量版的差异(开源说明)

`schema` 常量仍是 `clean-1.3`,以便挖掘 skill 与 `wiki-gate` 准入工具在两边通用。
差异只在**文档派生的记录能诚实填出什么**:

| 处 | 说明 |
|---|---|
| `evidence.summary.confidence` | 增加 `documented` 档:来自策展文档页,不是 harness 实测 |
| `worth.gain` | 统一 `basis: qualitative`、`pct: null`——文档没有实测收益,不编数字 |
| `anti-strategy.trace` | 不再必填:一条文档记载的陷阱没有可回溯的实测阶梯 |
| `scope` 词表 | 扩到本 wiki 覆盖的全部 vendor / arch / DSL |
| `scope.architectures` | 新增:承载一页同时适用于多个架构的显式作用域 |
| `technique-card.implementation` | 新增(可选):本库的手法卡是按文档章节切分的内容卡,逐字带该章节代码 |

后续由 trace / session 挖掘出的记录会正常填满 `worth.gain` 与 `evidence` 的实测字段。
