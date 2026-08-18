# anti-strategy 既定事实判定说明书

对 404 条历史 anti-strategy 记录做审查时使用的判定规则。规则来自前 38 条逐条精读后归纳,写下来是为了让后续批量判定与前 38 条**同标准**,并留下可审计依据。

## 三条硬判据(全满足才留库)

| # | 判据 | schema 落点 |
|---|---|---|
| 1 | 可检验条件 C:架构 / shape 区间 / dtype / 工具链之一 | `established_fact.condition` 至少一个非空 |
| 2 | 因果机制:为何在该条件下**必然**失败 | `established_fact.mechanism` ≥ 40 字 |
| 3 | verdict 有结论 | 不能是 `unknown` / `unstable`(已从 schema 枚举移除) |

判据 1 的关键约束:**「这个算子」不算条件**。一个手法在某个算子上失败是观察,不是定律。

## 算「机制」的表述(→ backfill)

前 38 条里所有被回填的记录都属于以下五型之一:

**① 定量对冲** —— 明确指出「省下的 < 付出的」,两边都有数字
> 「padding M 901→960 需要拷贝 A 矩阵,拷贝 8.22us > kernel 节省 2.1us」
> 「torch.bmm 的 Python dispatch 开销超过它省掉的 transpose(+0.5us@N=256,+5us@N=1024)」

**② 资源/容量硬限** —— 引用具体容量或规格得出必然结果
> 「N=8192 时 128MB 缓冲 > B200 的 50MB L2,transpose 必须读 HBM,23us vs 暖态 8us」
> 「2-CTA cluster 的 tile 需要 m≥256,M=128/256 时部分 cluster 空转」

**③ API / 语义限制** —— 工具或库在语义上不支持,不是调优不足
> 「cuDNN-FE 在 SM100 只接受 NHWC/RowMajor epilogue,channels_last 转换无法省略」
> 「batched-strided GEMM 无法表达跨完整 2048 维的归约,batch 化会算错(error=288)」
> 「torch.compile 把 weight data pointer 烘进图里,基于指针的缓存不可能成立」

**④ 微架构因果链** —— 从硬件行为推到性能结果
> 「PADDED_M=8 → 活跃 warp 更少 → ILP 下降 → load latency stall 超出 cp.async 流水深度」

**⑤ 度量模型限制** —— 指标本身决定了该类优化不可见(工具需写进 condition.toolchain)
> 「GPU 流水已把 CPU launch 开销与 kernel 执行重叠,CUPTI kernel-span 无法体现主机侧优化」
> 「kernel 是 dispatch-bound,主机 ctypes 开销 ~38us ≈ GPU 时间,GEMM 级 2-7% 改善被完全掩盖」

## 不算机制的表述(→ demote)

**Ⓐ 穷尽记账** —— 「找过了,没有新的」
> 「179th consecutive dead-end」「所有查询 New?=No」「STALL_COUNT=49」「ORCHESTRATOR QUIESCE RECOMMENDED」

**Ⓑ 纯对比数字** —— 有条件、可复现,但**没说为什么**
> 「Triton GEMM 比 cuBLASLt 慢 48-122%」「cuDNN FP16 比 CUTLASS 慢很多」
> 按标准 A 一律降级。这类事实有价值但缺机制,退暂存区等补齐。

**Ⓒ 噪声区间** —— 结论落在测量噪声内,即无结论
> 「1.2% 改善但多 seed 区间与 HEAD 重叠」「flat-within-noise」「+0.6% within noise」

**Ⓓ 方法学笔记** —— 与具体算子无关的工作方法
> 「sub-5% geomean delta 需 ≥2 次同 session 运行」「md5 未变时可复用 profile」
> 例外:若同一条记录另有真机制(如「瓶颈在闭源 nvjet 二进制内,主机层不可达」),按该机制回填。

**Ⓔ 断言式天花板** —— 只宣布到顶,不解释
> 「confirmed mma_v2 Ampere ceiling」「kernel at hardware floor」「at strong local optimum」

**Ⓕ 推测措辞** —— 机制带 may / might / possibly
> 「the 3-element inner loop may be less efficient than the grid-stride loop」

## 边界判例(前 38 条中实际遇到的)

| 情形 | 判定 | 依据 |
|---|---|---|
一条记录试了 8 个方案,只有 1 个有机制 | **backfill**,`established_fact` 只描述那 1 个 | 门要求的是「这条记录确立了什么」,不是「它试过的一切」 |
lesson 是方法学,但 attempted 里有真机制 | **backfill**,按 attempted 的机制填 | 机制存在即可,位置不影响其为事实 |
机制在 `attempted` 而非 `root_cause` | **backfill** | 历史记录普遍 `root_cause` 为空,机制散落在 attempted |
条件是「所有 shape 都失败」 | **backfill**,`shape_regime` 写实测范围 | 可检验;但机制必须另外成立 |
度量工具属性当条件 | **backfill**,写 `condition.toolchain` | 指标语义是版本化、可检验的 |
`rediscovered` 高但无机制 | **demote** | 复现次数不能替代机制(标准 A) |

## 回填纪律

- 机制**只能取自该记录自身的散文**,不得跨记录借用数字或结论
- 不得凭空补充记录里没有的因果解释
- 回填内容须过 `validate_fact()`:condition 至少一个非空且键合法、mechanism ≥40 字且不匹配空洞措辞黑名单
- 回填被 `validate_fact` 拒绝时,自动改为降级(见 `audit_anti_strategy.py` 的 `rejected_backfill` 分支)

## 降级处置

不合格记录**不删除**,移回挖掘 skill 的暂存区,置 `status: active` 并加 `demoted_from_main` / `demoted_reason` / `demoted_at`。将来补上机制或被多次独立复现后,可重新过 wiki-gate 入库。
