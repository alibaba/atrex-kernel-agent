# Light 延迟数据来源

`light_latency_status.py` 是只读工具。它自动发现 Campaign 与 pooled-search
trajectory，不提交、取消或查询新的 Agate Job。

## Kernel 身份

公开身份始终是 Agent 工作区 clean HEAD 的完整 Git commit ID，字段名为
`kernel_commit_id`。Supervisor 私有归档身份不向 Agent 或可观测性结果暴露。
Measurement Store 保存源码树快照、`kernel_artifact_digest`、请求、响应与
Gateway Record，并由 `MeasurementStore.read()` 校验完整性。

## `accepted_best`

接受事实只来自 Supervisor 私有 Journal：

```text
journals/episodes/e*/journal.json
└── kernel_submissions[]
    ├── accepted = true
    ├── kernel_commit_id
    └── evaluation_evidence.gateway_record_id
```

工具读取引用的完整 Evaluate Record，并要求：

- Record 是已完成、可缓存、未截断的标准 full Evaluate；
- 所有 Shape 正确性通过且性能数据完整可解析；
- Record 的 candidate `kernel_id` 与提交的 Agent commit 完全一致。

只有验证成功的接受记录才参与 lineage winner 和 `accepted_best`。Episode
`attempt.json`、旧 promotion、handoff、ABBA 或 Framework Baseline 状态均不是
接受来源。

## `best_passing_full_eval`

它表示所有合格标准 full Evaluate 中的最低 GeoMean latency，不要求 commit
已经调用 `submit-kernel`。Shape subset、自定义输入、ABBA、Profile、Check、
Disassemble、Dev、失败、运行中或不完整 Record 均不参与。

因此 `best_passing_full_eval` 可以优于 `accepted_best`：前者是观测事实，后者
只在 Agent 明确提交且最终源码审查通过后成立。

## 多 trajectory

`light_submission_status.py` 展示：

- 所有接受与拒绝的 submission；
- 每条 lineage 的最佳 verified accepted submission；
- 只有存在多条 lineage 时才展示 `global_best`，避免单线重复。

按 commit 或 Gateway Record 恢复源码可使用 `export_light_source.py`。
