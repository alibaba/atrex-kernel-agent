# Light 最佳延迟的数据来源

`light_latency_status.py` 是只读工具。输入顶层 Campaign workspace 后，它自动发现串行范围或 pooled search 的各 trajectory，不提交或查询新的 Agate Job。

## 统一的 Kernel 身份

当前 Light 只有一个公开 Kernel 身份：Agent 工作区 clean HEAD 的完整 Git commit ID。所有面向 Agent、Research、Measurement Record 和可观测性结果的字段统一命名为 `kernel_commit_id`。

Supervisor 会把同一源码提交到独立的私有归档 Git，并在 Measurement Store 内保存 Agent commit 与私有 commit 的映射。私有 commit 只用于恢复和审计，不会出现在公开状态中，也不能作为 Agent 的 `git checkout` 参数。

Measurement Store 中：

- `measurements/kernels/<kernel_commit_id>/kernel.py` 保存该 commit 的完整源码快照；
- `supervisor-git.json` 可选保存私有归档映射；
- `kernel_artifact_digest` 是源码快照的 SHA-256，不是另一种 Kernel ID；
- `measurements/records/gateway-*/record.json` 把请求、结果和一个或多个 Agent commit 绑定在一起。

每个 Record 都通过 `MeasurementStore.read()` 读取，校验请求摘要、结果摘要、源码快照、artifact digest 和评测输入。损坏或不完整的历史 Record 被跳过，不会污染最佳值。

## `accepted_best`

`accepted_best` 表示已通过 Light 接受流程的候选中，GeoMean latency 最低者。权威来源是每个 scope 的 `.atrex_long_horizon/episodes/e*/attempt.json`。

Framework Baseline 是 Episode 0，同样写入普通 Attempt。候选必须满足：

- `accepted=true`；
- 有合法的 `candidate_kernel_commit_id`；
- 有有限正数的 `latency_us_geomean`，或终态 verification 中的 candidate latency；
- 若 Attempt 引用了 Gateway Record，该 Record 的 candidate `kernel_id` 必须等于 Agent commit。

Episode 0 使用普通 full Evaluate 证据；后续晋升通常使用终态 ABBA Record。脚本不会尝试在 Supervisor/controller Git 中解析 Agent commit。

## `best_passing_full_eval`

`best_passing_full_eval` 表示所有合格标准 full Evaluate 中的最低 GeoMean latency，不要求候选已晋升。只有同时满足以下条件的 Record 才参与：

- 操作为 `evaluate`，已完成、可缓存且未截断；
- 使用标准 `test_kernel.py` 和 full evaluation policy；
- 没有自定义输入、单 Shape CLI 或 baseline 覆盖；
- 所有 Shape 正确性通过；
- 每个 Shape latency 都是有限正数。

普通 Evaluate 可能按最多 10 个 Shape 的 outer batch 运行；只有合并后的完整 Record 才算 full Evaluate。脚本优先使用回执中的 `latency_us_geomean`，缺失时才从完整逐 Shape latency 重算几何平均。

ABBA、Profile、Check、Disassemble、Dev、correctness-only、Shape subset、失败、运行中或不完整的 Record 均不参与。

## 为什么两个结果可能不同

`best_passing_full_eval` 可能优于 `accepted_best`，例如候选尚未完成 report、Production policy、终态 ABBA 或最小提升门禁。普通 Eval 最优不能替代正式 incumbent。

## 输出示例

```json
{
  "accepted_best": {
    "episode": 4,
    "gateway_record_id": "gateway-fedcba9876543210fedcba9876543210",
    "kernel_commit_id": "0123456789abcdef0123456789abcdef01234567",
    "latency_us_geomean": 123.45,
    "trajectory": 2,
    "verification_evidence": "verified",
    "verification_kernel_artifact_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "verification_kernel_commit_id": "0123456789abcdef0123456789abcdef01234567",
    "verification_operation": "same_allocation_abba"
  },
  "best_passing_full_eval": {
    "gateway_record_id": "gateway-0123456789abcdef0123456789abcdef",
    "kernel_commit_id": "89abcdef0123456789abcdef0123456789abcdef",
    "kernel_artifact_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "latency_us_geomean": 119.87,
    "shape_count": 90
  }
}
```

## 取回源码

按 Agent commit 或 Gateway Record 导出：

```bash
python3 observability/export_light_source.py "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-source

python3 observability/export_light_source.py "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-source
```

导出器优先从 Measurement Store 读取 commit 绑定的源码，因此即使 Agent 分支后来移动，已测源码仍可恢复；对于尚未测量但仍可达的本地 commit，才回退到 Agent Git。目标目录必须不存在。
