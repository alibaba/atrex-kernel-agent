# Light 最佳延迟的数据来源

本文说明 Light 运行过程中两项最佳延迟从哪里取回，以及哪些记录可以参与统计。本文不描述脚本实现。

## 输入

查询入口是 Light Campaign 的顶层 workspace。

Light 根据该 workspace 的绝对路径确定对应的 Supervisor Campaign 根目录。运行状态、轨迹目录和 Agate Measurement Records 都从这个根目录继续定位。操作者不需要指定轨迹数量或运行模式。

## 当前已接受的最佳延迟

输出字段为 `accepted_best`。

它表示已经通过 Light 接受流程的 Kernel 中，GeoMean latency 最低者。数据按以下顺序取回：

1. 首先检查 Supervisor Campaign 根目录下的 `search/state.json`。
2. `seed` 表示已经发布的当前最佳 Kernel，其中的 `latency_us_geomean` 是其权威 GeoMean latency。
3. 当前正在进行的一轮中，`active.jobs` 里状态为 `completed` 且带有 `result.retained_kernel` 的任务也已经通过对应 Episode 的接受门禁，因此一并参与比较。
4. 在上述候选中选择 `latency_us_geomean` 最低者。
5. 如果 Campaign 没有 `search/state.json`，则通过主线的 Workflow Export 读取已晋升的候选，只保留 `promoted=true`、结果状态为 `pass` 且测量完整的版本，再选择最低 GeoMean latency。

该结果同时保留可用的 Kernel Git commit、Episode、trajectory、version 和源码摘要。对于
Episode 晋升，工具还会读取对应 `attempt.json` 的 passing verification，取得
`gateway_record_id`，再验证该 Record 的 candidate Kernel 内容身份等于 accepted Git
revision。验证状态写入 `verification_evidence`。

普通 Eval 通过并不等于已经接受。只有形成 `retained_kernel`，或者在 Workflow Export 中被标记为已晋升且通过的候选，才能进入 `accepted_best`。

## Agate Eval 观测到的最佳延迟

输出字段为 `best_passing_full_eval`。

它表示运行至今所有合格的标准 Agate full Eval 中，GeoMean latency 最低者，不要求该 Kernel 已经被 Light 接受。

### Measurement Store 位置

工具从顶层 workspace 开始，自动发现该 Campaign 的全部持久化测量范围。每个范围的权威记录位于对应 Supervisor 根目录的：

- `measurements/records/gateway-*/record.json`：Agate 调用请求、返回结果和 Record 身份；
- `measurements/kernels/<kernel-id>/kernel.py`：与该测量绑定的候选源码快照；
- `measurements/identity.key`：证明该 Measurement Store 已经初始化，避免观测命令意外创建新 Store。

如果 Campaign 下存在 `search/trajectories/t*`，这些 workspace 对应的 Measurement Store 也会被自动发现并汇总。操作者仍然只需要提供顶层 Campaign workspace。

### Record 完整性

每个 `record.json` 都通过主线 `MeasurementStore.read` 读取，而不是直接信任 JSON 内容。读取过程会验证：

- Gateway Record ID 与目录一致；
- 请求摘要与保存的请求一致；
- 结果摘要与保存的响应一致；
- Kernel ID、源码快照和测量输入相互一致。

无法通过完整性检查的 Record 不参与统计，也不会中断其他合法 Record 的读取。

### Eval 筛选条件

只有同时满足以下条件的 Record 才能参与 `best_passing_full_eval`：

- 操作为 `evaluate`；
- Record 已完成并可缓存；
- 使用标准 `test_kernel.py` 入口；
- 评测模式为 `full`；
- 使用完整 contract 输入和 Shape 集；
- 没有指定单独的 Shape；
- 没有使用自定义输入、Shape 文件或 baseline 路径；
- Agate 调用正常退出；
- 返回没有被截断；
- 所有 Shape 均通过正确性检查；
- 每个 Shape 的 latency 都是有限正数。

以下记录不参与统计：

- correctness-only；
- 单 Shape 或 Shape subset；
- 自定义输入评测；
- ABBA；
- Profile、Check、Disassemble 和 Dev；
- 正确性失败；
- 基础设施失败、超时或仍在运行；
- 返回被截断或证据不完整。

对每个合格 Record，优先使用 Agate 返回的 `latency_us_geomean`。如果该字段缺失，但完整的逐 Shape latency 可用，则根据全部 Shape latency 重新计算几何平均。最后在所有合格 Record 中选择最低值。

结果同时返回对应的内容型 Kernel ID、Kernel artifact digest、Gateway Record ID、Shape 数量和 Measurement workspace，便于继续读取完整证据。

这里必须区分两种身份：

- 接受状态里的 `commit` 或 Workflow Export 的 `revision` 是 Git commit，输出统一命名为 `accepted_best.git_commit`；
- 当前主线 Measurement Store 的 `kernel_id` 不是 Git commit。它由候选源码的 SHA-256 通过 UUIDv5 派生，格式为 `kernel-<32 个十六进制字符>`，输出为 `best_passing_full_eval.kernel_id`；
- Measurement Record 同时保存原始内容摘要 `kernel_artifact_digest`，格式为 `sha256:<64 个十六进制字符>`。

因此不能用 `kernel_id` 执行 Git 操作，也不能把它重命名为 Git commit。需要取回源码时，应使用 Measurement Store 的 Kernel 读取接口和该 `kernel_id`。

`accepted_best.git_commit` 也不是 AKA/Supervisor 源码仓库自身的版本号。它属于 Supervisor 管理的 Campaign Kernel Git 仓库：

- Bootstrap 对应值通常是 Campaign 初始 workspace 的 HEAD；
- Episode 候选通过验收后，Supervisor 将候选 squash 到 incumbent/controller workspace，并以 `atrex-long-horizon` 身份创建 promotion commit；
- 搜索状态保留这个 promotion commit 作为被接受 Kernel 的源码 revision；
- Epoch 发布时，Supervisor 可能在顶层 Campaign workspace 另外创建 publication commit，但被接受 Kernel 记录仍指向原 promotion commit；
- 当前轮次已完成但尚未发布的候选，其 promotion commit 可能暂时只存在于对应的 controller workspace。

因此该字段表达的是“包含这份已接受 Kernel 源码的 Campaign Git revision”，而不是“运行这次任务的 Supervisor 软件版本”。

## 为什么两个结果可能不同

`best_passing_full_eval` 可能优于 `accepted_best`，常见原因包括：

- 候选刚完成普通 Eval，尚未进入终态验收；
- 候选尚未完成 ABBA；
- 候选在 ABBA 或其他晋升门禁中失败；
- 候选所在任务已完成 Eval，但其 Episode 尚未形成可接受的最终报告；
- 测量噪声导致普通 Eval 很好，但终态复测没有确认该提升。

因此，两项指标必须分别保留，不能用普通 Eval 最优值替代正式 incumbent。

## 运行中的可见时机

- Agate 任务仍为 queued 或 running 时，还没有终态 Measurement Record，不会进入 `best_passing_full_eval`。
- Agate 任务完成且 Record 原子发布后，下一次查询即可看到该结果。
- 候选只有在接受状态持久化后才会进入 `accepted_best`。
- 多个测量可能在不同时间完成，因此一次查询是当时的状态快照；持续查询会逐步纳入新完成的结果。

## 不使用的数据

这两项指标不通过以下内容推断：

- Agent 对话或推理文本；
- Supervisor 普通日志；
- Agent 自己写的 scratch 笔记；
- 逐 Shape Pareto envelope；
- 重新向 Agate 提交或查询一次相同评测。

权威来源只有 Light 的持久化接受状态和通过完整性校验的 Measurement Records。

## 输出示例

下面表示 Light 已经接受了一个 GeoMean latency 为 `123.45 us` 的 Kernel；运行过程中还观测到另一个通过完整 full Eval、GeoMean latency 为 `119.87 us` 的 Kernel。后者更快，但尚不能据此认定已经完成晋升。

```json
{
  "accepted_best": {
    "episode": 4,
    "gateway_record_id": "gateway-fedcba9876543210fedcba9876543210",
    "git_commit": "0123456789abcdef0123456789abcdef01234567",
    "latency_us_geomean": 123.45,
    "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "trajectory": 2,
    "verification_evidence": "verified",
    "verification_kernel_id": "kernel-00112233445566778899aabbccddeeff",
    "verification_operation": "same_allocation_abba"
  },
  "best_passing_full_eval": {
    "gateway_record_id": "gateway-0123456789abcdef0123456789abcdef",
    "kernel_artifact_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "kernel_id": "kernel-fedcba9876543210fedcba9876543210",
    "latency_us_geomean": 119.87,
    "measurement_workspace": "/workspace/.atrex-supervisor-runtime/0123456789abcdef/search/trajectories/t0001",
    "shape_count": 90
  },
  "campaign_workspace": "/workspace/campaign",
  "observed_at": "2026-11-06T08:30:00+00:00",
  "schema_version": 1
}
```

字段含义：

- `schema_version`：观测结果格式版本；
- `observed_at`：本次快照生成时间，使用 UTC；
- `campaign_workspace`：查询时提供的顶层 Campaign workspace；
- `accepted_best.latency_us_geomean`：已接受候选中的最低 GeoMean latency，单位为微秒；
- `accepted_best.git_commit`：主线搜索状态的 `commit` 或 Workflow Export 的 `revision`，是 Git commit；
- `accepted_best.gateway_record_id`：该 accepted Candidate 终态晋升时使用的验证 Record；
- `accepted_best.verification_evidence`：Attempt、Record 和 accepted source 的关联校验状态；
- `accepted_best.trajectory` 和 `episode`：产生该 Kernel 的运行位置；
- `accepted_best.source_sha256`：可用时返回的 Kernel 源码快照摘要；
- `best_passing_full_eval.latency_us_geomean`：合格 full Eval 中的最低 GeoMean latency，单位为微秒；
- `best_passing_full_eval.kernel_id`：主线 Measurement Store 根据源码内容派生的 Kernel ID，不是 Git commit；
- `best_passing_full_eval.kernel_artifact_digest`：被测 Kernel 源码 artifact 的 SHA-256 内容摘要；
- `best_passing_full_eval.gateway_record_id`：支持该数值的完整 Measurement Record；
- `best_passing_full_eval.shape_count`：该 full Eval 实际覆盖的 Shape 数量；
- `best_passing_full_eval.measurement_workspace`：保存该 Measurement Record 的运行 workspace。

如果相应结果尚未产生，字段为 `null`。例如 Campaign 已启动，但还没有任何合格 full Eval 时：

```json
{
  "accepted_best": {
    "episode": 0,
    "gateway_record_id": null,
    "git_commit": "0123456789abcdef0123456789abcdef01234567",
    "latency_us_geomean": 250.0,
    "source_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "trajectory": 0,
    "verification_evidence": "bootstrap_has_no_episode_verification",
    "verification_kernel_id": null,
    "verification_operation": null
  },
  "best_passing_full_eval": null,
  "campaign_workspace": "/workspace/campaign",
  "observed_at": "2026-11-06T08:00:00+00:00",
  "schema_version": 1
}
```

## 根据结果取回源码

仓库内的 `export_light_source.py` 可以根据上述输出的身份取回完整候选源码包：

- `accepted_best.git_commit`：在顶层 Campaign 和自动发现的 controller repositories 中定位对应 Git revision，再读取该 revision 的候选源码；
- `best_passing_full_eval.gateway_record_id`：读取通过完整性校验的 Measurement Record，并取回其 `candidate` 侧绑定的 Kernel artifact；
- `best_passing_full_eval.kernel_id`：直接从自动发现的 Measurement Stores 中读取内容绑定的 Kernel artifact；
- 完整 latency status JSON：选择 `accepted_best` 或 `best_passing_full_eval` 后自动采用上述身份。对于 Eval 结果优先使用 `gateway_record_id`，因为它同时绑定请求、结果和候选源码。

导出目标必须是一个尚不存在的路径。所有格式都导出为目录：单文件候选包含 `kernel.py`，源码树和 CUDA manifest 保留其原始文件布局及文件系统选择元数据。导出过程不会提交或取消 Agate 任务，也不会修改 Campaign Git 或 Measurement Store。
