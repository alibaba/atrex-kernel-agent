# AKA Light 可观测性脚本使用说明

本文按照《AKA Light 可观测性计划》的分类，说明每个脚本的用法、输出字段和预期结果。
所有脚本均为只读操作，不会提交、取消或修改 Agate 任务。

## 1. 通用用法

要求 Python 3.10 或更新版本。先进入 AKA 仓库并指定顶层 Campaign 工作区：

```bash
cd /path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign-workspace
```

每个状态脚本提供三种输出格式：

```bash
# 默认：便于人阅读的表格
python3 observability/light_runtime_status.py "$CAMPAIGN"

# 完整 JSON：适合 jq、存档和后续统计
python3 observability/light_runtime_status.py "$CAMPAIGN" --json

# 单行 JSON：适合周期采样和日志收集
python3 observability/light_runtime_status.py "$CAMPAIGN" --compact
```

脚本会自动发现顶层 Campaign 和 pooled search 的 `t01`、`t02`、`t03` 等轨迹工作区。
输出中的 `scope=campaign` 表示顶层工作区，`scope=t01` 等表示具体轨迹。

| 计划分类 | 脚本 |
|---|---|
| 运行状态 | `light_runtime_status.py` |
| 性能与正确性 | `light_performance.py` |
| 接受与晋升 | `light_acceptance.py` |
| Agate 调用与故障 | `light_agate_status.py` |
| Token 与时间 | `light_token_time.py` |
| Runtime Tool 使用 | `light_tool_usage.py` |
| 源码与可复现性 | `export_light_evidence.py`、`export_light_source.py` |
| GPU Wiki | `light_wiki_usage.py` |

## 2. 运行状态

运行：

```bash
python3 observability/light_runtime_status.py "$CAMPAIGN"
python3 observability/light_runtime_status.py "$CAMPAIGN" --json
```

默认 Campaign 状态和当前 Session 观测都超过 600 秒没有更新时，会被判断为停滞。可以修改阈值：

```bash
python3 observability/light_runtime_status.py "$CAMPAIGN" --stale-after 900
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| Campaign 当前阶段和整体状态 | `scopes[].phase`、`scopes[].health`、`scopes[].writer_active` |
| 每条轨迹运行到哪个 Episode | `scopes[].trajectory`、`scopes[].current_episode`、`scopes[].current_version` |
| 运行、完成、失败和停滞的 Episode | `scopes[].episodes[].category`、`scopes[].episode_counts` |
| 当前活跃 Agent Session | `scopes[].active_sessions[]`、`scopes[].active_session_count` |
| 是否健康推进及停滞时长 | `scopes[].health`、`scopes[].stagnation_seconds` |

`health` 可能为：

- `healthy`：Writer 正在运行，且最近更新时间没有超过停滞阈值；
- `stalled`：Writer 仍标为活跃，但状态更新时间已经超过阈值；
- `inactive`：没有活跃 Writer，Campaign 也没有进入终态；
- `completed`、`failed`、`interrupted`、`stopped`：Campaign 终态。

第一张表展示 Campaign/轨迹级状态。`Campaign Phase` 是 Workflow 阶段，`Episode` 是当前轮次，
`Active Sessions` 是当前状态仍为 `running` 的 Agent invocation 数量。正式 Episode 从 Episode
目录读取；Bootstrap／Framework Baseline 从 workspace 同级的 `.atrex-session-traces` 读取：

```text
Scope  Campaign Phase  Episode  Done  Health    Stale(s)  Active Sessions
t01    exploring       3        2     healthy   18.4      1
t02    exploring       2        1     stalled   742.1     1
t03    completed       5        5     completed  30.2     0
```

第二张表逐个展示活跃 Session，并明确它属于哪个 Stage／Episode、当前 Agent Phase 及该阶段已持续多久：

```text
Active Agent Sessions
Scope  Stage               Episode  Session       Backend  Agent Phase    Tool      Phase(s)  Health
t01    episode             3        run-a1b2c3    claude   thinking       —         18.4      active
t02    framework_baseline  0        session-xyz   claude   waiting_agate  run       742.1     stalled
```

Agent Phase 的含义：

| Agent Phase | 含义 |
|---|---|
| `thinking` | 当前没有尚未返回的工具调用，Agent 正在生成、分析或准备下一步 |
| `executing_tool` | Bash、读写文件等普通工具尚未返回 |
| `executing_runtime_tool` | Record、Journal 等不直接等待 Agate/Wiki 的 Runtime Tool 尚未返回 |
| `waiting_agate` | Evaluate、ABBA、Profile、Check、Disassemble 或 Dev 请求尚未返回 |
| `waiting_wiki` | GPU Wiki 请求尚未返回 |

这里的 `waiting_agate` 表示 Agate-backed Runtime 请求仍在前台等待，不能据此断言远端 Job
一定已经越过排队并开始执行。具体远端状态需结合 `light_agate_status.py` 查看。

阶段由当前 `run-*/conversation.jsonl` 中“已发出但尚未出现 tool result”的最后一个工具调用判断；
Session 身份来自同一 invocation 的 `token-usage.json`。正式 Episode 使用其 Episode 目录；启动阶段
使用 Supervisor 写入的 `memory/live.json` 区分 `bootstrap` 和 `framework_baseline`，不按时间猜测关联。

只看活跃 Session：

```bash
python3 observability/light_runtime_status.py "$CAMPAIGN" --json |
  jq '.scopes[].active_sessions[]'
```

只看等待 Agate 或 Wiki 的 Session：

```bash
python3 observability/light_runtime_status.py "$CAMPAIGN" --json |
  jq '.scopes[].active_sessions[] | select(.phase == "waiting_agate" or .phase == "waiting_wiki")'
```

`scopes[].sessions[]` 仍保留已经落盘的历史 Session 事实；当前运行态应以
`scopes[].active_sessions[]` 为准。

## 3. 性能与正确性

运行：

```bash
python3 observability/light_performance.py "$CAMPAIGN"
python3 observability/light_performance.py "$CAMPAIGN" --json
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| 当前已接受实现的最好 latency | `accepted_best` |
| 所有成功 Full Evaluate 的最好 latency | `best_passing_full_eval` |
| Bootstrap 和 Framework Baseline latency | `baselines[].bootstrap`、`baselines[].framework_baseline.result` |
| 每个 Episode／Attempt latency | `episodes[]` |
| 每个 Shape latency 和正确性 | `shape_results[]` |
| 每个 Shape 当前最快通过的 Kernel | `best_by_shape` |

AKA Light 中一个优化 Attempt 对应一个 Episode，所以脚本以 Episode 作为 Attempt 统计单位。

`shape_results[].correctness` 可能为：

- `pass`：该 Shape 明确通过，或其他 Shape 失败但该 Shape 有有效性能结果；
- `fail`：失败原因明确指向该 Shape；
- `unknown`：只有无法归属到具体 Shape 的整体失败，不能安全推断该 Shape 是否通过。

预期输出：

```text
Metric                  GeoMean(us)  Kernel/Commit  Record           Evidence
accepted_best           123.450000   012345...      gateway-abcd...  verified
best_passing_full_eval  119.820000   kernel-abcd... gateway-1234...  full_eval_record

Scope  Episode  Version  Promoted  State  Latency(us)
t01    1        2        True      pass   141.200000
t01    2        3        False     rejected 138.700000
```

`accepted_best.gateway_record_id` 是该候选终态晋升时引用的 ABBA/Evaluate Record。
脚本会验证 Record 中的 candidate Kernel 内容身份等于 accepted Git commit，并通过
`accepted_best.verification_evidence` 返回结果：

- `verified`：Episode、通过的 verification、Record 和 Kernel 源码四者一致；
- `bootstrap_has_no_episode_verification`：最佳实现仍是 Bootstrap，没有 Episode 晋升 Record；
- 其他状态：Attempt、Record 或源码不可读／不一致；如果 Attempt 中确实声明了 Record ID，
  脚本仍会将其保留，但不能把它当成已经完成内容校验的证据。

### Candidate State 的含义

这里的 `State` 是 Workflow 对候选结果的综合状态，不是 Agent Session 状态，也不是
Agate Job 状态。它来自 `episodes[].result.state`：

| State | 含义 |
|---|---|
| `running` | 候选仍在评测或验收，尚未形成终态 |
| `pass` | 正确性通过、晋升门禁通过，并且性能测量完整 |
| `correctness_failure` | 候选没有通过正确性检查 |
| `rejected` | 已经取得测量结果，但晋升门禁拒绝，例如 ABBA 未确认提升、提升不足或策略门禁失败 |
| `infrastructure_error` | 验收受到基础设施故障影响，不能把结果解释为 Kernel 失败 |
| `interrupted` | Episode 或验收流程被中断 |
| `unmeasured` | 没有形成完整、可用的测量结果 |

`State` 与 `Promoted` 应一起阅读：

```ini
Promoted=False  State=rejected  Latency(us)=138.7
```

这表示 Workflow 确实测到了 `138.7 µs`，但候选没有通过完整晋升门禁，因此没有替换当前
最佳实现。单次普通 Evaluate 比 incumbent 更快，并不等于 ABBA 或终态验收已经确认提升。
具体拒绝原因需要运行 `light_acceptance.py`，查看同一 `scope + episode + version` 的
`gate` 和 `decision_reason`。

几个容易混淆的状态来源：

| 状态 | 查询位置 | 回答的问题 |
|---|---|---|
| Candidate State | `light_performance.py` 的 `episodes[].result.state` | 这个候选在 Workflow 中形成了什么结果？ |
| Promotion | `light_acceptance.py` 的 `promoted`、`decision_reason` | 这个候选为什么晋升或被拒绝？ |
| Agent Session Phase | `light_runtime_status.py` 的 `active_sessions[].phase` | Agent 当前在思考、执行工具还是等待远端？ |
| Agate Job Status | `light_agate_status.py` 的 `physical_jobs[].status` | 远端 Agate Job 当前 queued、running、succeeded 还是 failed？ |

提取所有轨迹当前已接受的 Episode 结果：

```bash
python3 observability/light_performance.py "$CAMPAIGN" --json |
  jq '.episodes[] | select(.promoted == true)'
```

提取每个 Shape 的最快通过实现：

```bash
python3 observability/light_performance.py "$CAMPAIGN" --json |
  jq '.best_by_shape'
```

## 4. 接受与晋升

运行：

```bash
python3 observability/light_acceptance.py "$CAMPAIGN"
python3 observability/light_acceptance.py "$CAMPAIGN" --json
```

计划项均位于 `candidates[]`：

| 计划项 | 字段 |
|---|---|
| 是否通过正确性门禁 | `gate`、`correctness`、`accepted` |
| 相对当前最佳实现的性能变化 | `incumbent_latency_us`、`candidate_latency_us`、`improvement_pct` |
| 是否晋升及晋升原因 | `promoted`、`decision_reason` |
| 被拒绝的具体原因 | `decision_reason`、`status` |
| 晋升对应轨迹、Episode 和证据 | `scope`、`trajectory`、`episode`、`evidence` |

`evidence` 中可能包含 `gateway_record_id`，以及终态验收保存的 Evaluate 类型、时间、
latency 和源码 SHA256。

预期输出：

```text
Scope  Episode  Version  Gate  Promoted  Latency(us)  Improve(%)  Reason
t01    1        2        PASS  True      141.200000   8.42        promotion gate passed
t01    2        3        FAIL  False     150.700000  -6.73        correctness mismatch
```

只看晋升：

```bash
python3 observability/light_acceptance.py "$CAMPAIGN" --json | jq '.promotions[]'
```

只看拒绝：

```bash
python3 observability/light_acceptance.py "$CAMPAIGN" --json |
  jq '.candidates[] | select(.promoted == false and .episode != null)'
```

## 5. Agate 调用与故障

运行：

```bash
python3 observability/light_agate_status.py "$CAMPAIGN"
python3 observability/light_agate_status.py "$CAMPAIGN" --json
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| 各类调用次数 | `summary.operation_counts` |
| Bootstrap／Framework Baseline 物理任务 | `summary.trusted_operation_counts`、`physical_jobs[].source == "trusted"` |
| 排队、运行和完成的 Agate job | `physical_jobs[].status`、`summary.physical_job_status_counts` |
| 当前未完成 job、执行阶段和等待时长 | `pending_jobs[].job_id`、`execution_phase`、`pending_seconds` |
| 排队、执行和总耗时 | `physical_jobs[].timing.queue_seconds`、`execution_seconds`、`total_seconds` |
| 错误分类 | `calls[].error`、`summary.error_counts` |
| 自动重试及最终结果 | `calls[].retry_count`、`calls[].outcome`、`physical_jobs[].retry_count` |
| Evaluate／ABBA 分批和复用 | `calls[].batch` |
| 各类 timeout | `summary.timeout_counts` |
| 重复、缓存命中和同请求运行 | `summary.duplicate_cache_hits`、`same_request_running` |
| Agent 是否暂停 | `agent_paused_during_requests` |
| 未完成 Record、锁、槽位和队列 | `running_task_markers`、`runtime_concurrency` |

`calls[].outcome` 可能为：

- `completed`：完整且可缓存；
- `completed_not_cacheable`：调用返回，但不能作为可复用终态证据；
- `failed`：发现终态错误；
- `running_or_interrupted`：没有形成完整 Record。

错误类别为 `infrastructure`、`candidate`、`policy`、`protocol` 或 `unknown`。

`summary.operation_counts` 只统计有完整 Measurement Record 身份的逻辑调用。启动阶段的 trusted
任务没有 `gateway_record_id`，因此单独计入 `trusted_operation_counts`，并与普通 Record job 一起
出现在 `physical_jobs`。文本输出会额外显示 `Pending Agate jobs` 表，直接列出尚未终态的 Job ID、
Agate execution phase 和当前等待秒数。

分批字段示例：

```json
{
  "shape_batch_count": 8,
  "completed_shape_batch_count": 8,
  "physical_batch_jobs": 8,
  "reused_shape_batches": 5,
  "submitted_shape_batches": 3,
  "completed_result_reused": false
}
```

这表示本轮 8 个 Shape batch 中，5 个由已完成 checkpoint 恢复，只有 3 个重新提交。

调用摘要示例：

```json
{
  "operation_counts": {"evaluate": 12, "abba": 4, "profile": 2},
  "physical_job_status_counts": {"queued": 1, "running": 2, "succeeded": 14},
  "error_counts": {"infrastructure": 2, "candidate": 1},
  "retry_count": 3,
  "duplicate_cache_hits": 2,
  "same_request_running": 1
}
```

筛选失败调用：

```bash
python3 observability/light_agate_status.py "$CAMPAIGN" --json |
  jq '.calls[] | select(.error != null)'
```

筛选仍在排队或运行的物理 Job：

```bash
python3 observability/light_agate_status.py "$CAMPAIGN" --json |
  jq '.physical_jobs[] | select(.status == "queued" or .status == "running")'
```

当前主线没有持久化实际退避睡眠时长，因此只能得到重试次数，
`availability.retry_backoff_seconds` 会说明这一限制。Live lock、并发槽位和等待队列也只存在于
Runtime 进程内；脚本能读取未完成 Record marker，但不会把 marker 错报成仍被占用的锁。

## 6. Token 与时间

运行：

```bash
python3 observability/light_token_time.py "$CAMPAIGN"
python3 observability/light_token_time.py "$CAMPAIGN" --json
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| 每个 Session Token | `sessions[].usage` |
| Bootstrap／Framework Baseline Token | `sessions[].stage`、`startup_totals` |
| 每条轨迹和 Episode Token | `episodes[].usage` |
| 每个 Attempt 平均 Token | `average_tokens_per_attempt` |
| Agent 工作时间和墙钟时间 | `sessions[].wall_seconds`、`episodes[].wall_seconds`、`episodes[].active_seconds` |
| 等待 Agate 时间和占比 | `episodes[].agate_wait_seconds` |
| 单位 Token latency 改善 | `episodes[].latency_improvement_us_per_million_tokens` |
| 输出截断、限流、Session 重启、invalid handoff | `incidents` |

Token 使用结构：

```json
{
  "input_tokens": 120000,
  "output_tokens": 40000,
  "reasoning_tokens": null,
  "cache_read_tokens": 60000,
  "cache_write_tokens": 0,
  "total_tokens": 160000,
  "measurement": "exact"
}
```

当前主线没有单独的 `reasoning_tokens` 计数；不同 Agent 后端可能将推理计入
`output_tokens`。脚本不会自行拆分。当前主线也没有持久化精确 Agent active time 和 Agate
wait time，所以这两项为 `null`；已结束 Session 的墙钟时间仍可读取。
Bootstrap 和 Framework Baseline Session 来自 workspace 同级的 `.atrex-session-traces`，以
`episode=0` 和相应的 `stage` 返回；它们计入 `totals` 和 `startup_totals`，但不计入
“每个 Episode／Attempt 平均 Token”，避免把启动成本伪装成优化 Attempt。

预期表格：

```text
Scope  Episode  Tokens  Wall(s)  Accepted  Improve us/Mtok  Sessions
t01    1        120000  1820.4   True      8.9286           1
t01    2        160000  2314.8   False     0.0000           2
```

## 7. Runtime Tool 使用

运行：

```bash
python3 observability/light_tool_usage.py "$CAMPAIGN"
python3 observability/light_tool_usage.py "$CAMPAIGN" --json
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| 每种 Runtime Tool 调用次数 | `summary[].tool`、`summary[].calls` |
| 成功率、错误率和耗时 | `summary[].successes`、`failures`、`success_rate`、`average_duration_seconds` |
| 每次调用的结果 | `calls[]` |
| 调用所属 Session、Episode 和轨迹 | `session_attribution[]` |

预期输出：

```text
Tool         Calls  Success  Failure  Rate    Avg(s)  Timed
evaluate     12     10       2        0.8333  84.2    12
profile      3      2        1        0.6667  192.5   3
wiki-query   4      4        0        1.0000  1.8     4
```

`calls[]` 来自 Runtime 审计记录，结果和退出码更权威；`session_attribution[]` 来自 Session
事件，能够给出 Session、Episode 和轨迹。当前主线没有持久化两者之间的稳定关联 ID，因此脚本
不会把两份记录按相近时间强行拼接成同一次调用。

## 8. 源码与可复现性

先从性能脚本取得 `gateway_record_id`：

```bash
python3 observability/light_performance.py "$CAMPAIGN" --json > /tmp/performance.json
jq '.best_passing_full_eval' /tmp/performance.json
```

导出一次测量的完整证据：

```bash
python3 observability/export_light_evidence.py "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /tmp/light-evidence
```

输出目录必须不存在。成功后包含：

```text
/tmp/light-evidence/
├── kernel/                 # 被评测的完整 Kernel 源码包
├── agate-jobs.json         # 原始 Agate Job ID、响应和进程收据
├── agent-response.json     # Light 实际暴露给 Agent 的返回信息
└── manifest.json           # Kernel、Record、文件和证据索引
```

`agate-jobs.json` 中每一项包含：

- `job_id`：原始 Agate Job ID；
- `agate_response`：Agate 原始结构化返回；
- `raw_process`：执行 Agate CLI 时捕获的 stdout、stderr 和返回码；
- `submission`、`retries`、`checkpoint_phase`：提交与恢复状态。

如果只有 accepted best 的 Git commit，使用：

```bash
python3 observability/export_light_source.py "$CAMPAIGN" \
  --git-commit 0123456789abcdef0123456789abcdef01234567 \
  --output /tmp/accepted-kernel
```

也可以按 Kernel ID 或 Gateway Record ID 单独导出源码：

```bash
python3 observability/export_light_source.py "$CAMPAIGN" \
  --kernel-id kernel-0123456789abcdef0123456789abcdef \
  --output /tmp/measured-kernel
```

## 9. GPU Wiki

运行：

```bash
python3 observability/light_wiki_usage.py "$CAMPAIGN"
python3 observability/light_wiki_usage.py "$CAMPAIGN" --json
```

计划项与 JSON 字段对应关系：

| 计划项 | JSON 字段 |
|---|---|
| 查询次数和查询内容 | `summary.query_count`、`queries[].query` |
| 返回或采用的知识条目 | `queries[].knowledge_ids`、`queries[].adopted`、`journal_usage[]` |
| 查询耗时 | `queries[].queue_latency_ms`、`bridge_latency_ms`、`retrieval_latency_ms`、`total_latency_ms` |
| 失败和无结果 | `queries[].status`、`provider_error_code`、`provider_error_message`、`summary.failure_count`、`no_result_count` |

预期输出：

```text
Scope  Tool        Status     Cache  Total(ms)  Knowledge  Adopted  Query
t01    wiki-query  completed  False  1820       3          True     PPU shared-memory bank conflict
t02    wiki-query  completed  True   24         0          False    async copy alignment
```

`adopted=true` 表示后续 Experiment Journal 明确引用了该 query 或知识条目；它不等同于已经
证明该知识改善了性能。是否改善需要再结合 `light_performance.py` 和对应 Episode 的晋升证据判断。

## 10. 一次性采集全部状态

```bash
mkdir -p /tmp/light-observability

python3 observability/light_runtime_status.py "$CAMPAIGN" --json \
  > /tmp/light-observability/runtime.json
python3 observability/light_performance.py "$CAMPAIGN" --json \
  > /tmp/light-observability/performance.json
python3 observability/light_acceptance.py "$CAMPAIGN" --json \
  > /tmp/light-observability/acceptance.json
python3 observability/light_agate_status.py "$CAMPAIGN" --json \
  > /tmp/light-observability/agate.json
python3 observability/light_token_time.py "$CAMPAIGN" --json \
  > /tmp/light-observability/token-time.json
python3 observability/light_tool_usage.py "$CAMPAIGN" --json \
  > /tmp/light-observability/tools.json
python3 observability/light_wiki_usage.py "$CAMPAIGN" --json \
  > /tmp/light-observability/wiki.json
```

这些文件共同构成一个时间点的只读状态快照。需要连续监控时，可以由外部定时器周期调用
`--compact` 模式；脚本自身不会常驻，也不会改变 Campaign 状态。
