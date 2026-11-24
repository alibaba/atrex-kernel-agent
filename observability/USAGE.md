# AKA Light 可观测性完整手册

语言：**简体中文** | [English](USAGE.en.md)

本文档说明 AKA Light 的八个正式可观测性入口：如何选择命令、所有选项的
作用、预期 JSON 返回结构、字段含义、事实来源与常见误读。脚本自身的
`--help` 是 CLI 参数的最终权威；本文档解释参数和结果的语义。

## 1. 使用范围与安全边界

当前脚本适配合并后的 Light Campaign：公共模块使用 `aka.agent`、`aka.evaluation`、
`aka.execution` 等命名空间，Workflow 位于 `aka.workflow.light`。无需安装 Legacy。
Skill 和脚本属于 checkout 资源；更新 Python 包后也要同步同一 checkout 的这两部分。

输入始终是顶层 Campaign workspace：

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign
```

状态命令自动发现串行 Campaign 与 pooled-search 下的 `t0001`、`t0002` 等
trajectory，不需要逐个指定子目录。

并行搜索只有一棵 Campaign 私有状态树：各 lineage 的 Research、Session
和 Runtime audit 位于 `lineages/tNNNN/` 命名空间；不可变 Gateway Record 只在
Campaign Measurement Store 中保存一份。观测工具根据 Runtime audit 归属记录，
不会把中央共享证据重复统计成每条 lineage 各自产生的结果。中央 Record 被同伴
lineage 复用时，Record 内的原始 `runtime_request_id` 在整个 Campaign 的 audit
命名空间中解析；消费方只需有指向该 Record 的自身 audit，不要求复制创建方 audit。

- 六个状态/查询命令只读，不会提交、重试、取消或修改 Runtime/Agate 工作。
- 两个 `export_*` 命令只创建新的本地导出目录；目标已存在时拒绝覆盖。
- 一次状态检查不授权重启 Campaign、重提 Agate、清理历史或编辑私有状态。
- 必须使用与 Campaign 数据格式兼容的 AKA checkout；不要用另一版本的脚本
  强行解释旧数据。

八个正式入口如下：

| 问题 | 命令 | 主要权威数据 |
|---|---|---|
| 每条 lineage 是否活着、在做什么、Direction/Experiment 与 Token 状态 | `light_lineage_status.py` | writer lock、Session capture、Research Store、token ledger |
| 每条 lineage 产生过哪些 passing full Evaluate 与 accepted submission | `light_kernel_history.py` | Measurement Store、Campaign Submission Store、token checkpoints |
| accepted Kernel 的全局 GeoMean 与逐 Shape 最优 | `light_kernel_best.py` | verified accepted submissions |
| Runtime Tool 调用次数、成败、耗时及最近调用 | `light_runtime_calls.py` | Runtime lifecycle audit receipts |
| Agate 物理调用次数、成败、排队/执行/总耗时及最近调用 | `light_agate_calls.py` | Gateway Job checkpoints |
| 按 Agent commit 导出源码及全部绑定证据 | `export_light_kernel.py` | Agent Git、Measurement Store、Job checkpoints |
| 按 Gateway Record 展开所有 Agate submission | `light_gateway_jobs.py` | `submission-*.json` 与当前 checkpoint |
| 按 Gateway Record 导出 Runtime 请求、全部原始 Agate 尝试与 Agent 回执 | `export_light_gateway_record.py` | Runtime audit、Gateway Record、Job checkpoints |

顶层目录只保留上述八个正式命令；共享实现位于 `_internal/`，不是受支持的 CLI。

## 2. 共同概念

### 2.1 Scope、trajectory 与 lineage

`scope`/`lineage` 的典型值：

- `campaign`：串行 Campaign 或顶层控制器范围；
- `t0001`、`t0002`：pooled-search trajectory；
- `trajectory`：对应整数 `1`、`2`；顶层 `campaign` 通常为 `null`。

`--lineage` 接收的是 scope 名称，例如 `--lineage t0001`，不是 `T1`、Episode
号或 Session ID。重复传入表示并集：

```bash
--lineage t0001 --lineage t0003
```

### 2.2 五层身份

不要混用以下 ID：

```text
Agent Git commit ID
  └─ Runtime request ID
       └─ Gateway Record ID
            └─ primary Agate Job ID
                 └─ optional Ray/backend Job ID
```

| 身份 | 含义 |
|---|---|
| `kernel_commit_id` | Agent Git 中完整源码树的公开身份；最终 Kernel 身份 |
| `runtime_request_id` | 一次 Runtime HTTP 调用的稳定审计身份 |
| `gateway_record_id` | 一次逻辑测量的不可变规范化 Record；可包含多个分片、重试和 Job |
| `agate_job_id` | 一次物理 Agate submission 的公开可观测性字段 |
| `backend_agate_job_ids` | 外层 Ray Data Job 返回的内部 backend 身份；不保证能独立读取原始响应 |

所有公开可观测性 JSON 统一使用上述名称。只有不可改写的原始证据中可能保留
Runtime 上游 `request_id`，或 Agate 上游 `job_id` 和 `backend_job_ids`。

一个 Gateway Record 可以因为 Shape 分片或基础设施重试关联多个 Agate Job；一个
Agent commit 也可以关联多个 Evaluate、Profile、Check 或 Dev Record。

### 2.3 输出模式

六个状态/查询命令共享：

| 选项 | 行为 |
|---|---|
| 无输出选项 | 面向人的表格摘要 |
| `--json` | 缩进 JSON，适合审计与程序读取 |
| `--compact` | 单行 JSON，并隐含 `--json`，适合日志采集 |

所有结构化状态输出通常包含：

```json
{
  "schema_version": 1,
  "observed_at": "2026-11-13T02:00:00+00:00",
  "campaign_workspace": "/workspace/campaign"
}
```

- `schema_version`：该脚本输出结构版本，不是 Campaign 或 Agate 版本；
- `observed_at`：本次只读快照生成时间；
- `campaign_workspace`：解析后的顶层 workspace。

### 2.4 统一状态

Runtime/Agate 明细统一投影为：

| 状态 | 含义 |
|---|---|
| `succeeded` | 有持久化的成功终态 |
| `failed` | 有持久化的失败/拒绝/超时终态或结构化错误 |
| `pending` | 已知仍为 running/queued/submitted/pending，或存在未终结 checkpoint |
| `unknown` | 持久化证据不足以判定；不等于失败，也不等于仍在运行 |

基础设施错误、传输错误和 queue timeout 不是 Kernel 正确性失败。`unknown` 不应被
包装成候选失败或自动触发重提。

### 2.5 时间字段

| 字段 | 精确定义 |
|---|---|
| Runtime `duration_seconds` | Runtime request 从接收到返回/终结的端到端时间 |
| Agate `queue_seconds` | Agate `created_at → started_at`；无原生 `started_at` 时可使用持久化的 running 阶段边界 |
| Agate `execution_seconds` | Agate `started_at → finished_at`；与 `queue_seconds` 使用同一个阶段边界 |
| Agate `total_seconds` | Agate `created_at → finished_at`，提交到终态 |
| `pending_age_seconds` | 从已知开始/提交时间到本次观测时刻的年龄 |

`waiting_seconds` 及 `average_waiting_seconds` 只是旧消费者兼容别名，等于
`queue_seconds`；它不是 Runtime 端到端等待。

每个时间字段都伴随 `*_source`，汇总中提供 `*_source_counts`：

- `agate_started_at`：Agate 原生 `started_at`；
- `agate_phase_elapsed`：由轮询期间持久化的 `updated_at - phase_elapsed_ms` 得到阶段边界；
- `agate_running_updated_at`：首个 running 观测的 Agate `updated_at`；
- `runtime_observed_running`：首个 running 本地观测时刻，精度受轮询周期限制；
- `agate_created_finished`：Agate 原生 `created_at → finished_at`。

新 SDK 请求会私下持久化 `queued/running/terminal` 阶段观测，以便在
Agate 终态不提供 `started_at` 时仍能报告有来源的分阶段时间。历史 checkpoint
没有这些事实时仍返回 `null`，不从 `total_seconds` 擅自拆分。

统计字段均只使用实际持久化样本：

```json
{
  "average_duration_seconds": 12.4,
  "p50_duration_seconds": 8.0,
  "p95_duration_seconds": 31.2,
  "max_duration_seconds": 35.0,
  "duration_seconds_samples": 9,
  "duration_seconds_coverage": 0.9
}
```

`coverage=0.9` 表示十次调用中九次有可用时间，不代表九成成功率。

### 2.6 `evidence_health`

Kernel 历史、winner 与 Kernel 导出会附带证据健康检查：

```json
{
  "complete": false,
  "issue_counts": {
    "incomplete_gateway_records": 1,
    "unreadable_gateway_records": 0,
    "accepted_submissions_missing_evidence": 0,
    "missing_token_checkpoints": 1,
    "unresolvable_kernel_commits": 0,
    "unlinked_runtime_audits": 0,
    "gateway_records_missing_runtime_audit": 0
  },
  "incomplete_gateway_records": [
    {"scope": "t0001", "gateway_record_id": "gateway-..."}
  ]
}
```

`complete=false` 表示审计链有缺口，不必然说明 Kernel 或评测错误。各列表含义：

- `incomplete_gateway_records`：逻辑测量尚未形成完整 `record.json`；
- `unreadable_gateway_records`：Record 存在但完整性/解析失败；
- `accepted_submissions_missing_evidence`：accepted submission 无法连接到合格 full Evaluate；
- `missing_token_checkpoints`：事件存在，但当时累计 Token checkpoint 缺失；
- `unresolvable_kernel_commits`：accepted commit 无法从保存的 Agent Git/source artifact 解析；
- `unlinked_runtime_audits`：Runtime receipt 指向不存在的 Gateway Record；
- `gateway_records_missing_runtime_audit`：新格式 Gateway Record 有原始 Runtime request ID，但在整个 Campaign 的所有 lineage 中都找不到对应创建 receipt。条目同时给出 `gateway_record_id` 与 `visible_scopes`；同伴复用本身不属于缺失。

旧 Campaign 可能天然没有 request ID 或 token checkpoint；工具会报告缺失，不按时间戳
猜测关联。

## 3. `light_lineage_status.py`

### 3.1 用途

回答“各条搜索线现在是否在运行、Agent 在做什么、研究记录与 Token 花了多少”。
它是 Campaign 总览的首选入口。

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" [options]
```

### 3.2 选项

| 选项 | 默认 | 含义 |
|---|---:|---|
| `--stale-after SECONDS` | `600` | 没有活跃 Supervisor 请求且最近活动超过此秒数时，将持锁 lineage 标为 `running_no_recent_activity`；只影响分类，不停止进程 |
| `--lineage NAME` | 全部 | 只返回指定 lineage；可重复 |
| `--research-limit N` | `50` | 每条 lineage 最多返回 N 条 Direction 和 N 条 Experiment 明细；必须为正整数 |
| `--all-research` | 关闭 | 返回全部 Direction/Experiment，并覆盖 `--research-limit` |
| `--json` | 关闭 | 缩进 JSON |
| `--compact` | 关闭 | 单行 JSON |

### 3.3 代表性返回

```json
{
  "campaign_configuration": {
    "search": {"trajectories": 3, "synchronization": "asynchronous"},
    "search_binding_digest": "...",
    "token_budget": 0
  },
  "lineages": [
    {
      "lineage": "t0001",
      "trajectory": 1,
      "workspace": "/workspace/.../t0001",
      "runtime": {
        "phase": "exploring",
        "writer_active": true,
        "liveness": "running_recent_activity",
        "current_episode": 2,
        "completed_episodes": 1,
        "latest_activity_at": "...",
        "stagnation_seconds": 14.2,
        "main_session_silence_seconds": 14.2,
        "active_session_count": 1,
        "active_sessions": [
          {"phase": "waiting_agate", "tool": "evaluate", "activity_state": "recent_activity", "conversation_path": "/campaign/lineages/t0001/segments/episodes/e00000002/sessions/run-00000001/conversation.jsonl"}
        ],
        "latest_session": {
          "session_id": "session-example",
          "conversation_path": "/campaign/lineages/t0001/segments/episodes/e00000002/sessions/run-00000001/conversation.jsonl"
        },
        "active_supervisor_activity_count": 1,
        "supervisor_activity": {
          "phase": "claim_review",
          "operation": "record-experiment",
          "state": "running",
          "elapsed_seconds": 1842.0
        }
      },
      "runtime_configuration": {"optimization_mode": "ultra", "hardware": "L20D"},
      "research": {
        "enabled_modules": ["directions", "experiments"],
        "direction_counts": {"in_progress": 1, "deferred": 2},
        "directions": [{"direction_id": "direction_...", "status": "in_progress", "local_rank": 1}],
        "experiment_count": 3,
        "experiments": [{"experiment_id": "experiment_...", "gateway_record_ids": ["gateway-..."]}],
        "focus_direction_id": "direction_...",
        "active_direction_ids": ["direction_..."],
        "local_direction_queue": ["direction_..."],
        "shared_direction_queue": ["direction_peer_..."],
        "last_shared_queue_sync": {"sequence": 27, "origin": {"epoch": 2}},
        "peer_git_refs": {"available": true, "count": 2, "refs": []},
        "claim_reviews": {"count": 4, "quality_counts": {"supported": 3}}
      },
      "tokens": {
        "main_agent": {"total_tokens": 120000, "reasoning_tokens": 20000, "measurement": "exact"},
        "reviews": {"total_tokens": 18000, "measurement": "exact"},
        "all_paths": {"total_tokens": 138000, "measurement": "exact"}
      },
      "search_lineage": {"epoch": 2, "status": "running"}
    }
  ]
}
```

### 3.4 关键字段

- `writer_active`：唯一直接的当前执行锁事实；`false` 时不能声称 trajectory 仍在运行。
- `liveness`：`running_recent_activity`、`running_supervisor_activity`、
  `running_no_recent_activity` 或 `not_running`。`running_supervisor_activity` 表示主 Session
  可能暂时静默，但有权威 Runtime 请求仍在执行；长 Claim Review 不再误报为 lineage 停滞。
- `phase`：Campaign 阶段；Session `phase` 常见为 `thinking`、`executing_tool`、
  `executing_runtime_tool`、`waiting_agate`、`waiting_wiki`。超过 stale 阈值且没有
  Provider 事件的原 `thinking` 会显示为 `provider_wait_unknown`，避免把未知等待误称为持续思考。
- `main_session_silence_seconds`：主 Agent Session 距最后事件的时间；即使 Supervisor 正在
  Review 也保留，便于区分“Session 静默”和“整个 lineage 无活动”。
- `active_sessions[].conversation_path` 与 `latest_session.conversation_path`：主 Agent 捕获对话
  的权威路径；分析“最近在优化什么”时默认只读取该 JSONL 的最后 100 条事件。
- `supervisor_activity`：最近或当前的权威 Runtime 请求。`active=true` 时包含 operation、
  state 和已运行时间；`record-experiment` / `update-direction` 的长请求会投影为
  `claim_review`。已完成的后台 Runtime 请求也会更新 lineage 的最近活动时间。
- `runtime_configuration`：实际 Runtime 配置快照；用于确认 mode、hardware、timeout、
  shape batch 等，不应用启动命令猜测。新 Campaign 从顶层 `campaign.json` 读取；
  独立 `runtime-config.json` 仅作为旧 Campaign 的只读兼容输入。
- `details_truncated`：Direction/Experiment 是否因限制被截断。
- `local_direction_queue` 与 `shared_direction_queue`：本地偏好与同伴广播队列，不能
  据此声称 Agent 已采纳同伴实现。
- `peer_git_refs`：导入的只读同伴 Git refs；不代表当前 HEAD 被修改。
- `tokens.main_agent`、`tokens.reviews`、`tokens.all_paths`：分别为主 Agent、独立
  Review 和两者合计。Reasoning 是 breakdown，不应再次加到 total。

## 4. `light_kernel_history.py`

### 4.1 用途

回答“每条 lineage 在什么累计 Token 时产生了哪个正确 full Evaluate，以及哪些
commit 后来通过了 `submit-kernel`”。

```bash
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" [options]
```

### 4.2 选项

| 选项 | 含义 |
|---|---|
| `--lineage NAME` | 只返回指定 lineage；可重复 |
| `--kernel-commit-id SHA` | 只返回指定完整 Agent commit；可重复 |
| `--last N` | 每条 lineage 保留最新 N 条 `kernel_results`；不是全局 N 条 |
| `--json` / `--compact` | JSON 输出模式 |

### 4.3 代表性返回

```json
{
  "lineages": [
    {
      "lineage": "t0001",
      "trajectory": 1,
      "kernel_results": [
        {
          "kernel_commit_id": "012345...",
          "gateway_record_id": "gateway-...",
          "evaluated_at": "...",
          "performance": {
            "latency_us_geomean": 4.096,
            "latency_us_arith_mean": 4.21,
            "shape_count": 10,
            "latency_us_by_shape": {"0": 2.1, "1": 3.4}
          },
          "evaluate_token_usage": {
            "checkpoint_id": "...",
            "all_paths": {"total_tokens": 820000},
            "main_agent": {"total_tokens": 760000},
            "reviews": {"total_tokens": 60000}
          },
          "evaluate_token_usage_availability": "available",
          "submissions": [
            {
              "submission_id": "submission-...",
              "submission_revision": 2,
              "accepted": true,
              "evidence_status": "verified",
              "source_contract": "pass",
              "review": {
                "schema_version": 2,
                "verdict": "allow",
                "checks": [{
                  "id": "compute_provenance",
                  "decision": "allow",
                  "category": "self_authored_compute",
                  "reason": "候选源码自行实现核心计算",
                  "evidence": ["candidate/kernel.py:12"]
                }],
                "dependencies": [],
                "summary": "最终源码满足当前模式要求"
              },
              "implementation_review": {
                "availability": "available",
                "dsl_policy": "guided",
                "configured_dsl": null,
                "required_phase": "open_dsl"
              },
              "diagnostics": [],
              "submit_token_usage": {
                "checkpoint_id": "...",
                "all_paths": {"total_tokens": 850000}
              },
              "submit_token_usage_availability": "available"
            }
          ]
        }
      ]
    }
  ],
  "evidence_health": {"complete": true, "issue_counts": {}},
  "detail_selection": {
    "lineages": [], "kernel_commit_ids": [], "last_per_lineage": null
  }
}
```

### 4.4 资格与含义

有 `performance` 的 `kernel_results[]` 只包含同时满足以下条件的 Record：

1. operation 为标准 `evaluate`；
2. `evaluation_mode=full`，没有 Shape 子集、自定义 input/shapes 或 baseline；
3. 命令为标准 `python3 test_kernel.py`；
4. response 未截断、退出码为 0、`all_pass=true`；
5. 每个 Shape 都有正的可解析 latency；
6. Record 的 candidate identity 是完整 Agent Git commit。

`submissions[]` 按同一 lineage、commit 和 Gateway Record 精确嵌入对应 Evaluate，不再
复制性能字段。没有可关联合格 Evaluate 的 submission 仍会保留，但其
`performance=null`；不会按时间猜测关联。`accepted=true` 表示当时的正式提交回执接受，
只有同时满足 `evidence_status=verified` 的记录才可用于 winner。

`implementation_review.technologies` 是独立终态源码审查实际观察到的实现技术，例如
`cuda`、`triton`、`cutedsl` 或混合列表；它不是根据 Campaign mode 或配置 DSL 猜出的。
`availability=unavailable` 表示旧 submission 或审查事实没有持久化，工具不会反推。
面向人的表格将这些技术显示在 `Implementation` 列。

`evaluate_token_usage` 与 `submit_token_usage` 分别是 Evaluate 完成和提交审查完成时的
累计 checkpoint，不是用 Campaign 最终总量倒推。对应 availability 为
`missing_checkpoint` 时仍保留事件，但不能声称其精确 Token 成本。未接受的提交目前不会
生成提交里程碑，因而显示 `submit_token_usage_availability=not_recorded`。

提交也可以复用中央 Measurement Store 中同伴的完整 Evaluate。此时展示已核验的性能，
但消费方没有本地 Evaluate Token checkpoint：`evaluate_token_usage=null`，availability 为
`shared_evidence_no_local_evaluate`；不会将同伴的 Evaluate Token 算给消费方。

## 5. `light_kernel_best.py`

### 5.1 用途与选项

从 verified accepted submissions 中选择 Campaign winner；没有筛选选项：

```bash
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
```

仅支持公共 `--json` 与 `--compact`。

### 5.2 代表性返回

```json
{
  "accepted_submission_count": 4,
  "geomean_best": {
    "scope": "t0002",
    "kernel_commit_id": "89abcd...",
    "gateway_record_id": "gateway-...",
    "latency_us_geomean": 3.91,
    "shape_count": 10
  },
  "shape_best": {
    "0": {
      "shape_id": "0",
      "latency_us": 1.84,
      "kernel_commit_id": "012345...",
      "gateway_record_id": "gateway-...",
      "lineage": "t0001"
    }
  },
  "evidence_health": {"complete": true, "issue_counts": {}}
}
```

- `geomean_best` 是一个真实 accepted commit；无合格提交时为 `null`。
- `shape_best` 每个 Shape 独立选择，因此可能来自不同 commit。它是 Pareto 观察，
  不是一个可 checkout 的合成 Kernel。
- 未调用或未通过 `submit-kernel` 的 passing full Evaluate 不进入本工具，即使更快。

## 6. `light_runtime_calls.py`

### 6.1 用途

回答“各类 Runtime Tool 调了多少次、成功/失败多少、端到端耗时如何，以及最近哪些
调用失败或仍在执行”。

```bash
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" [options]
```

### 6.2 选项

| 选项 | 含义 |
|---|---|
| `--kind KIND` | Runtime tool kind，例如 `evaluate`、`profile`、`check`、`dev`、`command`、`kernel_submit`；可重复 |
| `--status STATUS` | `succeeded`、`failed`、`pending`、`unknown`；可重复 |
| `--lineage NAME` | 限定 scope；可重复 |
| `--since ISO_TIME` | 明细的包含式开始时间 |
| `--until ISO_TIME` | 明细的包含式结束时间 |
| `--last N` | 应用以上筛选后取最新 N 次 |
| `--json` / `--compact` | JSON 输出模式 |

`--since/--until` 接受 ISO-8601，例如 `2026-11-13T00:00:00Z`。开始时间晚于
结束时间或 `N<=0` 会报参数错误。

没有任何明细筛选时，`calls=[]`，避免默认输出巨大审计明细；`summary` 始终覆盖整个
Campaign。筛选只影响 `calls[]`，不会改变 `summary`。

### 6.3 代表性返回

```json
{
  "summary": [
    {
      "kind": "evaluate",
      "calls": 12,
      "succeeded": 9,
      "failed": 2,
      "pending": 1,
      "unknown": 0,
      "error_counts": {"job_deadline_exceeded": 2},
      "average_duration_seconds": 48.2,
      "p50_duration_seconds": 31.0,
      "p95_duration_seconds": 141.0,
      "max_duration_seconds": 156.0,
      "duration_seconds_samples": 11,
      "duration_seconds_coverage": 0.9167,
      "oldest_pending_seconds": 82.0
    }
  ],
  "calls": [
    {
      "scope": "t0001",
      "runtime_request_id": "4f...",
      "kind": "evaluate",
      "operation_family": "gateway",
      "status": "failed",
      "success": false,
      "success_basis": "request_lifecycle",
      "created_at": "...",
      "finished_at": "...",
      "duration_seconds": 156.0,
      "http_status": 503,
      "exit_code": null,
      "gateway_record_id": "gateway-...",
      "error": {"code": "infrastructure_unavailable", "error_class": "infra"}
    }
  ],
  "detail_selection": {
    "kinds": ["evaluate"],
    "statuses": ["failed"],
    "lineages": [],
    "since": null,
    "until": null,
    "last": 20,
    "returned": 1,
    "campaign_total": 12,
    "summary_scope": "campaign"
  }
}
```

`success_basis=request_lifecycle` 表示成功事实来自新格式 receipt；
`audit_exit_code` 表示旧审计只能依据进程退出码；`unavailable` 表示无法判定。

## 7. `light_agate_calls.py`

### 7.1 用途与选项

参数形式与 Runtime calls 相同，但统计单位是物理 Agate submission：

```bash
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" \
  --kind evaluate --status failed --lineage t0001 --last 20 --json
```

| 选项 | 含义 |
|---|---|
| `--kind KIND` | Agate operation kind；可重复 |
| `--status STATUS` | `succeeded/failed/pending/unknown`；可重复 |
| `--lineage NAME` | scope；可重复 |
| `--since/--until ISO_TIME` | 按 Job checkpoint 更新时间筛选明细，包含边界 |
| `--last N` | 筛选后最新 N 个物理 submission |
| `--json` / `--compact` | JSON 输出模式 |

同样地，无筛选时 `calls=[]`，`summary` 始终为 Campaign-wide。

### 7.2 代表性返回

```json
{
  "summary": [
    {
      "kind": "evaluate",
      "calls": 18,
      "succeeded": 14,
      "failed": 3,
      "pending": 1,
      "unknown": 0,
      "error_counts": {"queue_timeout": 1, "job_deadline_exceeded": 2},
      "average_queue_seconds": 12.0,
      "p95_queue_seconds": 44.0,
      "queue_seconds_samples": 15,
      "queue_seconds_coverage": 0.8333,
      "queue_seconds_source_counts": {"agate_phase_elapsed": 15},
      "average_execution_seconds": 60.0,
      "execution_seconds_source_counts": {"agate_phase_elapsed": 15},
      "average_total_seconds": 72.0,
      "total_seconds_source_counts": {"agate_created_finished": 18},
      "oldest_pending_seconds": 91.0,
      "average_waiting_seconds": 12.0
    }
  ],
  "calls": [
    {
      "scope": "t0001",
      "operation": "evaluate",
      "kind": "evaluate",
      "status": "failed",
      "agate_job_id": "ev_...",
      "gateway_record_id": "gateway-...",
      "checkpoint_path": "batch-0001/submission-0002.json",
      "checkpoint_phase": "terminal",
      "submission": 2,
      "runtime_request_ids": ["4f..."],
      "queue_seconds": 5.1,
      "queue_seconds_source": "agate_phase_elapsed",
      "execution_seconds": 600.0,
      "execution_seconds_source": "agate_phase_elapsed",
      "total_seconds": 605.1,
      "total_seconds_source": "agate_created_finished",
      "pending_age_seconds": null,
      "error": {"reason": "job deadline exceeded", "category": "timeout"}
    }
  ],
  "detail_selection": {"summary_scope": "campaign", "returned": 1}
}
```

这里 `calls=18` 是物理 submission 数，不是逻辑 Evaluate 数；一个十 Shape Evaluate
可能拆成多个物理 Job，重试也会增加 submission 数。

## 8. `export_light_kernel.py`

### 8.1 用途与选项

导出一个完整 Agent commit 及所有与它精确绑定的测量：

```bash
python3 "$AKA_REPO/observability/export_light_kernel.py" "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-kernel-evidence
```

| 选项 | 要求 |
|---|---|
| `--kernel-commit-id SHA` | 完整 Agent Git commit ID；必须能从 Campaign 保存的 Git/source artifact 解析 |
| `--output PATH` | 必须不存在；工具不会合并或覆盖 |

该命令总是向 stdout 输出 JSON manifest 摘要，没有 `--json` 选项。

### 8.2 目录结构

```text
new-kernel-evidence/
├── manifest.json
├── kernel/                         # 完整源码树，保留 source_tree/SOL/CUDA manifest 布局
└── measurements/
    └── gateway-.../
        ├── record.json             # 完整规范化 Gateway Record
        ├── request.json            # Runtime 规范化请求
        ├── agent-response.json     # 实际返回 Agent 的投影
        └── agate-jobs.json         # 每次物理 submission、原始 response/process
```

`manifest.json` 代表性字段：

```json
{
  "kernel_commit_id": "012345...",
  "source_manifest": {"output_directory": "kernel", "files": ["kernel.py", "impl/a.cu"]},
  "measurement_count": 3,
  "measurements": [
    {
      "scope": "t0001",
      "gateway_record_id": "gateway-...",
      "operation": "evaluate",
      "passing_standard_full_evaluate": true,
      "runtime_request_id": "4f...",
      "result_summary": {
        "all_pass": true,
        "latency_us_geomean": 4.096,
        "shape_count": 10,
        "performance_objective": "geomean"
      },
      "agate_job_ids": ["ev_..."],
      "backend_agate_job_ids": []
    }
  ],
  "kernel_submissions": [],
  "evidence_health": {"complete": true},
  "output_directory": "/path/to/new-kernel-evidence"
}
```

`kernel_submissions=[]` 只说明该 commit 没有 submission receipt，不表示测量不存在。

## 9. `light_gateway_jobs.py`

### 9.1 用途与选项

将一个逻辑 Gateway Record 展开成所有成功、失败、pending、被复用和重试的物理
submission：

```bash
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
```

| 选项 | 要求 |
|---|---|
| `--gateway-record-id ID` | 必填，必须是完整 `gateway-...` ID |
| `--json` / `--compact` | JSON 输出模式 |

### 9.2 代表性返回

```json
{
  "gateway_record_id": "gateway-...",
  "summary": {
    "record_location_count": 1,
    "submission_attempt_count": 3,
    "agate_job_id_count": 3,
    "backend_agate_job_id_count": 1,
    "reused_attempt_count": 1,
    "status_counts": {"failed": 1, "succeeded": 2}
  },
  "locations": [
    {
      "scope": "t0001",
      "operation": "evaluate",
      "complete": true,
      "runtime_request_id": "4f...",
      "submission_attempt_count": 3,
      "record_directory": "/private/.../gateway-..."
    }
  ],
  "attempts": [
    {
      "scope": "t0001",
      "checkpoint_path": "batch-0001",
      "checkpoint_id": "...",
      "submission": 2,
      "checkpoint_phase": "terminal",
      "runtime_request_ids": ["4f..."],
      "agate_job_id": "ev_...",
      "raw_status": "Completed",
      "status": "succeeded",
      "reused": false,
      "shape_ids": ["0", "1", "2"],
      "backend_agate_job_ids": ["ray_..."],
      "retry_counters": {"infra": 1},
      "error": null,
      "snapshot_path": "/private/.../submission-0002.json"
    }
  ]
}
```

- `submission_attempt_count` 包含历史尝试，不只当前 checkpoint；
- `agate_job_id_count` 与 `backend_agate_job_id_count` 分别去重统计外层 Agate Job
  和 Ray/backend ID；
- `reused=true` 表示结果来自已有 batch checkpoint，不代表新提交了 Job；
- `shape_ids` 是该物理 submission 的 Shape 子集；
- `error` 为结构化的 category/reason/message，保持原始故障语义。

## 10. `export_light_gateway_record.py`

### 10.1 用途与选项

按一个逻辑 Gateway Record ID 导出完整关联链。该入口会一次覆盖 Record 下的所有
分片、重试、primary Job 和 Ray/backend ID：

```bash
python3 "$AKA_REPO/observability/export_light_gateway_record.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-gateway-evidence
```

| 选项 | 要求 |
|---|---|
| `--gateway-record-id ID` | 必填，完整 `gateway-...` ID |
| `--output PATH` | 必须不存在 |

该命令总是输出 JSON manifest 摘要。同一 Record 若在本地与共享 trajectory 中有
多个持久化位置，导出器会先验证请求、Gateway Record 与 Agent 回执完全一致，再将
Runtime audit 和 Agate attempt 去重合并。导出目录始终只保留一份规范内容；物理来源
仅记录在 `manifest.json.source_locations` 中。相同 Gateway ID 的规范内容若发生冲突，
导出会以 `gateway_record_integrity_conflict` 失败，不会任意选择副本。

### 10.2 目录结构与返回

```text
new-gateway-evidence/
├── manifest.json
├── runtime-request.json
├── runtime-audits.json
├── agate-attempts.json
├── agent-response.json
└── gateway-record.json
```

```json
{
  "gateway_record_id": "gateway-...",
  "source_location_count": 2,
  "runtime_request_ids": ["4f..."],
  "agate_job_ids": ["ev_..."],
  "backend_agate_job_ids": ["ray_..."],
  "runtime_audit_count": 1,
  "agate_attempt_count": 3,
  "agent_response_available": true,
  "source_locations": [
    {
      "scope": "t0001",
      "runtime_request_ids": ["4f..."],
      "runtime_audit_count": 1,
      "agate_attempt_count": 3
    }
  ],
  "output_directory": "/path/to/new-gateway-evidence"
}
```

`agate-attempts.json` 为每次物理尝试保存 `agate_job_id`、
`backend_agate_job_ids`、状态、
重试计数、完整外层 Agate 响应和 raw process。backend ID 只嵌在 Ray 外层响应时，不要
把外层响应称为该 backend Job 的独立原始返回。`agent-response.json` 是由不可变
Gateway Record 按当前 Runtime HTTP 契约确定性重建的 Agent 可见 JSON body；它不是
Record 中保存的完整私有执行响应。重复持久化的 Runtime audit 和 Agate attempt 只导出
一次，各自的 `source_audit_paths` / `source_snapshot_paths` 保留审计来源。

## 11. 常用排查流程

### 11.1 当前整体状态

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" --last 5 --json
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
```

先用 `writer_active/liveness` 判断是否运行，再看 Session phase；不要从 scheduler
checkpoint 推断进程存活。

### 11.2 最近 Runtime 失败

```bash
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" \
  --status failed --last 20 --json
```

查看 `error.code/reason/error_class`、`http_status`、`runtime_request_id` 和
`gateway_record_id`。若已有 Gateway Record，继续沿 Record 排查，而不是重提。

### 11.3 最近 Agate 故障和排队

```bash
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" \
  --status failed --status pending --last 20 --json
```

- queue 高、execution 正常：主要是排队；
- queue 正常、execution 接近 deadline：执行慢或单 Shape 超时；
- `total_seconds` 与两者不一致：检查时间覆盖和缺失字段，不自行补值；
- 同一 Gateway Record 多个失败 Job：用 `light_gateway_jobs.py` 查看是否为重试或分片。

### 11.4 从 Kernel 追到原始 Agate

```text
light_kernel_history.py
  → kernel_commit_id + gateway_record_id
export_light_kernel.py
  → 全源码 + measurements/*/agate-jobs.json
light_gateway_jobs.py
  → 所有 agate_job_id/backend_agate_job_ids
export_light_gateway_record.py
  → 该逻辑请求的 Runtime 请求 + 全部 Agate 尝试 + Agent 实际回执
```

### 11.5 判断 Direction/Experiment 是否工作

查看 `light_lineage_status.py`：

1. `enabled_modules` 是否符合启动配置；
2. Direction lifecycle 与 local/shared rank 是否变化；
3. Experiment 是否有 `gateway_record_ids` 和 candidate commit；
4. `claim_reviews.quality_counts` 是否长期只有 partially supported/unreviewed；
5. `last_shared_queue_sync` 和 `peer_git_refs` 是否在同步边界更新。

Review 通过不等于 Kernel 被接受；Experiment supported 也不等于 Direction 应关闭。

## 12. Token 解释

统一 usage 字段可能包含：

```json
{
  "input_tokens": 100000,
  "output_tokens": 20000,
  "reasoning_tokens": 8000,
  "cache_read_tokens": 50000,
  "cache_write_tokens": 1000,
  "total_tokens": 120000,
  "measurement": "exact"
}
```

- `total_tokens` 采用 Provider/Session 的总量口径；
- `reasoning_tokens` 是 breakdown，不能再次加到 `total_tokens`；
- `main_agent` 与 `reviews` 分开；`all_paths` 才是全部路径合计；
- Review 还可按 `final_source_policy_review`、`direction_claim_review`、
  `experiment_claim_review`、`supplemental_numerical_review` 细分；
- `measurement=exact`：全部来自原生精确计数；
- `estimated`：Provider 无计数，使用最终 conversation 尽力估计；
- `partial`：只有部分调用/字段有计数；
- `unavailable`：没有可验证计数。

不要把估算 Token 与精确 Token 混在同一统计结论中而不说明 measurement。

## 13. 错误与退出码

- 成功为退出码 `0`。
- workspace 不存在、ID 格式不完整、Record/Job/commit 不可用、时间参数非法等为
  退出码 `1`，错误写入 stderr。
- argparse 缺少必填参数或选项格式错误通常为退出码 `2`。
- 导出目标已存在时拒绝，不会覆盖。

错误信息本身也是事实边界。例如“Job ID 不在此 Campaign”不证明 Job 从未存在，
只证明当前持久化 Campaign 中没有该 ID。

## 14. 报告规范

面向用户报告时至少保留：

- Campaign workspace 与观测时间；
- lineage/trajectory；
- Agent commit、Gateway Record、Runtime request 和 Agate Job 的正确身份；
- writer liveness 与最后活动时间；
- 成功/失败/pending/unknown 的原始分类；
- queue、execution、Agate total 与 Runtime duration 的区别；
- Token measurement 是 exact、estimated、partial 还是 unavailable；
- 会改变结论的 `evidence_health` 缺口。

不得用旧的 Framework Baseline、handoff、promotion、canonical `vN` 或 Episode
`attempt.json.accepted` 作为当前 accepted winner 权威。当前 winner 是：

```text
accepted submit-kernel receipt for exact Agent commit
∩ passing standard full Evaluate for that commit
∩ complete per-Shape performance evidence
```

## 15. 只读 Dashboard

`light_dashboard.py` 使用以上八个工具的同一套快照，展示状态、Research、Token、Kernel、调用和证据。

```bash
python3 "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --port 8765
python3 "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --capture /tmp/dashboard.json
python3 "$AKA_REPO/observability/light_dashboard.py" /tmp/dashboard.json --port 8765
```

默认仅绑定 `127.0.0.1:8765`；`--port 0` 自动选择端口，`--refresh-ms` 默认 5000 毫秒。
远端 Campaign 必须在远端执行 `--capture`，下载 JSON 后本地展示；存活状态以
`captured_at` 为准，不会用本地 PID/锁重新判断。页面没有启动、重试、取消或提交入口。
参数与 HTTP 说明见 [Dashboard 文档](../docs/optimization-dashboard.md)。

### Claude Token 统计口径

主 Agent 和独立 Review 均使用去重后的逐响应 usage 累计，运行中和结束后保持同一口径；`result.usage` 仅保存在 `terminal` 供审计，不覆盖累计值。输入、输出、缓存读写分别统计；reasoning 是单独展示的子项，不额外重复加入总量。缺少响应计数时明确标记不完整或不可用；对话文本估计仍标记为 estimated。历史不可变 checkpoint 不会自动重算。
