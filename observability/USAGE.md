# AKA Light 可观测性操作手册

所有命令只读；输入顶层 Campaign workspace，并自动发现 pooled-search
trajectory。默认输出表格，`--json` 输出结构化数据，`--compact` 输出单行 JSON。

## 职责边界

| 需求 | 命令 | 权威数据 |
|---|---|---|
| Campaign、Episode 分段、Session 活性 | `light_runtime_status.py` | public runtime state、Session capture |
| V0、Evaluate、accepted frontier、逐 Shape 结果 | `light_performance.py` | Measurement Store + submission Journal |
| 最佳 accepted 与最佳 observed full Evaluate | `light_latency_status.py` | submission Journal + Measurement Store |
| `submit-kernel` 接受/拒绝与证据 | `light_submission_status.py` | `kernel_submissions[]` |
| Agate 调用总体健康、当前 checkpoint、历史 retry 计数 | `light_agate_status.py` | Gateway Record job checkpoints |
| 单个 Gateway Record 的每次物理 Agate submission | `light_gateway_jobs.py` | `submission-*.json` |
| 主 Agent、每类 Review、总 Token 与里程碑快照 | `light_token_time.py` | Session/Review capture、token checkpoints |
| Runtime Tool 调用次数、时延、结果 | `light_tool_usage.py` | Runtime audit receipts |
| Direction、Experiment、队列与 Claim Review | `light_research_status.py` | Research Store 与私有 Claim audit |
| Wiki 查询与采用 | `light_wiki_usage.py` | Wiki audit 与 Experiment references |
| 导出一个 Gateway Record 的源码、Job、Agent 回执 | `export_light_evidence.py` | Measurement Store |
| 导出一个 commit/Record 的源码树 | `export_light_source.py` | Measurement Store / Agent Git |

`light_acceptance.py` 仅是 `light_submission_status.py` 的兼容别名。新脚本和自动化
不得把 Episode `attempt.json.accepted`、handoff、promotion 或 Framework Baseline
解释为当前接受事实。

## 常用命令

```bash
CAMPAIGN=/path/to/light-campaign

python3 observability/light_runtime_status.py "$CAMPAIGN"
python3 observability/light_performance.py "$CAMPAIGN" --json
python3 observability/light_submission_status.py "$CAMPAIGN" --json
python3 observability/light_agate_status.py "$CAMPAIGN"
python3 observability/light_token_time.py "$CAMPAIGN"
python3 observability/light_research_status.py "$CAMPAIGN"
```

持续读取延迟：

```bash
python3 observability/light_latency_status.py "$CAMPAIGN" --watch 30 --compact
```

## 接受事实与性能事实

`submit-kernel` 的接受记录位于 Supervisor 私有 Journal：

```text
journals/episodes/e*/journal.json -> kernel_submissions[]
```

`light_submission_status.py` 会将每一条提交与其
`evaluation_evidence.gateway_record_id` 精确连接，并验证：

1. Review 接受该 Agent Git commit；
2. 引用的是标准、完整、正确性通过且性能可解析的 full Evaluate；
3. Gateway Record 中的 candidate commit 与提交 commit 完全一致。

主要字段：

- `submissions[]`：所有接受/拒绝尝试；
- `accepted_submissions[]`：证据验证通过的接受记录；
- `lineage_best[]`：每条 lineage 最快的 accepted submission；
- `global_best`：仅多 trajectory 时展示；
- `evidence_status`：缺 Record、Record 不合格或 commit 不一致时的明确原因。

`light_performance.py` 区分：

- `accepted_best_by_shape`：只包含 accepted submission；
- `observed_best_by_shape`：包含尚未提交的 passing Evaluate，仅供诊断；
- `best_passing_full_eval`：全 Campaign 中最快的标准 full Evaluate；
- `accepted_best`：最快的 verified accepted submission。

## Agate 当前状态与历史尝试

`light_agate_status.py` 是概览：

- `current_checkpoint_job_count` / `current_jobs`：每个 checkpoint 当前状态；
- `submission_attempt_count` / `submission_attempts`：包括失败重试在内的历史尝试；
- `calls[]`：按 Gateway Record 汇总的逻辑 Runtime 调用。

深入一条 Record：

```bash
python3 observability/light_gateway_jobs.py "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
```

输出会列出每次 submission 的 Agate Job ID、最终状态、错误类别/原因、Shape、
backend Job ID、重试计数和原始 snapshot 路径。成功和失败尝试都会保留。

## Token

`light_token_time.py` 分开统计：

- `main_agent_totals`；
- `review_totals_by_kind`；
- `review_totals`；
- `all_paths_totals`；
- `passing_full_evaluate_token_checkpoints`；
- `accepted_submission_token_checkpoints`。

Reasoning token 单列，不重复加入 total。Provider 缾失 usage 时使用 Session 中可验证的
尽力估算，并标注 `measurement=estimated/partial/unavailable`。

## 导出证据

```bash
python3 observability/export_light_evidence.py "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-directory

python3 observability/export_light_source.py "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-source
```

工具不会覆盖已有输出目录。
