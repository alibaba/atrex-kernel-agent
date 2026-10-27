# 作业、附件与排障

## 作业身份与等待

`eval` 默认提交后等终态；`--no-wait` 只返回 `job_id`，之后用 `agate get JOB_ID --wait`。
原始 `agate submit payload.json` 默认不阻塞，加 `--wait` 才等待。
`execution_phase` 区分网关排队、后端提交／排队、执行与收集，`phase_elapsed_ms` 是当前阶段耗时。
取消受理不代表远端已停止，继续用原 `job_id` 查终态。

| 超时参数 | 范围 |
|---|---|
| `--job-timeout` | worker 作业预算，以对应命令与服务端配置为准 |
| `--wait-timeout` | CLI 等终态的预算 |
| `--http-timeout` | 单次 HTTP 请求预算 |

不同子命令支持的参数不同，先查其 `--help`；旧 `--timeout` 同时设置三者，已弃用。
默认网关排队上限为两小时，原有更短预算仍生效，重启和重试不会重置起点。
`timeout/gateway_queue_timeout` 表示在网关排队到期，`exclusive_lane_queue_timeout` 表示独占 profile 尚未提交后端。

运行中超时会返回 `failed` 和 `finished_at`。`cleanup_pending=true` 表示远端执行仍待停止确认，
有界清理期间暂保留预约；缺失回执立即归档，其他超时清理最多 60 秒。
`execution_quarantined=true` 且 `cleanup_pending=false` 表示未知执行已归档，原身份仍保留，
不自动访问原远端。404 不能证明未执行；不要因超时、取消或断连换身份重复提交。
远端终态已确认的结果仍按收集预算完成，原资源池停用不丢弃已收集路径。

## 限流、重试与权限

HTTP 429 的 `gateway_overloaded`／`gpu_queue_full`／`tenant_gpu_queue_full` 不创建作业。
客户端遵守 `Retry-After`，指数退避加随机抖动；30 秒只限制额外随机退避，不截短服务端等待要求。
预算不足时返回原错误，调用方在业务截止时间内沿用原请求和稳定幂等键稍后重试。
同一次逻辑提交自动生成一次调用级幂等键；跨进程去重仍需显式传入稳定键。
批量调用按同一 AK 的未完成 GPU 单位控制窗口，多进程共享 AK 时合计控制，不只限制 HTTP 并发。

`503 coordinator_busy` 可能来自历史需求补齐，不创建作业，也不触发 SDK 的自动 429 重试。
`503 cluster_configuration_changed` 等待配置同步后用同一键重试。
`403 cluster_ak_denied` 需管理员授权或选择其他环境，不能当过载重试；管理员身份与工号不替代 AK 权限。
权限变化不取消已有作业，原请求的查询、取消与幂等重放仍沿用原权限；跨租户查询为 404。

`503 job_record_unreadable` 且 `details.scope=job` 只影响该记录，保留原身份等待修复，其他作业可查询。
`owner_unavailable` 或连接中断同样需先确认原提交。
`ipc_connection_limit` 表示 IPC 容量触顶；`ipc_operation_failed`／`ipc_response_invalid` 是业务或编码错误，
保留 trace id 交运维排查，不能一律当 owner 不可达。

普通读取和统一幂等提交短暂传输失败时默认最多恢复两次，共享原 HTTP 预算和原请求体／幂等键。
上传、取消、旧提交接口、强制探测及永久 DNS／TLS 校验错误不自动恢复。
SDK `TransportError` 的 `status=None`，安全 details 包含原因、阶段、errno 和尝试次数；
可用 `Client(..., transport_retries=0)` 观察原始稳定性。恢复成功也不抹去已发生的失败。

## 依赖与环境变量

```bash
agate eval --gpu H20 --candidate my_kernel.py --reference-dir data/rms_norm -r requirements.txt
agate eval --gpu H20 --candidate my_kernel.py --reference-dir data/rms_norm -e TORCH_LOGS=+dynamo
```

`-r` 可重复，接受 pip 规格或 HTTPS wheel；拒绝 HTTP／file／git+／editable／本地路径。
默认 `freeze_installed` 复用 base 已满足的包，只安装缺少或不满足的依赖；`no_deps` 只装列出包，
缺少的间接依赖须自行列出。安装在每作业独立环境，支持 eval 两套 harness、profile、check 与 disassemble。
wheel 须能从目标 GPU 地域访问；安装失败回 `deps_install_failed`，工具错误在 `details.logs_tail`。
`-e/--env-var` 由服务端前缀白名单控制，例如 `TRITON_`／`TORCH_`，不是任意环境访问入口。

## OSS 附件

`eval`、SOL、`profile`、`check` 和 `disassemble` 可重复使用 `--oss-file DEST=LOCALPATH` 与
`--oss-output PATH`。客户端与 worker 直连 OSS，字节不过网关 JSON。
输出只在 `artifacts[]` 给临时 URL，不自动下载；不支持目录或通配符，可先打包普通文件。
上传前预留最终 `job_id`，上传或提交失败也打印该 ID，便于按同一前缀清理。
附件不得覆盖经网关校验的候选、solution 或 driver 文件；冲突在排队前拒绝。
`dev` 不支持这些附件参数，它使用 `--file`。

`oss_unavailable` 表示获准资源没有 OSS；`400 kind_not_allowed` 表示所有匹配资源都禁止该命令，
应更换卡型或命令，不能按 OSS 故障重试。上传准备和提交均遵循 AK、命令和放置约束。

## 错误归因与证据

- `code/*` 改候选，`validation/*` 改请求；`infra/*` 与 `timeout/*` 结合已执行证据和剩余预算判断。
  特别是 dev，不要因基础设施分类就重复执行有外部副作用的命令。
- Triton OutOfResources、ptxas 编译错误按候选归因，`details.failure_rule` 表示命中的证据。
  `ptxas-blackwell` 等回 `code/codegen_error`，优先保留具体汇编诊断。
- atrex-bench 环境／版本错误的 `next step:` 区分请求 pin 与镜像配置问题；客户端过旧时执行 `agate update`。
  包索引超时可返回 `infra/deps_install_timeout`。
- 没有独立结果或结构化终态证据，且日志不可达时回 `infra/logs_unavailable`；
  已持久化的原生失败保留原原因，并附 `details.output_unavailable_reason=logs_unavailable`。
  日志可读但无法识别根因才回 `infra/no_result`；诊断保留远端记录时有 `details.gc_deferred=true`。
- profile 采集前的候选异常回 `code/code_execution_failed`；成功结果的 note 按真实换行打印到 stderr，stdout JSON 不变。
  `eval_result_diagnostic=unsupported_verdict_layout` 只表示结构未知，不能据此判断代码失败。
- SOL 在首条 trace 前的 nvcc／TileLang 编译错误回 `code/code_execution_failed`，
  `failure_rule=sol_native_compile_error`；`details.execution_diagnostic` 保存有界异常摘要、退出码与字节数。
  `scan_complete=false` 表示证据不完整，字段缺省可能是旧 worker；有效 trace 优先。
  `details.result_payload_state` 区分空、不可解析、标记缺失与截断。
- SOL 的明确候选段错误回 `code/code_execution_failed`，保留 `sol_native_crash` 与 `segmentation_fault`，
  不自动重试或计集群故障；原生 abort 仍为 `infra/native_crash`、`failure_origin=unknown`。
  仅有 137／139 退出码不能证明 OOM 或候选错误，无其他证据仍为 `no_result`。
