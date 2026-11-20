# 本地 AKA Light Dashboard

## 例子：查看已经在本机运行的 Vector Add

用户：`给 runs/vector-add 做一个 Dashboard，看看性能、token 和调用错误。`

复用现有 AKA checkout、Python 环境和 Campaign。以下路径仅为示例，需要替换为实际路径：

```bash
AKA_REPO=/path/to/atrex-kernel-agent
AKA_PYTHON=/path/to/aka-venv/bin/python
CAMPAIGN=/path/to/runs/vector-add

test -f "$CAMPAIGN/campaign.json"
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" --help
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --port 0
```

终端打印实际地址，例如 `http://127.0.0.1:49152/`。打开该地址查看页面；不要把示例端口当成实际端口。
默认每 5 秒刷新。数据由可观测性工具直接读取本地 Campaign，不调用 Agate，也不重新测量 Kernel。
这里不需要重新询问 Agate AK/SK、卡型、模式、DSL 或 Atrex-Bench 路径。

若已知一个父目录但不清楚 Campaign 路径，可在该目录有限范围内查找 `campaign.json`；不要扫描整个用户主目录。
每个包含该文件的目录是一个独立 Campaign，不能把父目录或 Agent 工作区当成 Campaign。

## 后台启动与验证

使用新的输出目录保存 Dashboard 进程的日志和 PID，放在 Campaign 之外：

```bash
DASHBOARD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aka-light-dashboard.XXXXXX")
nohup "$AKA_PYTHON" -u "$AKA_REPO/observability/light_dashboard.py" \
  "$CAMPAIGN" --port 0 \
  > "$DASHBOARD_DIR/server.log" 2>&1 < /dev/null &
dashboard_pid=$!
printf '%s\n' "$dashboard_pid" > "$DASHBOARD_DIR/server.pid"
```

读取 `server.log` 中打印的 URL。确认进程未退出；若启动失败，读取原始错误并修正路径、环境或端口。
不要仅凭 PID 文件声称已启动。

对实际地址检查：

```bash
# 替换成 server.log 中的真实地址。
DASHBOARD_URL=http://127.0.0.1:49152
curl --fail --silent --show-error "$DASHBOARD_URL/" > /dev/null
curl --fail --silent --show-error "$DASHBOARD_URL/api/summary" \
  > "$DASHBOARD_DIR/verified-summary.json"
```

只读取摘要里的 `workspace`、`captured_at`、`source_mode` 和需要的统计字段，不把整份 JSON 打印进对话。
真实采集耗时取决于 Campaign 大小；请求未返回时不要当成空数据。没有任何 Evaluate 的新 Campaign
可以显示空结果；应明确“尚无结果”，不生成示例性能点。

完成后提供 URL、Campaign 路径、实时/快照模式；后台启动时附日志位置。停止 Dashboard 只影响它自己的
HTTP 进程，不应停止 Campaign。只操作本次启动并确认身份的进程。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| 位置参数 `source` | 必填 | 本地 Campaign 根目录，或已采集的 Dashboard JSON |
| `--host` | `127.0.0.1` | 仅允许本机回环地址 |
| `--port` | `8765` | HTTP 端口；`0` 自动分配可用端口 |
| `--refresh-ms` | `5000` | 正整数，浏览器刷新间隔，单位毫秒 |
| `--capture PATH` | 不设置 | 采集一次写入 JSON 后退出，不启动 HTTP 服务 |

## 保存某一时刻的数据

需要留档或离线查看时，直接从本地 Campaign 采集：

```bash
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" \
  --capture "$DASHBOARD_DIR/snapshot.json"
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" \
  "$DASHBOARD_DIR/snapshot.json" --port 0
```

使用新的目标文件；当前 CLI 会覆盖同名文件，因此未经用户要求不要覆盖已有留档。
这是 `captured_observability` 快照，状态只代表 `captured_at`。页面刷新仍读取同一份快照；要更新数据需重新采集。
当前捕获最多保留最近 500 条 Runtime 调用和最近 500 条 Agate 调用及关联详情，汇总统计仍是 Campaign 范围。
Kernel/Research 列表也遵循各工具的默认范围，不能声称包含所有历史。快照可能包含私有源码、参数和响应，
按本地审计数据保管。

## 按需展开详情

页面会读取这些 GET 接口，也可用于验证具体视图：

| 接口 | 内容 |
|---|---|
| `/api/summary` | 整体快照及采集时间 |
| `/api/lineages` | Lineage、当前工作、Direction/Experiment、token |
| `/api/kernels` | Full Evaluate 与对应 Submission 历史 |
| `/api/winners` | Accepted submission 的最优结果 |
| `/api/runtime-calls?kind=evaluate&status=failed&last=10` | 最近失败的 Evaluate Runtime 调用 |
| `/api/agate-calls?status=pending&last=10` | 最近尚未终态的 Agate 调用 |
| `/api/gateway-records/gateway-…` | 完整 Gateway ID 对应的证据 |
| `/api/runtime-requests/…` | 完整 Runtime request ID 对应的审计 |

调用筛选可用 `kind`、`status`、`lineage`、`last`；状态为 `succeeded/failed/pending/unknown`，
`last` 为 1–500。筛选只改变明细，调用汇总仍是整个 Campaign。快照模式只能筛选已经捕获的明细；
请求未被捕获的详情会返回 404，不代表原始 Campaign 丢失了证据。

多 Campaign 默认分别启动页面并返回链接。若要求合并页面，继续使用同一套可观测性快照作为数据输入，
保留每个 Campaign 的标识和采集时间；不要重新解析日志、合并不同算子的 Shape 或重新计算 token 口径。
