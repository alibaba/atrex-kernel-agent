---
name: atrex-gpu-gateway
version: 0.15.0
description: 使用 agate CLI 向 atrex-gpu-gateway 提交 GPU 算子评测、编译检查、性能分析、反汇编与受控研发命令，查询环境和异步作业结果。适用于 GPU 作业执行、CUDA solution.json 配置及 AKA 候选文件管理和排障；网关部署与容量管理按需读取运维参考。
---

# atrex-gpu-gateway

无状态 GPU 网关：通过 HTTP 把作业路由到 GPU 节点执行，统一按 `job_id` 查询和取消。
本机无需 GPU，也无需安装 Ray 或 profiler。详细参数以 `agate <命令> --help` 为准。

## AKA 中使用

在 AKA 管理的工作区中，按已选定的文件管理模式组织候选，框架类型不改变文件格式。
先读 [AKA 文件组织与调用](reference/aka-workspace.md)，再按需读
[manifest 字段、CUDA ABI 和仓库依赖](reference/eval-manifest.md)。
作业通过工作区的 `tools/sandbox.py` 提交；Supervisor 负责环境、鉴权、上传和记录。
本文件后续的直接 agate CLI 安装／配置命令适用于独立使用，不用于绕过 AKA Runtime。

## 先选命令

| 目的 | 命令 | 输入与边界 |
|---|---|---|
| 只检查编译与诊断 | `agate check kernel.py --gpu GPU` | 单文件 `class Model`；对应 `compile` 作业，不做完整正确性或测时 |
| 验证正确性、比较性能 | `agate eval --gpu GPU --candidate kernel.py --reference-dir DIR` | 自带参考实现、输入生成器和 shape |
| 声明多文件候选及依赖 | `agate eval --gpu GPU --manifest solution.json --reference-dir DIR` | atrex manifest v1；支持按版本准备 Git 仓库依赖，CUDA 入口由 worker 编译并生成绑定，无需 Python 候选；见[声明式输入](reference/eval-manifest.md) |
| 使用 SOL-ExecBench 题目评测 | `agate eval --backend sol --gpu GPU --solution solution.json` | 镜像题库或调用方自定义题 |
| 找热点与性能瓶颈 | `agate profile --gpu GPU --candidate kernel.py --reference-dir DIR` | `survey` 找热点，`sol` 看计算／访存瓶颈，`deep` 深挖 kernel |
| 看 GPU 汇编 | `agate disassemble --gpu GPU --candidate kernel.py --fmt auto` | 返回 PTX／SASS／ISA；与编译检查共享导入、构造路径 |
| 查 GPU 与工具链 | `agate env`、`agate env GPU --frameworks` | 读取环境，无需提交研发命令 |
| 尚无一等接口的研发动作 | `agate dev 'COMMAND' --gpu GPU` | 特权 raw shell，默认关闭；须有专用 AK 权限 |

`--backend atrex|sol` 选择评测 harness，执行后端由网关配置选择。
`--profile prod|pre` 选择网关环境，`profile --level sol` 的 `sol` 表示性能上限分析。
新命令用 `eval`／`env`；旧别名 `test`／`run`／`sol`／`list hw` 仍兼容。

## 安装与鉴权

```bash
pip install -U atrex-gateway-client \
  --index-url http://artlab.alibaba-inc.com/1/pypi/pypi-releases \
  --trusted-host artlab.alibaba-inc.com
export AGATE_AK='<ak>'
export AGATE_SK='<sk>'
agate config
agate env
```

默认使用正式环境 `https://atrex-gateway.alibaba-inc.com`，不必额外设置 URL。
也可将 `ak`／`sk` 写入 `~/.atrex/config.json`。CLI 使用 AK／SK；浏览器集团登录不能替代签名，
不要配置服务端尚不接受的 `AGATE_TOKEN`。`author` 和自报工号不授予权限。
仅在用户明确要求时切换预发；本文件示例使用正式环境。

安装后可用 `agate update --check` 检查新版，`agate update` 更新；没有该子命令的旧版用上面的 pip 命令升级。

## 从编译到评测

```bash
# 先检查导入与构造阶段能触发的编译
agate check my_kernel.py --gpu L20D
# Model 构造函数有必需参数时提供 JSON object
agate check my_kernel.py --gpu L20D --init-kwargs init_kwargs.json

# 再用真实输入验证正确性和性能
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
# 只验证正确性
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --mode correctness_only

# 需要性能分析时按层深入
agate profile --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --level survey
agate profile --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --level deep --kernel-regex rms
```

`check` 对应 `POST /v1/jobs/compile`，CLI 没有 `agate compile` 子命令。
它导入候选、执行 `Model(**init_kwargs)`，不自动调用 `forward`。Triton 等首次 launch 才编译的
JIT 实现可能只有导入／构造诊断，`ptxas` 为空；`ok=true` 不证明这些 kernel 已编译或运行正确。
要触发真实输入路径请用 `eval` 或 `profile`。结果字段、`--arch` 与 sanitizer 的边界见
[编译检查](reference/compile.md)。

Python 候选和参考实现需暴露 `class Model`；原生 CUDA 候选由 worker 生成该适配层。准备算子包时先读
[reference 目录格式](reference/reference-dir.md)，准备 SOL 候选时读
[solution.json 格式](reference/solution-json.md)；自定义 SOL 题还需读
[自定义题与最小验收](reference/custom-sol-problem.md)。这些材料和示例均随 skill 分发。

atrex eval 的 shape 并行默认开启，按现有配额和准入确定 worker 预算；显式 `--shape-shards 1` 强制串行。
旧版或未验证的 bench 保留串行回退；并行参数、分片恢复和回退边界见[进阶评测](reference/evaluation.md#shape-并行)。

### input.py 大小限制

`eval`／`profile` 的输入生成器 `input.py` 默认上限是 **16 KiB＝16384 个 UTF-8 字节**，
按源码字节数计算，包含注释与空白；不是字符数，也不是生成的张量大小。
部署方可通过 `max_input_source_bytes` 调整，重启后生效；实际额度以
`GET /v1/env/{gpu}/capabilities` 的 `limits.max_input_source_bytes` 为准。
超限在排队前返回 `400 invalid_source`，错误包含 `reference.input_py` 的实际字节数和上限。
候选与 `reference.py` 使用独立额度，默认每份 128 KiB，见 `limits.max_source_bytes`。
`check` 不接收 reference 目录或 `input.py`，这项 16 KiB 默认额度不适用于编译候选。

## 提交、查询与失败处理

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --no-wait
agate get <job_id> --wait
agate jobs --status running
agate cancel <job_id>
```

`eval` 默认等到终态，成功退出 0、失败非 0；`check` 默认等待，使用 `--no-wait` 可只提交。
取消受理不等于远端已停止，继续查询原 `job_id`。
作业阶段区分网关排队、后端提交／排队、执行与收集；等待 GPU 扩容时保持原作业身份。

- 满额或暂时过载时遵守 `Retry-After`，沿用原请求与稳定幂等键；SDK 的自动重试也受调用预算限制。
- 提交结果未知、连接中断或存储暂不可用时，保留原 `job_id`／幂等键确认结果，避免重复执行。
- `403 cluster_ak_denied` 表示当前 AK 无匹配集群权限，联系管理员或选择获准环境。
- `code/*` 优先改候选，`validation/*` 改请求；`infra/*` 与 `timeout/*` 结合错误证据、预算处理。

超时分三层：`--job-timeout` 限制作业执行，`--wait-timeout` 限制 CLI 等待终态，
`--http-timeout` 限制单次 HTTP；并非每个子命令都支持全部参数，先查对应帮助。
完整重试、错误归因、附件与依赖说明见[作业与排障](reference/jobs-and-troubleshooting.md)。

## 按需阅读

| 场景 | 参考 |
|---|---|
| 编译结果、JIT、构造参数、架构或 sanitizer | [编译检查](reference/compile.md) |
| shape 并行、ABBA、源码包、runner config、评测器版本与锁频 | [进阶评测](reference/evaluation.md) |
| CUDA 原生源码、JSON 编译与依赖声明 | [manifest v1](reference/eval-manifest.md)、[AKA 工作区](reference/aka-workspace.md) |
| SOL 内置题与自定义题 | [solution.json](reference/solution-json.md)、[自定义题](reference/custom-sol-problem.md) |
| 多卡通信算子与 NCU | [通信算子](reference/collectives.md) |
| OSS 大文件、依赖、异步状态、重试或错误诊断 | [作业与排障](reference/jobs-and-troubleshooting.md) |
| raw dev 权限、自部署、观测、迁移与 preview 能力 | [运维边界](reference/operations.md) |
| Ray 扩缩容、GPU 保底、暂停或跨环境交接 | [容量控制](reference/capacity-control.md) |

手动 CLI、SDK 与 HTTP API 文档分别在网关 `/sdk/cli/`、`/sdk/`、`/docs/api`。
`/docs/api` 提供静态字段和 JSON 示例；Swagger／ReDoc 与 `/openapi.json` 不开放。
