# 进阶评测

基本候选和算子包见 [reference-dir.md](reference-dir.md)，SOL 格式与题目契约见
[solution-json.md](solution-json.md) 和 [custom-sol-problem.md](custom-sol-problem.md)。

## shape 并行

```bash
# 自动模式按现有配额与准入确定 worker 预算
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
# 限制并发单卡 worker 数，1 强制串行
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --shape-shards 2
```

仅适用于 atrex eval，默认开启，无需新增 Diamond 配置。镜像中的 bench 须经能力核验；
显式 `--atrex-bench-version` 在作业隔离安装后探测实际能力，支持则并行，过旧、未知或未验证则串行回退。
已知仅串行的版本可指定 `--shape-shards 1`，避免回退后仍持有多卡预算。
先升级服务端再使用新字段；旧服务端使用兼容客户端，不更换幂等键重跑在途作业。

获准 worker 数与 shape 分片数分开，每片最多 4 个 shape，其余分片在 Ray 排队；
原始 shape ID、输入 seed 和评测配置保持，最终合为同一个作业结果。显式预算超限拒绝；
worker 终态证据上限为 900，自动模式缩小预算但保留全部 shape，显式超限拒绝。
作业账本峰值卡数显示获准 GPU 总预算，CPU driver 不代表作业只用 0 卡。

`eval_execution` 回显请求／实际模式、获准预算和回退原因；隔离安装探测期间为 `pending`，
终态以实际执行为准。显式并行回退时 CLI 提示，预算保留至清理完成，幂等重放沿用原决策。
无法识别的历史诊断可为 null，不据此重新提交。

配套 bench 默认只重试 worker／节点故障分片，最多 3 次，即每片最多执行 4 次；
成功分片复用，代码错误不重试，分布式终态失败不重放整批，driver／结果缓存丢失则明确失败。
v1 能力探测不替代镜像／wheel 的分片恢复验收；调度与分片重试控制由网关和 bench 管理，不能用 `--set` 覆盖。
Ray Data 不覆盖共享 GPU 池容量保底；无法安全隔离容量控制的环境在评测前明确失败。

回滚到原版 0.14.19 须先停止入口并独立确认全部并行执行已停止，再由原版收尾并恢复串行；
这项评测回滚不包含 CDC 目标交接，已经切换 CDC 的环境还须单独满足观测回退条件。

## ABBA 基线比较

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm \
  --baseline baseline.py --abba-repeats 2 --job-timeout 1800
```

在同一次 GPU 分配内，每组按 A→B→B→A 完整评测。基线和候选均暴露 `class Model`，
共享 reference、依赖与评测器版本；仅支持 atrex + full，不能组合 `correctness_only`。
`--abba-repeats` 是完整组数，范围 1–8，默认 1；`--job-timeout` 覆盖整个作业。

- 每轮记录在 `result.abba.sdk_results[].abba.runs`；只有 `result.abba.valid=true` 才有有效的全 shape 加速结论。
- `speedup_geomean > 1` 表示候选更快；任一轮正确性失败或有效计时缺失，整体加速比为 null。
- `shapes.<id>.paired` 给出各组 AB／BA 延迟差、顺序平衡后的平均差及顺序效应估计，单位 ms，负差表示候选更快。
- 至少两组才有 `corrected_delta_se_ms`，一组时为 null，无效 shape 的 `paired` 为 null。
  标准误以完整组为单位，假设组间误差独立；顺序平衡不识别实例化 slot bias，也不保证消除时间漂移。

这是完整评测轮次的 ABBA，不是逐次 `forward` 交替计时，也不提供统计显著性或自动晋级判断。
同一份源码同时作 baseline 和 candidate 可做轮次级 A/A。

## 多文件源码

```bash
agate eval --gpu L20D --reference-dir ./data/op \
  --candidate candidate.tar.gz --candidate-entry src/model.py \
  --baseline baseline.zip --baseline-entry pkg/entry.py \
  --abba-repeats 2 --job-timeout 1800
```

两侧可各用 ZIP／TAR 包，也可与单文件混用；普通 eval 省略 baseline 参数即可。
每棵树最多 2000 个成员、解包后 4 MiB，成员须为 UTF-8 文本，包内 Python 入口导出 `Model`。
两份包使用同一批 OSS 直传，worker 解包后保留独立完整源码树。
HTTP／SDK 用 `{archive, entry_point}` 引用同批 `oss_files`；路径不得越界，不接受链接文件。
无法打开源码包返回 `infra/source_archive_unavailable`；已存在的空包或坏包返回
`validation/source_archive_invalid`，后者不重试。

## runner config 与评测器版本

```bash
agate eval --gpu L20N --config eval.json
agate eval --gpu L20N --candidate my_kernel.py --reference-dir data/rms_norm \
  --set validation_mode=correctness_only --set benchmark_mode=cuda_graph_replay
agate eval --gpu L20D --config abba.json --abba-repeats 2 --job-timeout 1800
```

JSON config 的 `input`／`reference_dir` 按配置文件所在目录解析；装了 PyYAML 也可用 YAML。
ABBA config 中的 `baseline_input` 同样按该目录解析并内联，显式 `--baseline` 优先。
其他 runner 参数转给 worker 上的 atrex-bench 校验，拼错会收到 `unsupported runner config key`。
`output`／`checkpoint_dir` 由网关决定，配置文件中的值会被忽略。

atrex eval／profile 当前服务端默认 wheel 为 `0.1.0.post20260917160311`，请求／集群 pin 优先；
可用 `--atrex-bench-version VERSION` 显式指定。未显式 pin 时，仅复用与 cluster／默认 pin 完全相同的镜像 wheel。
用 `agate get JOB_ID --spec --json` 的 `request_digest.atrex_bench_runtime` 核对版本来源、
镜像版本和是否发生作业级安装，不要把默认行为描述成每次解析索引最新版。

## 放置与测量可信度

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm \
  --cluster-label stage=preview --node-label ray.io/market-type=spot
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --lock-clocks
```

cluster 标签匹配配置，node 标签由后端原生调度；硬约束无匹配不会回退，OSS 上传引用绑定同一放置条件。
`JobView.placement` 回显显式请求；GPU 环境别名读取只代表选中成员的缓存，不保证整池镜像一致。

锁频只在驱动可见单卡时生效，共享多卡环境跳过；默认锁到最大频率，也可由集群配置覆盖。
GPU 测试结束后立即解锁，提前失败或 SIGTERM 由退出钩子恢复，SIGKILL 遗留锁由后续作业先重置。
NCU 始终使用 `--clock-control none`，由 Agate 的 `--lock-clocks` 统一管理。
结合结果 `environment.clock_sm_pct_of_max`／`clock_throttle_reasons` 和各 shape 的
`timing_stats.cv` 判断可信度。`cv=null` 表示只有一个样本，不能判断离散度；`runs` 是实际样本数。
有 `roofline.json` 才能计算相对自带 roofline 的 `sol.pct`，无法计算时为 null 并给出 `sol.reason`。

## profile 与 SOL 提示

`profile` 默认 `--level sol`；`survey` 找热点，`deep --kernel-regex REGEX` 返回原始报告链接。
PPU 按 vendor 自动选择 ACU，显式 `--profiler ncu` 返回 400；使用 `acu` 或省略 profiler。
NCU 可选指标不可访问时保留已有 kernel，空捕获保留工具退出码和诊断，不能据此断言候选异常。
多卡通信算子、NCU 2026.1+ 的 rank 顺序和 context 范围见 [collectives.md](collectives.md)。

SOL 默认不测参考实现，需 `--benchmark-reference` 才有可解释的 speedup，代价约两倍运行时间；
默认 speedup 为 0 不表示没有加速。自定义题的通过只证明与自带 reference 一致，
应同时核对通过数量、`problem_source=inline` 和 `problem_digest`，不能混入官方题统计。
