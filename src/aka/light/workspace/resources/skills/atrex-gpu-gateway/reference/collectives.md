# 单机多卡通信算子

需要客户端和服务端均为 0.13.6 或以上。`eval`、`profile`、`dev` 都接受
`--num-gpus N`，默认 `1`，范围为 `1`～`64`。它声明同一节点上的 GPU 分配数量，
不会把普通 eval 或 dev 命令自动复制 N 次。普通 eval 的 atrex／SOL harness 语义保持不变。
当前只有具备原子分配能力的 Ray 环境支持 N 大于 1；固定 pod 和 local 后端明确拒绝。
通过 `agate env` 选择目标别名，并请管理员确认单节点卡数和镜像依赖。

`eval/profile --collective` 中的 `--collective` 是显式的作业契约切换：CLI 会提交
`kind=collective` 到 `/v1/jobs/collective`，候选、参数和结果结构都不再是普通 atrex／SOL
评测。保留 eval／profile 动作词是为了让“正确性与计时”和“独立 nsys 采样”保持统一入口；
它不是普通 eval 的附加算子选项，不能与 `--backend`、`--reference-dir` 等 harness 参数混用。

## 最小运行

从本 skill 目录运行。`GPU` 替换为有多卡资源的环境别名；示例随 skill 分发，不需要仓库权限。

```bash
export GPU=REPLACE_MULTI_GPU_ALIAS
agate eval --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --numel 1048576 --numel 16777216
agate eval --gpu "$GPU" --num-gpus 4 --collective allgather \
  --candidate scripts/collectives/allgather.py --numel 1048576
agate eval --gpu "$GPU" --num-gpus 4 --collective reduce-scatter \
  --candidate scripts/collectives/reduce_scatter.py --numel 1048576
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --numel 1048576
```

这三个候选都直接调用 NCCL，只用于验证契约和运行环境，**不是优化实现，也不承诺加速**。
复制候选文件，将 `Model.forward(tensor, group)` 换成自定义通信实现即可与同配置 NCCL 比较。
不传 `--reference-dir` 或 `--backend`。默认 float32、5 次预热、20 次计时；可用
`--dtype float16|bfloat16`、`--warmup-iters`、`--bench-iters`、`--job-timeout` 调整。
三个不同随机输入在所有 rank 上校验；正确性失败的 shape 不返回性能数字，CLI 退出码为 1。
默认数值容差为 `atol=rtol=0.01`，需要其他阈值时通过 HTTP 请求的 `options` 设置。

## 候选契约

候选是单个 Python 文件，定义 `class Model(torch.nn.Module)`，不需要构造参数。
runner 每张卡启动一个 rank，负责初始化 NCCL 进程组、设置当前设备、启动与清理进程。
候选接受一维连续 CUDA Tensor 和已经初始化的 `group`，返回同设备、同 dtype 的 Tensor。
可原地修改输入；不得初始化／销毁 runner 的进程组，也不要自行设置 GPU 可见性。
自定义库可通过 `-r requirements.txt` 安装；仍需通过源码校验。

`--numel` 指完整逻辑 buffer 的元素数，可重复，最多 32 个不同 shape：

| 算子 | 每个 rank 的输入 | 每个 rank 的输出 | 语义 |
|---|---|---|---|
| `allreduce` | numel | numel | SUM |
| `allgather` | numel / N | numel | 按 rank 顺序拼接 |
| `reduce-scatter` | numel | numel / N | SUM 后按 rank 连续切分 |

后两种要求 numel 能被 N 整除。N 等于 1 时没有跨卡通信，只适合烟测，不能据此判断通信收益。

## 如何读结果、比较卡数

`result.measurements` 按 shape 返回 `passed`、`baseline`、`candidate` 和 `speedup`。
每轮先同步，计时覆盖通信调用及完成同步，取最慢 rank 的耗时；`latency_us` 为多轮中位数，
另有 `min_us`、`p95_us` 和 `samples`。这是同步的 host 端端到端延迟，含 Python／分配开销，
不是纯 kernel 时间，也不等价于 nccl-tests 的 CUDA-event 数字。

`algorithmic_gbps` 和 `bus_gbps` 单位均为十进制 **GB/s**。算法带宽为完整 buffer 字节数除以
延迟；总线修正系数为 allreduce 的 `2*(N-1)/N`，以及另外两种的 `(N-1)/N`。
定义采用 [NCCL tests 的带宽约定](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。
`speedup` 是**同卡数、同 shape 下 NCCL 延迟／候选延迟**，不是 N 卡相对单卡的加速比。

保持 dtype、逻辑 numel、候选和镜像版本不变，分别测 2／4／8 卡，并保留每次 job_id：

```bash
for n in 2 4 8; do
  agate eval --gpu "$GPU" --num-gpus "$n" --collective allreduce \
    --candidate scripts/collectives/allreduce.py --numel 16777216 \
    --dtype float32 --warmup-iters 10 --bench-iters 50 > "allreduce-${n}gpu.json"
done
```

先确认每个 shape 的正确性，再比较延迟和总线带宽；多卡通信本身不保证随卡数增加而变快。
重复实验并记录环境、拓扑和竞争负载。当前离线测试不构成 GPU 性能证据，以上命令需在目标环境执行。

## Profile 与原始报告

collective profile 先完成无 profiler 的正确性和计时，再独立运行 profiler，默认 Nsight Systems。
`profiled_timings=false` 表示性能数字来自前一次无 profiler 执行。
nsys 的 `result.profile.kernels` 是所有 rank、基线和候选的汇总表，包含校验／预热；不能用它的总耗时
直接计算候选收益。原始时间线里，正式采样有 `collective/<算子>/baseline` 和
`collective/<算子>/candidate` NVTX 区间，每个 shape 最多追踪 5 轮。

如果目标环境配置了 OSS，可以保留完整时间线：

```bash
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --oss-output collective.nsys-rep
```

NCU 采样使用单机 shmem communicator，要求 worker 安装 **NCU 2026.1 或更新版本**，
支持 `--communicator shmem` 和 `--communicator-shmem-num-peers`，并具备兼容驱动、
profiling 权限和各 rank 可访问的共享内存。开始前会检查实际 NCU 可执行文件，旧版返回
`collective_ncu_shmem_unsupported`，缺少 NCU 返回 `collective_ncu_unavailable`。

```bash
# 随包示例调用 NCCL，因此这里匹配 nccl；自定义通信 kernel 要换成自己的名字。
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --profiler ncu --level deep \
  --kernel-regex 'nccl.*' --oss-output collective.ncu-rep
```

`--profiler ncu` 默认 `--level sol`，采 basic 指标；deep 采 full 指标。必须二选一提供
`--kernel-name` 或 `--kernel-regex`，且选中的通信 kernel 必须由**全部 rank 按相同顺序和次数**
调用。第一版不支持 rank 间启动序列不同的非对称通信。候选执行时才开启采样，正确性、预热、
基线、输入生成和 rank 间 barrier 都在采样范围外。性能对比仍来自独立的无 profiler 执行。

`--counter` 可重复指定，覆盖 basic／full 指标集。`--launch-skip` 默认 0，上限 10000；
`--launch-count` 默认 1，上限 100；`--top-kernels` 默认 30，上限 100。
launch 计数针对匹配的 kernel，不等于候选调用次数或 shape 数；需要某个 shape 的指标时，
每次只传一个 `--numel`。shmem 默认采每组第一个 CUDA context，其他 rank 配合执行，
**不保证采到 rank 0，也不代表返回所有 GPU 的指标**。
`result.profile.kernels[].identity` 保留报告中的 process_id、device、context、stream 和
launch_id，同名 kernel 不合并；缺少的标识为 null。原始 `.ncu-rep` 可通过 OSS 输出附件下载。
每个 shape 的 NCU trace 调用轮数为 `min(bench_iters, launch_skip + launch_count)`，
不会超过 `--bench-iters`（上限 1000）。一次候选调用可能匹配多个或零个 kernel，因此不保证采满
`--launch-count`；空采样时请检查选择器、skip／count 和候选实际 launch 序列。
启用此功能须先升级网关至 0.13.17 或以上，再升级客户端；旧网关可能忽略新参数而执行 nsys。
`eval --collective` 不接受 profiling 参数，请使用 `profile --collective`。
`--source` 暂不对 collective 开放。

普通多卡 profile 也只允许 `--level survey`（NVIDIA nsys）；多卡锁频暂不支持。
缺少 torch／NCCL／nsys、设备可见数量不符、超时或 profile 失败都会返回明确错误，不会伪造成功报告。
