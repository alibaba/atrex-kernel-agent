# 编译检查：agate check 与 compile 作业

适合快速查看导入、构造阶段的编译诊断；需要正确性或性能结论时用 `eval`，需要热点分析时用 `profile`。

## 入口与输入

```bash
agate check my_kernel.py --gpu L20D
agate check my_kernel.py --gpu L20D --init-kwargs init_kwargs.json
agate check my_kernel.py --gpu L20D -r requirements.txt --no-wait
agate get <job_id> --wait
```

CLI 子命令是 `check`，作业类型是 `compile`，HTTP 入口是 `POST /v1/jobs/compile`。
候选是暴露顶层 `class Model` 的单个 Python 文件。`--gpu` 必填，在目标 worker 环境运行，
不是本地静态检查。源码的 AST 校验在提交时执行，拒绝请求直接返回 400，不会先排 GPU。

`init_kwargs.json` 必须是 JSON object，默认 `{}`，紧凑编码后最多 64 KiB。例如：

```json
{"hidden_size": 1024, "eps": 0.000001}
```

它传入 `Model(**init_kwargs)`，不会从 `shapes.json` 自动读取，也不作为 `forward` 输入。
这些参数是普通作业输入，不要包含密码或 token。
`check` 不读取 `reference.py`、`input.py` 或 `shapes.json`；候选大小使用独立的源码额度。
额外依赖可用 `-r`，默认 `freeze_installed` 复用 base 已满足的包。

## 实际执行范围

worker 导入候选，再构造 `Model(**init_kwargs)`，收集该作业隔离缓存中的编译产物。
构造阶段触发的 AOT 编译会执行；驱动不会自动调用 `forward`，也不会生成测试输入。

因此，Triton 等首次 launch 才 JIT 编译的实现可能只完成导入与构造，没有 kernel 产物。
`ok=true` 此时只说明已执行的路径没有诊断错误，不代表所有 kernel 编译成功、正确性通过或性能合格。
需要实际输入时用 `eval`／`profile`；`disassemble` 共享导入、构造方式，也不会自动补出 JIT 输入。

## 结果怎么读

在作业 `result` 中同时检查这些字段：

| 字段 | 含义 |
|---|---|
| `ok` | 导入／构造路径是否无诊断错误；不是评测通过标志 |
| `diagnostics[]` | 有界的 `file`／`line`／`severity`／`message` 编译或导入诊断 |
| `ptxas` | 产物存在时的 `regs`／`spills`／`smem` 等统计；为空表示没有可用统计，不是零资源消耗 |
| `asm_available`、`instruction_histogram` | 是否找到汇编，以及其中的指令计数；不能替代耗时或吞吐测量 |
| `notes[]` | JIT 未触发、工具不可用、架构覆盖范围等诊断边界 |

多个 Triton kernel 有统计时，`ptxas` 展示 spill 优先、寄存器数次之的最重 kernel，
并用 `kernels_compiled` 返回扫描到的数量；不是所有 kernel 的平均值。
驱动无可解析结果时会返回 `ok=false` 和诊断，不应解读为编译成功。

## 架构与 sanitizer

```bash
agate check my_kernel.py --gpu H20 --arch sm_90
agate check my_kernel.py --gpu H20 --sanitize memcheck
```

`--arch` 映射到工具链可接受的 `TRITON_OVERRIDE_ARCH`／`TORCH_CUDA_ARCH_LIST`。
并非所有框架都读取它们，例如 CuteDSL 需在自己的编译参数中选架构。
结合 `arch_requested`、`arch_enforced_by` 与 `notes` 判断实际覆盖；没有触发 JIT 就没有对应编译证据。

`--sanitize` 可选 `memcheck`／`racecheck`／`initcheck`／`synccheck`，使用目标 worker 的厂商检查工具，
执行预算更长，但仍不会自动调用 `forward`。只有候选自身在导入／构造路径触发执行，工具才可能检查到 kernel。
必须检查 `sanitize.checked` 和 `sanitize.note`；`checked=false` 表示没有检查对象，不能当成内存或竞争检查通过。
