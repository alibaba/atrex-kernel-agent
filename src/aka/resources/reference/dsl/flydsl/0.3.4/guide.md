# FlyDSL 0.3.4 Programming Guide

Applicability: framework FlyDSL; version 0.3.4; AMD ROCm.
Read the relevant sections when implementing a framework baseline, optimizing an Episode,
or repairing an import, compilation, Context, layout or numerical failure.
This packaged reference is shared by Legacy and Light.

本指南按固定版本 DSL、Compiler、Kernels 三份官方资料整理。API 存在性核对使用
官方 Linux x86_64 CPython 3.12 wheel；MI308X／gfx942 是本文示例的目标之一。
安装版本由 Task 声明，本文不为任何 Campaign 自动选择或安装依赖。
源码示例、可导入 wheel API 和架构专用指令的范围分别说明。

| 来源 | 固定身份 |
| --- | --- |
| 官方源码 | ROCm/FlyDSL v0.3.4，commit `6bb37ba4e23c3a2729e3af244251b55fc5938237` |
| 核对的发行包 | `flydsl-0.3.4-cp312-cp312-manylinux_2_27_x86_64.whl` |
| wheel SHA-256 | `280d9fb8830c0ccfb65e1cad3a6bb57f8b9c8a74b5352dafeebd3b4041425ed1` |

Job 安装的依赖可能不同于基础镜像或 Agent 本地 Python。先从所选 runtime 的环境查询
读取 `task_worker_requirements`，再核对 Job 安装结果。版本未知或不匹配时记录缺失事实，
取得对应版本资料或在获准 worker 中检查实际接口，不默认套用 0.3.4。
本文示例仍需按 Task 的真实输入、数值容差和完整验证流程取得运行证据。

## 使用顺序

1. 核对 `task_worker_requirements` 中的实际 FlyDSL 版本；本指南对应 0.3.4。确认 wheel ABI、平台和 Job 的安装结果，换版本时重新核对接口。
2. 按第 3、4、5 节选择编译入口、参数和已安装表达式 API。
3. 需要控制流、LDS 或调优时分别读第 6、7、8 节；原始 MLIR 和仓库 kernels 有额外边界。所有 import、JIT、数值和性能实验通过所选 sandbox 执行。
4. 用实际输入调用 `Model.forward` 触发 JIT，按第 14 节确认数值结果；只有导入成功不能完成验收。

## 1. 项目概览与安装范围

FlyDSL 将 Python 中的显式 layout、tensor、copy 和 MMA 表达式追踪为 MLIR，
再由 ROCm 后端生成 GPU 代码。CuTe 风格的 shape／stride 代数用于组织线程、寄存器和内存视图。
MI308X 的目标架构为 gfx942，按 CDNA 路径编写；不能因某个 RDNA API 可导入就把它用于 MI308X。

| 范围 | 内容 | 当前 wheel 是否提供 |
| --- | --- | --- |
| `python/flydsl/compiler/` | JIT、kernel、参数适配、后端 | 是 |
| `python/flydsl/expr/` | layout、类型、GPU／ROCDL、copy／MMA | 是 |
| `python/flydsl/runtime/`、`utils/` | runtime、环境选项、旧 LDS 工具 | 是 |
| 构建生成的 `flydsl/_mlir/` | 嵌入式 MLIR bindings／库 | 是；原始 MLIR API 有独立稳定性边界 |
| 仓库根 `kernels/`、`examples/`、`tests/` | 参考 kernel、示例、上游测试 | 否 |

`kernels.common.buffer_ops` 在官方仓库中存在，但它不是当前 wheel 的可导入包。
阅读 kernels 源码可学习实现；要运行它，必须另外声明对应 revision 的源码、
导入路径和依赖。安装 `flydsl==0.3.4` 不等于安装整个官方仓库。[来源：S3、S5]

## 2. 编译流水线

首次调用的核心过程是：

```text
Python 调用 @flyc.jit
  → 参数适配与 specialization
  → AST 重写、追踪 host／kernel，构造 MLIR
  → Fly layout／atom／寄存器等 lowering
  → ROCDL 与 LLVM lowering
  → GPU binary 与可调用的 host artifact
  → 缓存并执行
```

0.3.4 的 ROCm 后端按 `RocmBackend._pipeline_parts()` 分成三个阶段：

| 阶段 | 作用 |
| --- | --- |
| `pre_binary_fragments` | signature rewrite、Fly canonicalize／layout lowering、寄存器 SSA、Fly → ROCDL |
| `binary_prep_fragments` | target attach、控制流／GPU／vector／arith／func → LLVM、cast reconciliation |
| `binary_fragment` | `gpu-module-to-binary` |

具体 pass 和 target 参数见固定版本 [Compiler API][compiler] 与
[ROCm backend 源码][rocm-backend]。旧指南“固定 14 个 pass”的表不作为 0.3.4 的实现契约。
MLIR 的 `vector` dialect 仍参与编译；它与已删除的 Python `flydsl.expr.vector` 包装模块不同。

## 3. 核心 API 与启动配置

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
```

| 入口 | 0.3.4 用法 |
| --- | --- |
| `@flyc.kernel` | 定义 GPU body；在 JIT 追踪中调用，并对返回的 launcher 调用 `.launch(...)` |
| `@flyc.jit` | 定义 host launcher；首次实际调用触发编译，匹配 specialization 后可复用 |
| `flyc.from_dlpack(tensor)` | 获得包含 layout 元数据的参数适配对象 |
| `flyc.from_torch_tensor(tensor)` | 当前 compiler 顶层提供的 PyTorch 适配入口 |
| `flyc.compile(...)` | 预编译接口；返回的 `CompiledFunction` 仅接受与原 JIT 函数对应的顺序位置参数 |

调用形式是 `kernel_fn(...).launch(grid=..., block=..., stream=...)`；
第 13 节给出完整例子，避免把省略 kernel body 的片段当成可运行候选。

- `grid`／`block` 支持静态整数、动态 DSL 整数／IR 值和 1–3 维 tuple／list；缺失维为 1。
- `smem=None` 时可自动推断 dynamic SharedAllocator 所需字节；显式值至少覆盖推断的使用量。
- `stream` 使用当前调用的 GPU stream；ROCm PyTorch 仍使用 `torch.cuda` 命名空间。
- 一般候选沿 Task／harness 给定的输入与 stream 运行；调优不能改变算子语义或输入合同。

历史内部接口 `CompilationContext` 的实际路径是
`flydsl.compiler.kernel_function.CompilationContext`，不是 compiler 顶层导出。
新 LDS 代码使用第 7 节的 allocator；需要旧内部接口时按固定版本核对，不能把它视为稳定公共 API。[S1、S2、S4]

## 4. 参数类型与自定义协议

| 类型／表达式 | 用途 |
| --- | --- |
| `fx.Tensor` | 追踪时的 GPU tensor 参数；host PyTorch tensor 经适配转换 |
| `fx.Constexpr[T]` | specialization 中的编译时值；不占 runtime ABI 参数槽 |
| `fx.Int32`、`fx.Int64` | 显式整数类型；offset／索引计算按范围选择，较大 offset 优先 Int64 |
| `fx.Float32`、`Float16`、`BFloat16` | DSL 数值类型；与原始 MLIR type 对象不同 |
| `fx.Stream` | host → JIT 的 stream 参数 |
| `flyc.from_dlpack(x).mark_layout_dynamic(...)` | 按支持的标记声明动态 layout；遵守真实 stride 与 alignment |

自定义参数通过 `JitArgumentRegistry.register(py_type, dsl_type=...)` 关联 host adapter 和 DSL 类型。
0.3.4 的协议位于 `flydsl.compiler.protocol`：

| 协议 | 方法 | 必须满足的关系 |
| --- | --- | --- |
| `JitArgument` | `__get_ir_types__`、`__cache_signature__`、`__c_abi_spec__` | IR 类型、缓存 specialization、lowering 后 C ABI 的顺序与类型相符 |
| `DslType` | `__extract_to_ir_values__`、`__construct_from_ir_values__` | 提取与重建按 SSA 数量、顺序和类型形成往返 |
| `Storable` | `__dsl_size_of__`、`__dsl_align_of__`、`__peek_from_ptr__`、`__poke_into_ptr__` | 字节布局、对齐与 typed memory 访问一致 |

旧指南使用的 `__fly_types__`／`__fly_ptrs__`／`__fly_values__`／`__fly_construct__`
不作为 0.3.4 协议模板。ABI fill 回调应读取每次调用的新实参，而不是捕获第一次调用的指针。
详细定义和完整边界见 [DSL Protocols][protocols]，不自行猜测 MLIR lowering 后的 ABI 槽位。[S6]

## 5. Expression API

### 5.1 layout、坐标和 partition

下表汇编 [DSL API][dsl]，按 wheel 的导出模块核对存在性；“存在”不代表任意 shape／dtype 组合有效。

| 任务 | API |
| --- | --- |
| 构造 | `make_shape`、`make_stride`、`make_coord`、`make_layout`、`make_ordered_layout`、`make_identity_layout` |
| 检查 | `size`、`cosize`、`rank`、`depth`、`get_shape`、`get_stride` |
| 代数 | `composition`、`complement`、`right_inverse`、`coalesce`、`recast_layout` |
| Divide | `logical_divide`、`zipped_divide`、`tiled_divide`、`flat_divide` |
| Product | `logical_product`、`zipped_product`、`tiled_product`、`flat_product`、`raked_product`、`blocked_product` |
| 坐标／视图 | `crd2idx`、`idx2crd`、`slice`、`make_view`、tensor 下标与 `.shape.unpack()` |
| 单项读取 | `fx.get_(tuple_value, mode).unpack()`，或 `IntTuple[].unpack()`；旧 `fx.get` 已弃用 |

传给 `recast_layout` 的第二、三个参数是 old／new type 的位宽；
按 wheel 中的实际签名 `recast_layout(layout, old_type_bits, new_type_bits)` 使用，
不要仅根据文档简写将 dtype 类直接传入。

### 5.2 内存、copy 与 MMA

| 任务 | API 与约束 |
| --- | --- |
| 寄存器 fragment | `make_fragment_like(tensor, dtype=None)`、`make_fragment_layout_like`、`make_rmem_tensor` |
| Load／store | tensor 的 `.load()`／`.store(vec)`；或 `memref_load`／`memref_store`／vector 版本 |
| Copy atom | `make_copy_atom(op_type, elem_type)`、`UniversalCopy32b()`／`UniversalCopy128b()` |
| Tiled copy | `make_tiled_copy_tv(atom, thr_layout, val_layout)`；`get_slice(tid)`、`partition_S/D`、`retile` |
| Copy 执行 | `copy(atom, src, dst, pred=mask)`、`copy_atom_call(atom, src, dst, pred=mask)` |
| MMA | `make_mma_atom(mma_op_type)`、`make_tiled_mma`、`gemm(atom, d, a, b, c)` |
| Per-thread MMA | `ThrMma.partition_A/B/C` |
| Indexed copy | `gather`／`scatter`；按同版本源码的 offset tensor layout 与 predicate 合同 |

`CopyAtom`／`MmaAtom` 定义在 `flydsl.expr.typing`；
`ThrCopy`／`ThrMma` 和 tiled helper 位于 `derived`。
对边界 tile 显式构造 predicate；vectorized copy 还须满足元素宽度和真实对齐要求，
不能只检查 vector 的首元素坐标。[S1]

### 5.3 GPU、算术与 vector

线程／block 信息使用 `fx.thread_idx.x`、`fx.block_idx.x`、`fx.block_dim`、
`fx.grid_dim`；workgroup 同步使用 `fx.gpu.barrier()`。
MI308X 的 wavefront 通常为 64 lanes，block 配置仍须匹配选定 kernel。

数值计算使用 DSL 类型与运算符；转换使用 `Numeric.to(dtype)`／`Vector.to(dtype)`。
原始 `ArithValue` 不是同一个类型合同，不能假定具有 Numeric 的 `.to()`。
`as_ir_value` 是 DSL → raw IR 的边界；直接 `flydsl._mlir.dialects.*` 调用不承诺跨版本稳定性。

```python
import flydsl.expr as fx
from flydsl.expr.typing import Vector, ReductionOp, full

import flydsl.compiler as flyc

@flyc.kernel
def vector_value_example():
    values = full((4,), 1.0, fx.Float32)
    total = values.reduce(ReductionOp.ADD)
    converted = values.to(fx.Float16)
```

这是 vector 值构造片段，不包含输出写回／完整 harness，不能作为数值验证候选。
`Vector.from_elements(elements, dtype)`、`Vector.filled(shape, value, dtype)`、
下标、`.bitcast(dtype)` 与 `.store(memref, indices)` 都在当前实现中提供。
旧 `flydsl.expr.vector` 模块自正式 v0.3.0 起删除；从 `typing` 取得这些接口。[S1、S7]

### 5.4 Buffer／ROCDL

```python
import flydsl.expr as fx

@flyc.kernel
def buffer_view_example(A: fx.Tensor):
    buffered_A = fx.rocdl.make_buffer_tensor(A)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
```

此片段仅说明入口；执行 copy 仍需正确的 source／destination layout、dtype、predicate。
`make_buffer_ptr`／`get_buffer_rsrc` 使用 Pointer 合同，
不是旧 `buffer_ops.create_buffer_resource` 的同签名替换。

| 接口 | 0.3.4／MI308X 边界 |
| --- | --- |
| `rocdl.make_buffer_tensor`、`make_buffer_ptr`、`get_buffer_rsrc` | wheel 提供；遵守 Tensor／Pointer 参数合同 |
| `rocdl.BufferCopy32b/128b` | wheel 提供；是否能选指令取决于目标、layout、dtype 和对齐 |
| `rocdl.MFMA(m,n,k,elem_ty_ab,elem_ty_acc=None)` | CDNA MMA atom；选择 gfx942 支持的组合 |
| `rocdl.WMMA`、`WMMAScale`、TDM | 用于相应 RDNA 目标；本指南不把它们列为 MI308X 指令方案 |
| `rocdl.BufferCopyLDS64b` | 官方列为弃用且会拒绝；不要用它推导一个可用 8-byte LDS DMA |
| `flydsl.expr.buffer_ops` | wheel 不提供；v0.3.0 起迁至仓库 `kernels.common.buffer_ops` |

旧 `buffer_load`／`buffer_store` 函数在仓库迁移后的文件中仍存在，
但该文件不随该版本 wheel 安装。官方 0.3.4 Kernel Guide 的速查表仍残留
`buffer_ops.buffer_load`，该项不进入本指南的 wheel 可用清单。[S1、S4、S5、S7]

## 6. 控制流与编译时值

- `fx.Constexpr[T]` 声明 specialization；`fx.const_expr` 标记编译时表达式。
- `fx.range_constexpr(...)` 用于编译时展开；用限定名调用，避免示例漏导入。
- Python `range(runtime_value)` 由 AST rewriter 降为运行时循环。
- 动态 `if` 分支中需要向后传递的值，应在所有路径定义，并保持相容的 DSL 类型和数量。
  `None` 不能冒充另一分支的数值结果。
- 运行时 shape、stride 和 offset 使用对应 DSL 值；Python 模块层只保存普通配置，
  不提前执行需要追踪上下文的 IR 操作。

遇到控制流类型错误，用原始 traceback 定位值产生位置，再检查 branch／loop-carried 值。
AST diff 的使用见第 10 节。[S2、S7]

## 7. Shared memory／LDS 与 Context

新代码使用 `fx.SharedAllocator` 和 typed storage，分配器只能在 `@kernel` body 中创建。

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def lds_example():
    @fx.struct
    class SharedStorage:
        data: fx.Array[fx.Float32, 256, 16]

    shared = fx.SharedAllocator().allocate(SharedStorage).peek()
    view = shared.data.view(fx.make_layout(256, 1))
```

这是分配／view 片段；共享数据的跨线程读写还需要正确索引和 barrier。

- 默认 `static=True`，每个 leaf 使用静态 LDS global，通常不填写 launch 的 `smem`。
- `static=False` 使用动态 shared memory；`smem=None` 可由 launcher 推断使用量。
- MI308X／gfx942 的 LDS 容量与 occupancy 必须按实际 kernel／设备预算确认；
  官方指南给出的每 CU 64 KiB 不意味着一个 block 可以忽略其他资源限制。
- 旧 `flydsl.utils.smem_allocator.SmemAllocator` 仍在 wheel 中。
  它保留手工 byte-offset、`_align`、`finalize`、`get_base` 路径与 `SmemPtr`，
  但当前类没有旧指南使用的 `allocate_array` 方法。
  因此不能照搬旧 LDS 示例；新代码使用上面的 SharedAllocator。
  维护未迁移代码时按实际实现核对 GPU module 插入点和 raw MLIR 类型。

Context 生命周期：
`T.f16`／`T.bf16`／`T.f32` 等是访问时构造 MLIR type 的属性，
不是 `T.f32()` 形式，也不是可在普通模块层随意缓存的 dtype 常量。
在正在追踪的 `@kernel`／`@jit` body 中构造所需 type／attribute；
模块层可以保留 `fx.Float32` 类、字符串和整数配置。
使用 FlyDSL 管理的 Context；不能把另一个手动 Context 中的 Type／Value 带入后续 JIT。
首次 `forward` 才出现的 Context 错误，不能靠导入检查排除。[S6、S7、S8]

## 8. Autotuning

wheel 提供 `from flydsl import Config, autotune`，
以及 `flydsl.autotune.do_bench(fn, warmup=5, rep=25, quantiles=None)`。
它们属于当前版本存在的工具；不因此获得所有 compiler 公共接口的跨版本稳定性保证。

`Config` 描述一次候选配置，`autotune(configs=..., key=...)` 包装 JIT launcher。
key 覆盖影响 specialization／选择的输入特征；
有原地写入时，按 API 声明 reset／restore 或 hook，防止某配置污染后续验证。

0.3.4 中 `FLYDSL_AUTOTUNE=1` 才要求重新搜索；
`FLYDSL_AUTOTUNE_CONFIG_DIR` 的非空目录用于离线配置制品。
不再沿用旧指南“所有结果必定在 ~/.flydsl/autotune/”的假设。
内部 do_bench 的耗时不代替 Agate 正式数值和性能回执；调优规模须在该版本任务预算内。

## 9. 编译缓存

内存与磁盘缓存按编译相关参数复用 artifact。缓存身份涉及源码／依赖／closure、
参数 signature 和 compiler／toolchain；精确实现见固定版本
[环境选项][env-source]、[JIT 实现][jit-source]。
本指南不把源码目录路径或一次 cache hit 当作正确性证据。

- 默认磁盘根目录：`~/.flydsl/cache`；可以用 `FLYDSL_RUNTIME_CACHE_DIR` 指定。
- `FLYDSL_RUNTIME_ENABLE_CACHE=0` 在该进程中关闭缓存使用。
- `FLYDSL_RUNTIME_RUN_ONLY=1` 跳过 JIT、仅用 AOT 缓存；miss 会失败，不能用来验证新候选。
- 并行 Job／进程保持各自需要的输入与运行身份；不得为调试某个候选删除其他任务的共享缓存。
- 缓存 artifact 的有效性不能替代 shape／dtype／stream 的数值验证。

## 10. 调试与错误定位

在有权运行的同一 worker／Job 依赖环境中按顺序定位：

1. import 阶段：检查实际 wheel 版本、包文件与有效导入路径。
2. constructor 阶段：检查 Model 输入配置、普通 Python 类型和 Context 是否被过早使用。
3. 首次 forward／JIT：检查 tracing、layout、控制流 yield、Context 与指令 selection。
4. 数值阶段：检查 mask、索引、dtype、stride、同步和输出语义。
5. 性能阶段：在数值通过后比较该版本正式测量结果。

`FLYDSL_DUMP_IR=1`、`FLYDSL_DUMP_DIR` 可输出 IR，
`FLYDSL_DEBUG_AST_DIFF=1` 可观察重写，
`FLYDSL_DEBUG_SHOW_STACKTRACE=1` 可取得完整编译 traceback。
具体变量见下一节。

0.3.4 `fx.printf` 的签名是 `printf(*args, format_str="")`；
按它的实现／目标支持情况使用 format 参数，不照搬旧指南把 format string 当第一个位置实参的例子。
IR dump 文件名与 pass 输出按当次实现读取，不承诺旧指南的固定编号目录或必有 `final_isa.s`。

Agate 编译检查通常只 import 并构造 Model，不能保证 first-launch JIT 已执行。
有源码错误时保留具体 traceback／编译诊断；`timeout`／`infra`／未知执行状态
应先检查原 Job 的回执和执行阶段，不从“没拿到数值结果”推断候选数值失败。

## 11. 环境变量

下表是该 wheel 的 `utils/env.py` 声明默认值，不是当前远端进程的环境转储。

| 变量 | 默认值／用途 |
| --- | --- |
| `FLYDSL_COMPILE_OPT_LEVEL` | 2，范围 0–3 |
| `FLYDSL_COMPILE_BACKEND` | rocm |
| `ARCH` | 空，compiler target override |
| `COMPILE_ONLY` | false，只编译，不执行 |
| `FLYDSL_DUMP_IR`、`FLYDSL_DEBUG_DUMP_ASM` | false |
| `FLYDSL_DUMP_DIR` | ~/.flydsl/debug |
| `FLYDSL_DEBUG_AST_DIFF`、`FLYDSL_DEBUG_PRINT_AFTER_ALL` | false |
| `FLYDSL_DEBUG_LOG_LEVEL` | WARNING |
| `FLYDSL_DEBUG_ENABLE_VERIFIER` | true |
| `FLYDSL_DEBUG_SHOW_STACKTRACE` | false |
| `FLYDSL_RUNTIME_KIND` | rocm，需与 compile backend 相容 |
| `FLYDSL_RUNTIME_CACHE_DIR` | ~/.flydsl/cache |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | true |
| `FLYDSL_RUNTIME_RUN_ONLY` | false |
| `FLYDSL_AUTOTUNE_CONFIG_DIR` | 空；非空时配置制品目录 |

`get_rocm_arch()` 的探测顺序为 `FLYDSL_GPU_ARCH`、
`HSA_OVERRIDE_GFX_VERSION`、`rocm_agent_enumerator`，最后回退 gfx942。
该回退是 best effort，不能证明真实设备就是 gfx942；
compiler 的 `ARCH` override 是另一个配置入口，应保持两者与实际设备一致。
候选向 Agate 传环境选项时遵守服务端支持的字段／白名单；
本表不赋予任意改远端环境或换目标设备的权限。

## 12. 构建与安装

该版本使用已发布 wheel，CPython ABI 和平台必须匹配上表；
ROCm／PyTorch／runtime 库仍由实际 worker 环境提供。
AKA 通过 Task 冻结的 requirements 声明 wheel，不依赖 Agent 本地装同一包才能准备远端输入。

在独立 Linux 环境中，复现这份安装身份的 requirements 行是：

```text
https://files.pythonhosted.org/packages/b2/af/ee8dd02aa3809cb39a83212399b2f924ca4baf6aa7b4f933602856175e31/flydsl-0.3.4-cp312-cp312-manylinux_2_27_x86_64.whl#sha256=280d9fb8830c0ccfb65e1cad3a6bb57f8b9c8a74b5352dafeebd3b4041425ed1
```

需要开发 FlyDSL 或运行仓库 kernels 时，使用固定 commit 的源码 checkout，
按 [Install FlyDSL][installation] 准备 LLVM／MLIR／ROCm 后构建；
将仓库源码、附属文件和导入路径作为明确依赖处理。
`pip install -e .` 与已发布 wheel 是不同安装模式，不用前者的成功证明后者包含仓库 kernels。

`fly-opt` 用于对 .mlir 运行 pass，例如 `fly-opt --help`；
先确认当前环境确有此 CLI。纯 wheel 已可导入不等于所有构建工具／源码测试脚本都已安装。

## 13. 完整 VecAdd 示例

下面保留官方 [01-vectorAdd.py][vecadd] 的 kernel／launcher body，
将顶层 demo 改为 Agate 可调用的 `torch.nn.Module.Model.forward(A,B)`。
适用输入：同 GPU、同 shape 的非空 contiguous float32 二维矩阵，N 为 4 的倍数，
输入基地址按 16 bytes 对齐。
它保留 tiled predicate，M 不必是 block tile 的倍数。
对其他 dtype、stride、N 尾部粒度或算子签名，应另外实现并验证。

示例已完成语法、导出与源码比对；未在 Agate GPU worker 实测。
正式使用时，reference 与 input generator 必须产生上述输入，
用第 14 节的流程取得实际数值结果。

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Vectorized, predicated, target-neutral 2D elementwise add (C = A + B).

This example is **target-neutral**: it uses only the backend-agnostic ``flydsl.expr`` API, so it
supports on any backend.

Highlights:
  1. **float4 vectorization** via ``UniversalCopy128b`` -- each copy atom moves 128 bits
     (4 x f32) along the contiguous (N) axis, so every thread loads/stores one ``float4``.
  2. **Predicated OOB masking**: the (M, N) shape need not be a multiple of the block tile,
     so border blocks have threads whose float4 lies past the tensor. A per-atom boolean
     predicate (``coord < (M, N)``) gates each copy, so a load/store never touches OOB memory.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,
):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    # Identity (coordinate) tensor: value == logical coord (m, n).
    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))

    TileMN = tiled_copy.tile_mn

    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]

    thr_copy = tiled_copy.get_slice(tid)

    thr_gA = thr_copy.partition_S(gA)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]

    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rB = fx.make_fragment_like(thr_gB)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)

    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)

    thr_rC.store(thr_rA.load() + thr_rB.load())

    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)


@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),
        fx.make_ordered_layout((1, 4), order=(0, 1)),
    )
    tile_m, tile_n = tiled_copy.tile_mn.unpack()

    M, N = A.shape.unpack()
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    vector_add_kernel(A, B, C, tiled_copy).launch(grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream)

class Model(torch.nn.Module):
    """C = A + B for contiguous float32 matrices with N divisible by four."""

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        if A.ndim != 2 or B.shape != A.shape:
            raise ValueError("A and B must have the same two-dimensional shape")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise ValueError("this example supports float32")
        if not A.is_cuda or B.device != A.device:
            raise ValueError("A and B must be on the same GPU")
        if not A.is_contiguous() or not B.is_contiguous():
            raise ValueError("this example requires contiguous tensors")
        if A.data_ptr() % 16 or B.data_ptr() % 16:
            raise ValueError("this example requires 16-byte aligned inputs")
        if A.shape[0] == 0 or A.shape[1] == 0 or A.shape[1] % 4:
            raise ValueError("M and N must be positive; N must be divisible by four")
        C = torch.empty_like(A)
        vector_add(A, B, C, stream=torch.cuda.current_stream(A.device))
        return C


if __name__ == "__main__":
    A = torch.randn(100, 1000, dtype=torch.float32, device="cuda")
    B = torch.randn_like(A)
    C = Model()(A, B)
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A + B)
    print("PASS")
```

## 14. 测试、benchmark 与仓库 kernel 参考

验收按三层完成，结果分别记录：

| 层次 | 完成标准 |
| --- | --- |
| 安装／API | 同 Job 环境中安装身份符合要求，导入路径有效；只能证明准备阶段 |
| GPU 数值 | 调用真实 forward；全部声明 shape 和 dtype 获得输出并按 reference 验证 |
| 性能／最终 gate | 数值通过后完成规定测量、阈值、证据绑定与最终 gate |

Job 已启动、Running 或 Agate command 成功返回不等于候选数值通过；
检查其具体 compile／correctness／performance 结果和 skipped／failed shape。
保存 Job 身份与有界诊断，避免拿旧轮结果证明新源码。

上游 `scripts/run_tests.sh`、MLIR lit、Python IR／GPU tests 和
`scripts/run_benchmark.sh` 需要源码 checkout 和相关构建／硬件条件；
`tests.test_common.perftest`／`checkAllclose` 也不是 wheel 的 API。
查看固定版本 [Testing & Benchmarking][testing] 获取源码测试方式。

第三份 [Kernels API][kernels] 在本指南中作为**源码参考目录**使用：

| 类型 | v0.3.4 源码模块 |
| --- | --- |
| GEMM | `kernels.gemm.preshuffle_gemm`、`mxfp4_preshuffle`、`fp4_gemm_4wave` |
| MoE | `kernels.moe.moe_gemm_2stage`、`mxfp_moe` |
| Attention | `kernels.attention.pa_decode_fp8` |
| Norm／Softmax | `kernels.norm.layernorm_kernel`、`rmsnorm_kernel`、`softmax_kernel` |
| 公共 helper | `kernels.common.kernels_common`、`layout_utils`、`mma.mfma_preshuffle_pipeline` |

gfx950 FP4、RDNA WMMA／TDM 等专用方案不能直接套用到 MI308X；
上表不保证各 builder 或每种配置在所有 GPU 上可运行。
旧指南的 `kernels.layernorm_kernel` 等路径按新目录结构定位，
并核对 fixed source 的函数签名；库源码还可能导入未在上表列出的附属 helper。

## 固定来源

所有 S 编号来源固定到 commit `6bb37ba4e23c3a2729e3af244251b55fc5938237`：

- S1：[DSL API][dsl]。
- S2：[Compiler API][compiler]。
- S3：[Kernels API][kernels]，内容属于源码仓库范围。
- S4：[API Stability][stability]，存在性、稳定性与架构可运行性分别判断。
- S5：[setup.py][setup]，wheel／sdist 的 package 收集范围。
- S6：[DSL Protocols][protocols]、[Arithmetic Types][arithmetic]、[Storage and Allocator][storage]。
- S7：[Kernel Authoring Guide][authoring]；速查表的旧 buffer 入口不沿用。
- S8：当前 wheel 中 `typing.py`、`gpu.py` 和 compiler Context／launcher 实现，与对应标签源码逐字一致。

[dsl]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/api/dsl.rst
[compiler]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/api/compiler.rst
[kernels]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/api/kernels.rst
[stability]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/api_stability.md
[setup]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/setup.py
[protocols]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/language/dsl_protocols.md
[arithmetic]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/language/arithmetic_types.md
[storage]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/language/storage_and_allocator.md
[rocm-backend]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/python/flydsl/compiler/backends/rocm.py
[env-source]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/python/flydsl/utils/env.py
[jit-source]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/python/flydsl/compiler/jit_function.py
[authoring]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/kernel_authoring_guide.md
[installation]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/installation.rst
[vecadd]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/examples/01-vectorAdd.py
[testing]: https://github.com/ROCm/FlyDSL/blob/6bb37ba4e23c3a2729e3af244251b55fc5938237/docs/testing_benchmarking_guide.md

