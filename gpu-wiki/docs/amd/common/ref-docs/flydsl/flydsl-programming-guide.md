# FlyDSL Programming Guide

Applicability: backend: flydsl; hardware: amd; topic: reference

FlyDSL (**F**lexible **l**ayout p**y**thon **DSL**) is a Python DSL based on MLIR for writing high-performance AMD GPU kernels, implementing CuTe layout algebra. It compiles to ROCDL/HSACO binaries via the Fly dialect.

> **Source Project**: [FlyDSL](https://github.com/ROCm/FlyDSL), licensed under Apache 2.0

---

## 1. Project Overview

### 1.1 Positioning

| Dimension | Description |
|------|------|
| **Language** | Python DSL + MLIR compilation stack |
| **Target Hardware** | AMD GPU: gfx942 (MI300X), gfx950 (MI350/MI355X), gfx90a (MI250X), gfx1250 (RDNA) |
| **Core Concept** | AMD implementation of CuTe layout algebra: (Shape, Stride) pairs describe data layout, enabling tiling/partitioning via algebraic operations |
| **Compilation Path** | Python → Fly MLIR dialect → ROCDL → LLVM → HSACO binary |
| **Version** | 0.1.1 |

### 1.2 Comparison with CuTe/CUTLASS

| Dimension | CuTe C++ (CUTLASS) | FlyDSL |
|------|-----|---------|
| Language | C++ templates | Python + MLIR |
| Hardware | NVIDIA CUDA | AMD ROCm/HIP |
| IR Backend | C++ → CUDA/PTX | Fly MLIR → ROCDL → HSACO |
| Wave/Warp Size | 32 threads (warp) | 64 threads (wavefront) |
| Memory Hierarchy | GMEM → SMEM → RMEM | GMEM → LDS → VGPR |
| Matrix Instructions | HMMA/GMMA | MFMA |

### 1.3 Project Structure

```
FlyDSL/
├── python/flydsl/ # Python DSL core
│ ├── compiler/ # JIT compilation(@flyc.jit, @flyc.kernel)
│ ├── expr/ # API(layout , arith, vector, gpu, rocdl)
│ ├── runtime/ # row(GPU )
│ └── utils/ # tool(SmemAllocator, , variable)
├── include/flydsl/ # C++ Fly file
├── lib/ # C++
├── kernels/ # kernel
├── examples/ # rowexample(vectorAdd, tiledCopy, tiledMma, GEMM)
└── tests/ #
```

---

## 2. Compilation Pipeline

### 2.1 High-Level Flow

```
Python function (@flyc.kernel / @flyc.jit)
        │
 ▼ AST (for/if -> scf.for/scf.if)
 conversion Python function
        │
 ▼ Tracing( MLIR Context execute)
 MLIR Module (gpu, arith, scf, memref )
        │
 ▼ MlirCompiler.compile - 14 stage pass pipeline
   ┌──────────────────────────────────────────┐
   │  1. gpu-kernel-outlining                 │ → gpu.func
 │ 2. fly-canonicalize │ -> FlyDSL
 │ 3. fly-layout-lowering │ -> layout
 │ 4. convert-fly-to-rocdl │ -> Fly ops -> ROCDL
 │ 5. canonicalize │ -> standard MLIR
   │  6. convert-scf-to-cf + convert-gpu-to-rocdl │
   │  7. rocdl-attach-target{chip=gfxNNN}     │
   │  8-13. SCF/CF/GPU/Arith/Func → LLVM     │
 │ 14. gpu-module-to-binary{format=fatbin} │ -> HSACO
   └──────────────────────────────────────────┘
        │
        ▼
 JITCFunction(ExecutionEngine )
```

### 2.2 JIT Compilation Flow

1. **Cache Check**: Look up by parameter type signature (memory → disk)
2. **AST Rewriting**: Python `for`/`if` → MLIR `scf.for`/`scf.if`
3. **MLIR Module Creation**: `gpu.container_module` + target architecture
4. **Parameter Conversion**: Python parameters → IR types (Tensor → memref via DLPack)
5. **Function Tracing**: Execute function body to generate MLIR ops
6. **GPU Kernel Launch**: `@kernel` call generates `gpu.func`
7. **Pipeline Compilation**: 14-stage pass pipeline
8. **Execution**: `JITCFunction` wraps MLIR ExecutionEngine
9. **Cache Storage**: Serialize compiled results to disk

---

## 3. Core API

### 3.1 `@flyc.kernel` — GPU Kernel Definition

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def vec_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    idx = bid * 256 + tid
    # ... kernel body ...
```- Can only be called inside the `@flyc.jit` function
- The call returns `KernelLauncher`, which must be dispatched using `.launch()`
- Generates `gpu.func` + `gpu.kernel` attributes

### 3.2 `@flyc.jit` — Host-side Launcher

```python
@flyc.jit
def vec_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    vec_add_kernel(A, B, C, N).launch(
        grid=(N // 256,),
        block=(256,),
        stream=stream,
    )
```

- The first call triggers compilation; subsequent calls with the same type signature use the cache
- `Constexpr[T]` parameters become compile-time constants (affecting the cache key)
- Supports CUDA Graph capture

### 3.3 Launch Configuration

```python
kernel_fn(args...).launch(
 grid=(num_blocks_x, num_blocks_y, num_blocks_z), # 1-3
 block=(threads_x, threads_y, threads_z), # 1-3
 smem=shared_mem_bytes, # sharedmemory
    stream=stream,                                      # CUDA/HIP stream
)
```

Grid and block dimensions accept `int` (static) or `ir.Value` (dynamic), with missing dimensions defaulting to 1.

---

## 4. Parameter Types

| Type | Description | Host-side Mapping |
|------|-------------|-------------------|
| `fx.Tensor` | GPU tensor, converted to memref via DLPack | `torch.Tensor` auto-conversion |
| `fx.Constexpr[T]` | Compile-time constant, embedded in IR | Different values produce different compilation results |
| `fx.Int32` | Runtime i32 parameter | Python `int` auto-conversion |
| `fx.Stream` | CUDA/HIP stream | `torch.cuda.Stream` |

### Custom Parameter Types

```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyType, dsl_type=MyDslType)
class MyAdaptor:
 def __fly_types__(self): return [...] # MLIR type
 def __fly_ptrs__(self): return [...] # ctypes pointer
```

### DslType / JitArgument Protocol

```python
class DslType(Protocol):
    @classmethod
    def __fly_construct__(cls, values: List[ir.Value]) -> "DslType": ...
    def __fly_values__(self) -> List[ir.Value]: ...

class JitArgument(Protocol):
    def __fly_types__(self) -> List[ir.Type]: ...
    def __fly_ptrs__(self) -> List[ctypes.c_void_p]: ...
```

---

## 5. Expression API (`flydsl.expr`)

### 5.1 Thread/Block Hierarchy

```python
from flydsl.expr import gpu

# (returns Int32)
tid_x = gpu.thread_idx.x  # .y, .z
bid_x = gpu.block_idx.x   # .y, .z
bdim_x = gpu.block_dim.x
gdim_x = gpu.grid_dim.x

# synchronousbarrier
gpu.barrier # workgroup s_barrier
```

### 5.2 Arithmetic Operations (`fx.arith`)

```python
from flydsl.expr import arith

# constant(recommended DSL type)
c42 = fx.Index(42) # index type
c3_14 = fx.Float32(3.14)    # f32
mask = fx.Int32(0xFF)       # i32

# English note
result = a + b
result = a * 2
result = a // 4
result = a % 16

# English note
result = arith.select(cond, true_val, false_val)

# bit
result = arith.andi(a, b)
result = arith.xori(a, b)
result = arith.shli(a, b)
```

### 5.3 Vector Operations (`fx.vector`)

```python
from flydsl.expr import vector

vec = vector.from_elements(vec_type, [a, b, c, d])
vector.store(vec, memref, [idx])
elem = vector.extractelement(vec, idx)
```

### 5.4 Buffer Operations (`fx.buffer_ops`)

AMD buffer load/store intrinsics for efficient global memory access:

```python
from flydsl.expr import buffer_ops

rsrc = buffer_ops.create_buffer_resource(memref_value)
data = buffer_ops.buffer_load(rsrc, byte_offset, vec_width=4)
buffer_ops.buffer_store(data, rsrc, byte_offset)
```### 5.5 ROCm Built-ins (`fx.rocdl`)

```python
from flydsl.expr import rocdl

# Buffer tensor
A_buf = rocdl.make_buffer_tensor(A)

# MFMA
result = rocdl.mfma_f32_16x16x16f16(result_type, [a, b, acc])
result = rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, [a, b, acc])
result = rocdl.mfma_i32_16x16x32_i8(result_type, [a, b, acc])

# GFX950 scaled MFMA (MXFP4/FP6/FP8)
result = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    result_type, [a, b, acc, cbsz, blgp, opselA, scaleA, opselB, scaleB]
)

# schedulingbarrier
rocdl.sched_mfma(cnt) # wait MFMA complete
rocdl.sched_vmem(cnt) # wait VMEM complete
rocdl.sched_dsrd(cnt) # wait LDS complete
rocdl.sched_dswr(cnt) # wait LDS complete

# ( VALU)
result = rocdl.exp2(T.f32, x)   # v_exp_f32
result = rocdl.rcp(T.f32, x)    # v_rcp_f32

# Warp shuffle
val = rocdl.ds_bpermute(idx, src)
```

### 5.6 Copy/MMA Atom Types

| Type Factory | Description |
|--------------|-------------|
| `fx.UniversalCopy128b()` | Generic 128-bit copy |
| `fx.UniversalCopy64b()` | Generic 64-bit copy |
| `fx.UniversalCopy32b()` | Generic 32-bit copy |
| `fx.rocdl.BufferCopy128b()` | AMD buffer 128-bit copy |
| `fx.rocdl.BufferCopy64b()` | AMD buffer 64-bit copy |
| `fx.rocdl.BufferCopy32b()` | AMD buffer 32-bit copy |
| `fx.rocdl.MFMA(m, n, k, elem_ty)` | MFMA MMA atom |

---

## 6. Control Flow

### 6.1 Loops

`ASTRewriter` automatically converts Python loops into MLIR ops:

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
 # compilationloop
    for i in range_constexpr(N):
        ...

 # rowloop(-> scf.for)
    for i in range(runtime_value):
        ...
```

### 6.2 Compile-Time Constants

```python
from flydsl.expr import const_expr

tile_size = const_expr(N // 4)
for i in range_constexpr(tile_size):
    ...
```

---

## 7. Shared Memory (LDS)

### 7.1 SmemAllocator

```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.expr.typing import T

# create
allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")

# type
lds_a = allocator.allocate_array(T.f16, 8192)
lds_b = allocator.allocate_array(T.f16, 8192)

# Kernel : pointertype
lds_base = allocator.get_base()
lds_a_ptr = lds_a(lds_base)   # SmemPtr
lds_b_ptr = lds_b(lds_base)   # SmemPtr

# Load/Store
val = lds_a_ptr.load([idx])
lds_b_ptr.store(val, [idx])
```

### 7.2 Finalize

Emit `memref.global` inside the GPU module body:

```python
comp_ctx = CompilationContext.get_current()
with ir.InsertionPoint(comp_ctx.gpu_module_body):
    allocator.finalize()
```

### 7.3 LDS Capacity

| Architecture | GPU | LDS/CU |
|--------------|-----|--------|
| gfx942 | MI300X | 64 KB |
| gfx950 | MI350/MI355X | 160 KB |
| gfx90a | MI250X | 64 KB |

---

## 8. Autotuning

```python
from flydsl import Config, autotune

configs = [
    Config(num_warps=4, waves_per_eu=2, BLOCK_M=128, BLOCK_N=128),
    Config(num_warps=8, waves_per_eu=1, BLOCK_M=256, BLOCK_N=64),
]

@autotune(configs=configs, key=["M", "N", "K"])
@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, ...):
    ...
```

- `Config(num_warps=, waves_per_eu=, maxnreg=, **kwargs)` — Single tuning configuration
- `@autotune(configs=[...], key=[...])` — Auto-benchmark all configurations
- Results cached to `~/.flydsl/autotune/`
- `do_bench(fn, warmup=5, rep=25)` — Standalone GPU benchmark

## 9. Compilation Cache

### Automatic Caching

- **Memory Cache**: Indexed by parameter type signature
- **Disk Cache**: Stored in `~/.flydsl/cache/` (configurable)
- **Cache Key**: Source hash + dependency source + closure values + FlyDSL/LLVM version

### Invalidation Conditions

- Function source code or dependencies change
- Parameter type changes (different tensor shape/dtype)
- `Constexpr` values change
- FlyDSL or LLVM version changes

### Disabling the Cache

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_script.py
```

---

## 10. Debugging

### 10.1 IR Dump

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_script.py
```

Generates numbered `.mlir` files:

```
dumps/my_func_name/
├── 00_original.mlir
├── 01_gpu-kernel-outlining.mlir
├── ...
├── 14_gpu-module-to-binary.mlir
└── final_isa.s # AMD ISA
```

### 10.2 AST Diff

```bash
FLYDSL_DEBUG_AST_DIFF=1 python my_script.py
```

### 10.3 In-Kernel printf

```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

---

## 11. Environment Variables

### Compilation Options

| Variable | Default | Description |
|------|--------|------|
| `FLYDSL_COMPILE_OPT_LEVEL` | `2` | Optimization level (0-3) |
| `COMPILE_ONLY` | `0` | Compile only, do not execute |
| `ARCH` | Auto-detect | Target GPU architecture |

### Debugging Options

| Variable | Default | Description |
|------|--------|------|
| `FLYDSL_DUMP_IR` | `false` | Dump IR for each stage |
| `FLYDSL_DUMP_DIR` | `~/.flydsl/debug` | IR dump directory |
| `FLYDSL_DEBUG_DUMP_ASM` | `false` | Dump final ISA assembly |
| `FLYDSL_DEBUG_PRINT_AFTER_ALL` | `false` | Print IR after each pass |
| `FLYDSL_DEBUG_LOG_LEVEL` | `WARNING` | Log level |

### Runtime Options

| Variable | Default | Description |
|------|--------|------|
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Cache directory |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `true` | Enable cache |

### Architecture Detection Priority

`get_rocm_arch()` is detected in the following order:
1. `FLYDSL_GPU_ARCH` environment variable
2. `HSA_OVERRIDE_GFX_VERSION` (supports `9.4.2` → `gfx942`)
3. `rocm_agent_enumerator` system tool
4. Default: `gfx942`

---

## 12. Build and Installation

### Prerequisites

- ROCm 6.x / 7.x
- cmake ≥ 3.20, C++17 compiler
- Python 3.10+

### Build Steps

```bash
# 1. LLVM/MLIR( 30 )
bash scripts/build_llvm.sh -j64

# 2. FlyDSL
bash scripts/build.sh -j64

# 3. (mode)
pip install -e .

# 4. verification
bash scripts/run_tests.sh
```

### Troubleshooting

- **LLVM Version Conflict**: Rebuild after `unset MLIR_PATH`
- **`No module named flydsl`**: Run `pip install -e .` or set `PYTHONPATH`
- **Cache Issues**: Clear with `rm -rf ~/.flydsl/cache`

---

## 13. Complete Example: VecAdd

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

 # by block
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tA = fx.slice(tA, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))

    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tB = fx.slice(tB, (None, bid))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))

    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))
    tC = fx.slice(tC, (None, bid))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

 # register + copy atom
    RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1),
                              fx.AddressSpace.Register)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

@flyc.jit
def vectorAdd(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream,
    )

# use
n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()
vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())
```## 14. Testing and Benchmarking

### Test Categories

| Category | Location | Requires GPU | Description |
|------|------|----------|------|
| MLIR lit Tests | `tests/mlir/` | No | Fly dialect lowering verification |
| Python IR Tests | `tests/pyir/` | No | Python MLIR generation verification |
| GPU Kernel Tests | `tests/kernels/` | Yes | End-to-end compilation + execution |
| AOT Examples | `tests/python/examples/` | Depends | Pre-compiled examples |

### Running Tests

```bash
bash scripts/run_tests.sh # GEMM Correct
bash scripts/run_benchmark.sh # performance

# row
python tests/kernels/test_preshuffle_gemm.py --in_dtype fp8 -M 16 -N 5120 -K 8192
```

### Performance Measurement Tools

```python
from tests.test_common import perftest, checkAllclose

@perftest(num_iters=20, num_warmup=3)
def my_test(Input, Output):
    ...

# verification
err = checkAllclose(output, reference, rtol=1e-2, atol=1e-2)
```

---

## Related Documents

- [FlyDSL Layout Algebra](flydsl-layout-algebra.md) — Detailed reference for layout algebra
- [FlyDSL Pre-built Kernel Library](flydsl-prebuilt-kernels.md) — Production-grade kernel reference
- [AMD GPU Kernel Optimization Framework Overview](../amd-kernel-optimization-frameworks.md) — FlyDSL's positioning within the framework lineage
- [Fused MoE Optimization (FlyDSL)](../../../cdna3/mi308x/kernel-opt/flydsl/cdna3-fused-moe-flydsl.md) — MoE optimization case study on MI300X
- [AMD MFMA Matrix Core Programming Guide](../amd-mfma-matrix-cores.md) — Detailed reference for MFMA instructions
