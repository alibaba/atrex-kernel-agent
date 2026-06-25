# AMD GPU Kernel Optimization Framework Overview

## Framework Lineage

AMD GPU kernel development has multiple tiers of toolchains:

| Framework | Tier | Language | Use Cases |
|------|------|------|---------|
| **HIP** | Low-level | C++ | Handwritten high-performance kernels, precise hardware control |
| **Composable Kernel (CK)** | Mid-level | C++ Templates | Composable GEMM/Attention kernels |
| **FlyDSL** | High-level | Python (MLIR) | Rapid prototyping + production-grade performance |
| **TileLang** | High-level | Python (TVM) | Declarative tile programming |
| **Triton** | High-level | Python | General-purpose GPU programming |

---

## FlyDSL

### Positioning

FlyDSL (Flexible Layout Python DSL) is built on the MLIR compilation stack, using FLIR (Flexible Layout IR)—a CuTe-inspired (Shape, Stride) layout algebra.

### Compilation Pipeline

```
Python kernel → FLIR (Layout Algebra) → MLIR → Canonicalization/CSE 
→ GPU-to-ROCDL Lowering → gfx942/gfx950 Binary
```

### Key Features

1. **Python-native development**: Through the `flydsl` package, new kernels can go from prototype to testing to benchmarking in hours
2. **Hierarchical control**: Block → Warp → Thread → MFMA instruction-level mapping
3. **Layout algebra**: Composable (Shape, Stride) abstractions handle tiling, swizzling, and vectorization
4. **Multi-target**: The same code compiles to both gfx942 and gfx950

### Usage

```bash
export AITER_ENFORCE_DSL=1
export DSL2_ROOT=/opt/FlyDSL
export AITER_USE_FLYDSL_MOE=1
export MLIR_PATH=/opt/mlir_install
```

### Typical Performance

Fused MoE (tokens=16384, E=384, topk=8):

| Comparison | BF16 | W4A16 |
|------|------|-------|
| vs Triton | 1.39x speedup | 3.22x speedup |
| vs PyTorch | 13.8x speedup | 13.4x speedup |

---

## TileLang

### Positioning

TileLang uses "Tile" as the fundamental programming unit, providing declarative GPU programming that supports both AMD and NVIDIA.

### Core API

```python
# Kernel definition
with T.Kernel(dim_x, dim_y, threads=N) as (bx, by):
 # memory
    shared = T.alloc_shared([M, K], dtype)       # LDS
 frag = T.alloc_fragment([M, K], accum_dtype) # register

 # data
    T.copy(src, shared, coalesced_width=width)

 # compute
    T.gemm(A, B, C, transpose_B=True, k_pack=k, policy=GemmWarpPolicy.FullRow)

 # reduction
    T.reduce_max(tensor, result, dim=1)
    T.reduce_sum(tensor, result, dim=1)

 # pipeline
    for k in T.Pipelined(bound, num_stages=stages):
        ...
```

### JIT and Autotuning

```python
# JIT compilation
@tilelang.jit(out_idx=[3])
def kernel(Q, K, V, Output):
    ...

# Autotuning
@tilelang.autotune(configs=get_configs(), ...)
def kernel(Q, K, V, Output):
    ...
```

### MI300X Tuning Parameters

| Parameter | Purpose |
|------|------|
| `block_M` / `block_N` | Tile sizes for Q and KV |
| `threads` | Number of threads per block |
| `num_stages` | Number of pipeline stages |
| `qk_coalesced_width` | QK operation memory coalescing width |
| `v_coalesced_width` | V operation memory coalescing width |
| `k_pack` | Data packing factor |
| `panel_size` | Swizzle panel size |
| `enable_rasterization` | Memory rearrangement switch |

---

## Composable Kernel (CK)

### Positioning

CK is AMD's C++ template library that provides composable high-performance GEMM, Attention, and other kernels, similar to NVIDIA CUTLASS.

### TensorDescriptor System

CK uses a transform tree to manage data layouts:

```cpp
// English comment
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, K), make_tuple(K, 1));

// (Unmerge, Merge, PassThrough, Embed)
auto transformed = transform_tensor_descriptor(desc, transforms, lower_ids, upper_ids);

// computephysicaloffset
auto offset = transformed.CalculateOffset(make_multi_index(1, 3, 2));
```

### Four Core Transforms

- **Embed**: Multi-dimensional coordinates → linear address
- **Unmerge**: Split a dimension, e.g., (256) → (4, 64)
- **Merge**: Combine dimensions, e.g., (64, 128) → (8192)
- **PassThrough**: Identity transform

### Vectorized Memory

```cpp
vector_type<float, 16> buf;  // 16 VGPRs
auto dynamic_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(ptr, size);
auto data = dynamic_buf.Get<d4_t>(offset, true); // 128-bit vectorread
```### Compile-Time Loop Unrolling

```cpp
static_for<0, 4, 1>{}([&](auto i) {
 // compilation
});
```

---

## HIP Low-Level Optimization

### MFMA Programming

```c
// FP8 MFMA (CDNA4)
#include <hip/hip_runtime.h>

// directuse MFMA function
__device__ float4 __builtin_amdgcn_mfma_f32_16x16x128_fp8_fp8(
    long a, long b, float4 c, int cbsz, int abid, int blgp);
```

### Key Compiler Built-ins

```c
__builtin_amdgcn_s_barrier // wave barrier
__builtin_amdgcn_s_setprio(x) // wave (0-3)
__builtin_amdgcn_sched_barrier(x) // schedulingbarrier
__builtin_amdgcn_s_waitcnt(x) // wait
```

### Direct Global-to-LDS (CDNA4)

```c
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

// Buffer Resource Descriptor
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;  // 0x110000
};
```

---

## ROCm 7.2 Key Features

### Compilation Optimizations

- **ThinLTO**: Cross-file optimization, supporting function inlining, specialization, and dead code elimination
- **rocMLIR FP4 Support**: New precision format for CDNA4
- **FP8 Extension**: Full compiler and graph stack support

### hipBLASLt Improvements

- Swizzle A/B optimized memory access patterns
- GEMM tuning for FP8/BF16/FP16 on MI300X/MI350/MI355
- Log restore (restore-from-log) for reproducible performance

### CDNA4 New Features

- FP4 support (rocMLIR, MIGraphX)
- Node Power Management (MI355X/MI350X)
- SR-IOV virtualization support

---

## Framework Selection Guide

| Scenario | Recommended Framework | Rationale |
|------|---------|------|
| Rapid kernel prototyping | TileLang / Triton | Fastest development speed |
| Production-grade MoE/GEMM | FlyDSL | Balances development efficiency and performance |
| Extreme performance tuning | HIP + built-in functions | Full hardware control |
| Composable kernel library | CK | Template-based, reusable |
| Cross-platform porting | Triton | Dual AMD/NVIDIA support |
